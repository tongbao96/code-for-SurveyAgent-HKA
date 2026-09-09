from __future__ import annotations

import csv
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .llm import LLMClient
from .models import (
    Outline,
    Paper,
    SurveyDraft,
    SurveySection,
    normalize_doi,
    normalize_title,
)
from .progress import ProgressCallback, report
from .prompts import draft_subsection


CITATION_PLAN_FIELDS = (
    "ref_no",
    "section_order",
    "section_title",
    "subsections",
    "rank_in_section",
    "paper_key",
    "paper_id",
    "doi",
    "title",
    "source",
    "year",
    "selected_stage",
)


def draft_survey(
    topic: str,
    outline: Outline,
    papers_by_section: Mapping[str, Sequence[Paper]],
    llm: LLMClient,
    max_workers: int = 4,
    max_context_characters: int = 140_000,
    max_attempts: int = 4,
    citation_plan: Optional[Sequence[Mapping[str, Any]]] = None,
    progress: ProgressCallback = None,
) -> SurveyDraft:
    plan = (
        list(citation_plan)
        if citation_plan is not None
        else build_citation_plan(outline, papers_by_section)
    )
    usable_papers, skipped_count = _filter_papers_by_citation_plan(
        papers_by_section, plan
    )
    if skipped_count:
        report(
            progress,
            f"    skipped {skipped_count} paper record(s) excluded by the citation plan",
        )
    bibliography, numbers = _citation_plan_indexes(plan, usable_papers)
    tasks: List[Tuple[str, str, Sequence[Paper]]] = []
    empty_sections = 0
    for section in outline.outline:
        papers = _fit_context(
            usable_papers.get(section.section_title, []), max_context_characters
        )
        if not papers:
            empty_sections += 1
            continue
        subsections = list(section.subsections) or [section.section_title]
        tasks.extend((section.section_title, subsection, papers) for subsection in subsections)
    if empty_sections:
        report(
            progress,
            f"    skipped {empty_sections} section(s) with no usable paper evidence",
        )
    if not tasks:
        raise RuntimeError("No survey sections have usable paper evidence")

    results: List[SurveySection | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _draft_one,
                topic,
                outline,
                section,
                subsection,
                papers,
                numbers,
                llm,
                max_attempts,
            ): index
            for index, (section, subsection, papers) in enumerate(tasks)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            section, subsection, _ = tasks[index]
            report(
                progress,
                f"    drafted subsection {completed}/{len(tasks)}: {section} / {subsection}",
            )
    return SurveyDraft(
        title=outline.title,
        sections=[section for section in results if section],
        bibliography=bibliography,
    )


def _draft_one(
    topic: str,
    outline: Outline,
    section: str,
    subsection: str,
    papers: Sequence[Paper],
    numbers: Dict[str, int],
    llm: LLMClient,
    max_attempts: int,
) -> SurveySection:
    if not papers:
        raise ValueError(f"No ranked reference evidence is available for section: {section}")
    error = ""
    allowed_numbers = sorted(numbers[paper.key] for paper in papers)
    allowed_tokens = ", ".join(f"ref [{number}]" for number in allowed_numbers)
    for _ in range(max_attempts):
        prompt = draft_subsection(topic, outline, section, subsection, papers, numbers)
        if error:
            prompt += f"""

Your previous response failed validation: {error}
Return a complete replacement JSON object, not a correction note.
The subsection content must contain at least one citation token copied exactly
from this allowed list: {allowed_tokens}
Place citations inside the content itself, with at least one citation in every paragraph.
Do not use any other reference number."""
        response = llm.json(prompt)
        content = _normalize_citations(
            _response_content(response, subsection), allowed_numbers
        )
        try:
            references = _validate_references(
                content, response.get("references", []), papers, numbers
            )
            if len(content.split()) < 300:
                raise ValueError("subsection is shorter than the requested 300 words")
            return SurveySection(section, subsection, content, references)
        except ValueError as exc:
            error = str(exc)
    raise ValueError(f"Invalid writer output for '{subsection}': {error}")


def _response_content(response: Mapping[str, Any], subsection: str) -> str:
    # `subsection_title` is a label, not article content. Some compatible LLM
    # endpoints return both fields even when the prompt uses the subsection
    # title itself as the JSON key.
    for key in (subsection, "content"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            nested = value.get("content")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    for key in ("subsection", "result", "data"):
        value = response.get(key)
        if isinstance(value, Mapping):
            nested = value.get("content")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def _normalize_citations(
    content: str, allowed_numbers: Sequence[int] = ()
) -> str:
    """Convert supported `ref [N]` variants to internal `[ref:N]`."""

    def replace_group(match: re.Match[str]) -> str:
        numbers = re.findall(r"\d+", match.group(1))
        return " ".join(f"[ref:{number}]" for number in numbers)

    content = re.sub(
        r"\brefs?\.?\s*\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]",
        replace_group,
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(
        r"\[\s*ref\s*:?\s*(\d+)\s*\]",
        r"[ref:\1]",
        content,
        flags=re.IGNORECASE,
    )
    allowed = {int(number) for number in allowed_numbers}
    if not allowed:
        return content

    def replace_bare_group(match: re.Match[str]) -> str:
        numbers = [int(value) for value in re.findall(r"\d+", match.group(1))]
        if numbers and all(number in allowed for number in numbers):
            return " ".join(f"[ref:{number}]" for number in numbers)
        return match.group(0)

    return re.sub(
        r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]",
        replace_bare_group,
        content,
    )


def _fit_context(papers: Sequence[Paper], max_characters: int) -> List[Paper]:
    output: List[Paper] = []
    used = 0
    for paper in papers:
        size = len(paper.title) + len(paper.evidence_text())
        if output and used + size > max_characters:
            break
        output.append(paper)
        used += size
    return output


def build_citation_plan(
    outline: Outline,
    papers_by_section: Mapping[str, Sequence[Paper]],
    skipped_papers: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Create one stable global numbering table with section assignments."""
    section_details = [
        (
            section.section_title,
            list(section.subsections) or [section.section_title],
        )
        for section in outline.outline
    ]
    known_sections = {title for title, _ in section_details}
    section_details.extend(
        (title, [title])
        for title in papers_by_section
        if title not in known_sections
    )

    numbers_by_alias: Dict[str, int] = {}
    canonical_by_number: Dict[int, str] = {}
    doi_by_number: Dict[int, str] = {}
    rows: List[Dict[str, Any]] = []
    skipped = skipped_papers if skipped_papers is not None else []
    for section_order, (section_title, subsections) in enumerate(section_details, 1):
        seen_in_section = set()
        for rank, paper in enumerate(papers_by_section.get(section_title, []), 1):
            aliases = _paper_aliases(paper)
            existing_numbers = {
                numbers_by_alias[alias]
                for alias in aliases
                if alias in numbers_by_alias
            }
            if len(existing_numbers) > 1:
                skipped.append(_skipped_paper(
                    paper,
                    section_title,
                    "conflicting DOI/title identities",
                ))
                continue
            ref_no = (
                next(iter(existing_numbers))
                if existing_numbers
                else len(canonical_by_number) + 1
            )
            doi = normalize_doi(paper.doi)
            if doi and doi_by_number.get(ref_no, doi) != doi:
                skipped.append(_skipped_paper(
                    paper,
                    section_title,
                    "normalized title is already assigned to a different DOI",
                ))
                continue
            if doi:
                doi_by_number[ref_no] = doi
            for alias in aliases:
                numbers_by_alias[alias] = ref_no
            canonical_key = canonical_by_number.setdefault(ref_no, aliases[0])
            if ref_no in seen_in_section:
                continue
            seen_in_section.add(ref_no)
            rows.append(citation_plan_row(
                paper,
                ref_no,
                section_title,
                section_order=section_order,
                subsections=subsections,
                rank_in_section=rank,
                selected_stage="section_ranking",
                paper_key=canonical_key,
            ))
    return rows


def citation_plan_row(
    paper: Paper,
    ref_no: int,
    section_title: str,
    *,
    section_order: int | str = "",
    subsections: Sequence[str] = (),
    rank_in_section: int | str = "",
    selected_stage: str = "section_ranking",
    paper_key: str = "",
) -> Dict[str, Any]:
    return {
        "ref_no": int(ref_no),
        "section_order": section_order,
        "section_title": section_title,
        "subsections": " | ".join(str(value) for value in subsections if value),
        "rank_in_section": rank_in_section,
        "paper_key": paper_key or paper.key,
        "paper_id": paper.paper_id,
        "doi": paper.doi,
        "title": paper.title,
        "source": paper.source,
        "year": paper.year or "",
        "selected_stage": selected_stage,
    }


def citation_plan_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CITATION_PLAN_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _filter_papers_by_citation_plan(
    papers_by_section: Mapping[str, Sequence[Paper]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Dict[str, List[Paper]], int]:
    rows_by_section: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        section = str(row.get("section_title") or "")
        if section:
            rows_by_section.setdefault(section, []).append(row)

    output: Dict[str, List[Paper]] = {}
    skipped = 0
    for section, papers in papers_by_section.items():
        selected = []
        seen_numbers = set()
        for paper in papers:
            match = next(
                (
                    row
                    for row in rows_by_section.get(section, [])
                    if _paper_matches_plan_row(paper, row)
                ),
                None,
            )
            if match is None:
                skipped += 1
                continue
            ref_no = int(match.get("ref_no") or 0)
            if ref_no in seen_numbers:
                continue
            seen_numbers.add(ref_no)
            selected.append(paper)
        output[section] = selected
    return output, skipped


def _paper_matches_plan_row(paper: Paper, row: Mapping[str, Any]) -> bool:
    paper_doi = normalize_doi(paper.doi)
    row_doi = normalize_doi(row.get("doi"))
    if paper_doi and row_doi:
        return paper_doi == row_doi
    return normalize_title(paper.title) == normalize_title(row.get("title"))


def _skipped_paper(
    paper: Paper, section_title: str, reason: str
) -> Dict[str, Any]:
    return {
        "stage": "citation_plan",
        "section_title": section_title,
        "paper_id": paper.paper_id,
        "doi": paper.doi,
        "title": paper.title,
        "source": paper.source,
        "reason": reason,
    }


def _citation_plan_indexes(
    rows: Sequence[Mapping[str, Any]],
    papers_by_section: Mapping[str, Sequence[Paper]],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    numbers: Dict[str, int] = {}
    keys_by_number: Dict[int, str] = {}
    bibliography_by_number: Dict[int, Dict[str, Any]] = {}
    assignments = set()
    for row in rows:
        paper_key = str(row.get("paper_key") or "")
        ref_no = int(row.get("ref_no") or 0)
        section_title = str(row.get("section_title") or "")
        if not paper_key or ref_no < 1 or not section_title:
            raise ValueError("Citation plan contains an incomplete row")
        previous = numbers.get(paper_key)
        if previous is not None and previous != ref_no:
            raise ValueError(
                f"Citation plan assigns {paper_key} both ref [{previous}] and ref [{ref_no}]"
            )
        previous_key = keys_by_number.get(ref_no)
        if previous_key is not None and previous_key != paper_key:
            raise ValueError(
                f"Citation plan assigns ref [{ref_no}] to multiple papers"
            )
        numbers[paper_key] = ref_no
        keys_by_number[ref_no] = paper_key
        for alias in _row_aliases(row):
            previous_number = numbers.get(alias)
            if previous_number is not None and previous_number != ref_no:
                raise ValueError(
                    f"Citation identity {alias} has conflicting reference numbers"
                )
            numbers[alias] = ref_no
            assignments.add((section_title, alias))
        entry = bibliography_by_number.setdefault(ref_no, {
            "refNo": ref_no,
            "paper_id": str(row.get("paper_id") or ""),
            "doi": str(row.get("doi") or ""),
            "title": str(row.get("title") or ""),
            "source": str(row.get("source") or ""),
        })
        for field in ("paper_id", "doi", "title", "source"):
            if not entry[field] and row.get(field):
                entry[field] = str(row[field])

    for section_title, papers in papers_by_section.items():
        for paper in papers:
            aliases = _paper_aliases(paper)
            assigned_numbers = {
                numbers[alias]
                for alias in aliases
                if (section_title, alias) in assignments
            }
            if not assigned_numbers:
                raise ValueError(
                    f"Citation plan does not assign '{paper.title}' to '{section_title}'"
                )
            if len(assigned_numbers) > 1:
                raise ValueError(
                    f"Citation plan assigns multiple numbers to '{paper.title}'"
                )
            numbers[paper.key] = next(iter(assigned_numbers))
    bibliography = [bibliography_by_number[key] for key in sorted(bibliography_by_number)]
    return bibliography, numbers


def _paper_aliases(paper: Paper) -> List[str]:
    aliases = []
    doi = normalize_doi(paper.doi)
    title = normalize_title(paper.title)
    if doi:
        aliases.append(f"doi:{doi}")
    if title:
        aliases.append(f"title:{title}")
    return aliases or [paper.key]


def _row_aliases(row: Mapping[str, Any]) -> List[str]:
    aliases = [str(row.get("paper_key") or "")]
    doi = normalize_doi(row.get("doi"))
    title = normalize_title(row.get("title"))
    if doi:
        aliases.append(f"doi:{doi}")
    if title:
        aliases.append(f"title:{title}")
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _validate_references(content: str, raw_references: Any, papers: Sequence[Paper], numbers: Mapping[str, int]) -> List[Dict[str, Any]]:
    allowed = {numbers[paper.key]: paper for paper in papers}
    cited = {int(value) for value in re.findall(r"\[ref:(\d+)\]", content)}
    unknown = cited - set(allowed)
    if unknown:
        raise ValueError(f"content cites unknown reference numbers: {sorted(unknown)}")
    if not cited:
        raise ValueError("content contains no [ref:N] citations")
    references = []
    for number in sorted(cited):
        paper = allowed[number]
        references.append({"refNo": number, "paper_id": paper.paper_id, "doi": paper.doi, "title": paper.title})
    return references
