from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Mapping, Sequence

from .embeddings import TextEmbedder
from .human_knowledge import load_peer_reviews, select_related
from .llm import LLMClient
from .models import SurveyDraft, SurveySection
from .progress import ProgressCallback, report, report_bar
from . import prompts


def build_checklist(
    topic: str,
    paths: Sequence[str],
    llm: LLMClient,
    embedder: TextEmbedder,
    limit: int = 20,
) -> tuple[Dict[str, Any], List[Dict[str, str]]]:
    reviews = load_peer_reviews(paths)
    related = select_related(topic, reviews, embedder, limit)
    if not related:
        return _default_checklist(), []
    compact = [
        {"title": row["title"], "review": row["review"][:12_000]}
        for row in related
    ]
    return llm.json(prompts.extract_checklist(topic, compact)), related


def review_draft(
    topic: str, draft: SurveyDraft, checklist: Dict[str, Any], llm: LLMClient
) -> Dict[str, Any]:
    return llm.json(prompts.review_survey(topic, draft, checklist))


def revise_draft(
    topic: str,
    draft: SurveyDraft,
    comments: Dict[str, Any],
    added_literature: Dict[str, Any],
    llm: LLMClient,
) -> SurveyDraft:
    response = llm.json(prompts.revise_survey(topic, draft, comments, added_literature))
    payload = _survey_payload(response, "Revised Survey", draft.title)
    payload["bibliography"] = draft.bibliography
    revised = SurveyDraft.from_dict(payload)
    revised.revision_round = draft.revision_round + 1
    raw_log = response.get("Revision Log") or response.get("revision_log") or []
    revised.revision_log = [*draft.revision_log, *_normalize_revision_log(raw_log)]
    return revised


def score_draft(
    topic: str, draft: SurveyDraft, checklist: Dict[str, Any], llm: LLMClient
) -> float:
    response = llm.json(prompts.score_survey(topic, draft, checklist))
    values = []
    for value in response.values():
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if 1 <= score <= 5:
            values.append(score)
    return sum(values) / len(values) if values else 0.0


def polish_draft(
    topic: str,
    draft: SurveyDraft,
    llm: LLMClient,
    max_workers: int = 2,
    max_context_tokens: int = 1_050_000,
    progress: ProgressCallback = None,
) -> SurveyDraft:
    """Polish independent subsections without risking the completed draft.

    A long whole-survey request is both slow and vulnerable to provider timeout.
    Each subsection is therefore bounded as an independent Refiner task.  If a
    task times out, returns truncated prose, or changes its citation set, the
    original subsection is retained and the remaining tasks continue.
    """
    if not draft.sections:
        return draft

    total = len(draft.sections)
    workers = max(1, min(int(max_workers), total))
    results: List[SurveySection | None] = [None] * total
    failures: Counter[str] = Counter()
    report(
        progress,
        f"[ROUTER] dispatching {total} subsection polish task(s) to "
        f"[REFINER] with {workers} worker(s)",
    )
    report_bar(progress, "Refiner subsection polish", 0, total)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _polish_one,
                topic,
                draft.title,
                section,
                llm,
                max_context_tokens,
            ): index
            for index, section in enumerate(draft.sections)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            original = draft.sections[index]
            try:
                results[index] = future.result()
            except Exception as exc:  # One provider/model failure must not abort the survey.
                results[index] = _copy_section(original)
                failures[type(exc).__name__] += 1
            completed += 1
            report_bar(progress, "Refiner subsection polish", completed, total)

    if failures:
        breakdown = ", ".join(
            f"{name}={count}" for name, count in sorted(failures.items())
        )
        report(
            progress,
            f"    retained original text for {sum(failures.values())}/{total} "
            f"subsection(s) after controlled Refiner failures ({breakdown})",
        )

    return SurveyDraft(
        title=draft.title,
        sections=[section for section in results if section is not None],
        bibliography=[dict(item) for item in draft.bibliography],
        revision_round=draft.revision_round,
        revision_log=[dict(item) for item in draft.revision_log],
    )


def _polish_one(
    topic: str,
    survey_title: str,
    section: SurveySection,
    llm: LLMClient,
    max_context_tokens: int,
) -> SurveySection:
    max_content_characters = _polish_content_character_limit(max_context_tokens)
    chunks = _split_polish_content(section.content, max_content_characters)
    polished_chunks = []
    for chunk in chunks:
        chunk_section = SurveySection(
            section_title=section.section_title,
            subsection_title=section.subsection_title,
            content=chunk,
            references=section.references,
        )
        polished_chunks.append(
            _polish_content(topic, survey_title, chunk_section, llm)
        )
    content = "\n\n".join(polished_chunks).strip()
    _validate_polished_content(section.content, content)
    return SurveySection(
        section_title=section.section_title,
        subsection_title=section.subsection_title,
        content=content,
        references=[dict(item) for item in section.references],
    )


def _polish_content(
    topic: str,
    survey_title: str,
    section: SurveySection,
    llm: LLMClient,
) -> str:
    response = llm.json(
        prompts.polish_subsection(topic, survey_title, section)
    )
    content = _polished_content(response)
    content = re.sub(
        r"\bref\s*\[(\d+)\]",
        r"[ref:\1]",
        content,
        flags=re.IGNORECASE,
    ).strip()
    if not content:
        raise ValueError("Refiner returned empty subsection content")
    _validate_polished_content(section.content, content)
    return content


def _validate_polished_content(original: str, polished: str) -> None:
    original_citations = _citation_numbers(original)
    polished_citations = _citation_numbers(polished)
    if polished_citations != original_citations:
        raise ValueError("Refiner changed the subsection citation set")

    original_words = len(original.split())
    polished_words = len(polished.split())
    if original_words >= 100 and polished_words < int(original_words * 0.6):
        raise ValueError("Refiner returned truncated subsection content")
    if original_words >= 100 and polished_words > int(original_words * 1.5):
        raise ValueError("Refiner expanded subsection beyond polish-only scope")


def _polish_content_character_limit(context_tokens: int) -> int:
    # Use a conservative three characters per token and reserve the same token
    # volume for rewritten output, plus fixed prompt/JSON overhead.
    usable_tokens = max(1_024, int(context_tokens) - 4_096)
    return max(1_000, usable_tokens * 3 // 2)


def _split_polish_content(content: str, max_characters: int) -> List[str]:
    if len(content) <= max_characters:
        return [content]

    chunks: List[str] = []
    current = ""
    paragraphs = re.split(r"\n\s*\n", content)
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_characters:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(paragraph) > max_characters:
            split_at = paragraph.rfind(" ", 0, max_characters + 1)
            if split_at < max_characters // 2:
                split_at = max_characters
            chunks.append(paragraph[:split_at].strip())
            paragraph = paragraph[split_at:].strip()
        current = paragraph
    if current:
        chunks.append(current)
    return chunks or [content]


def _polished_content(response: Mapping[str, Any]) -> str:
    content = response.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Mapping):
        for key in ("content", "Polished Subsection", "polished_subsection"):
            nested = content.get(key)
            if isinstance(nested, str):
                return nested.strip()
    for key in ("Polished Subsection", "polished_subsection"):
        nested = response.get(key)
        if isinstance(nested, str):
            return nested.strip()
        if isinstance(nested, Mapping) and isinstance(nested.get("content"), str):
            return str(nested["content"]).strip()
    return ""


def _citation_numbers(content: str) -> set[int]:
    return {int(number) for number in re.findall(r"\[ref:(\d+)\]", content)}


def _copy_section(section: SurveySection) -> SurveySection:
    return SurveySection(
        section_title=section.section_title,
        subsection_title=section.subsection_title,
        content=section.content,
        references=[dict(item) for item in section.references],
    )


def _survey_payload(
    response: Mapping[str, Any],
    wrapper: str,
    default_title: str,
) -> Dict[str, Any]:
    nested = response.get(wrapper) or response.get(wrapper.lower().replace(" ", "_"))
    payload = dict(nested) if isinstance(nested, Mapping) else dict(response)
    payload.setdefault("title", default_title)
    sections = []
    for item in payload.get("sections", []) or []:
        if not isinstance(item, Mapping):
            continue
        section = dict(item)
        section["content"] = re.sub(
            r"\bref\s*\[(\d+)\]",
            r"[ref:\1]",
            str(section.get("content", "")),
            flags=re.IGNORECASE,
        )
        sections.append(section)
    payload["sections"] = sections
    return payload


def _normalize_revision_log(rows: Any) -> List[Dict[str, Any]]:
    output = []
    for item in rows if isinstance(rows, list) else []:
        if not isinstance(item, Mapping):
            continue
        output.append({
            "comment_id": item.get("comment_id") or item.get("Comments ID") or "",
            "summary": item.get("summary") or item.get("Modification Summary") or "",
            "location": item.get("location") or item.get("Location of Modification") or "",
            "status": item.get("status") or item.get("Status") or "",
            "notes": item.get("notes") or item.get("Notes (if any)") or "",
        })
    return output


def retrieval_requests(comments: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = comments.get("reference_modifications") or comments.get("Reference Modifications") or []
    output = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        trigger = item.get("trigger_retrieval", item.get("Trigger for further retrieval", False))
        if str(trigger).lower() not in {"true", "1", "yes"}:
            continue
        topic = str(item.get("literature_topic") or item.get("Topic of literature to add") or "").strip()
        target = str(item.get("target") or item.get("Location to add") or "").strip()
        if topic:
            output.append({"topic": topic, "target": target, "comment_id": str(item.get("comment_id") or item.get("Comments ID") or "")})
    return output


def has_actionable_comments(comments: Dict[str, Any]) -> bool:
    return any(isinstance(value, list) and value for value in comments.values())


def _default_checklist() -> Dict[str, Any]:
    return {
        "Literature Coverage and Relevance": ["Are seminal and recent works both covered and integrated?"],
        "Structure Depth and Coherence": ["Do sections follow a complete and logical progression?"],
        "Writing Quality and Clarity": ["Is the writing precise, cohesive, and non-redundant?"],
        "Critical Analysis and Future Outlook": ["Are limitations, comparisons, and future directions critically discussed?"],
    }
