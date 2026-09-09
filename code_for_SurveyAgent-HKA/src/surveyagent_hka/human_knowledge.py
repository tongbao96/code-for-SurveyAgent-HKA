from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

from .embeddings import TextEmbedder
from .models import normalize_title
from .progress import ProgressCallback, report


def load_json_records(path: str | Path) -> List[Dict[str, Any]]:
    return list(iter_json_records(path))


def iter_json_records(path: str | Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
                if isinstance(item, dict):
                    yield item
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        try:
            value = json.load(handle)
        except json.JSONDecodeError as exc:
            # Some collected peer-review corpora use JSONL content despite a
            # .json suffix.  Accept that real-world format without requiring
            # users to rename the source files.
            handle.seek(0)
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON/JSONL at {path}:{line_number}") from exc
                if isinstance(item, dict):
                    yield item
            return
    records = value if isinstance(value, list) else [value]
    yield from (item for item in records if isinstance(item, dict))


def load_human_outlines(
    path: str | Path,
    domain: str = "",
    progress: ProgressCallback = None,
) -> List[Dict[str, Any]]:
    output = []
    scanned = 0
    report(progress, f"    reading human survey structures from {Path(path).name}")
    for row in iter_json_records(path):
        scanned += 1
        meta = row.get("metadata") or row.get("meta") or {}
        if not _matches_domain(meta, domain):
            continue
        title = str(row.get("title") or meta.get("title") or "")
        abstract = str(row.get("abstract") or meta.get("abstract") or "")
        if "outline" in row:
            outline = row["outline"]
        else:
            outline = _outline_from_sections(row.get("sections", []) or [])
        if title and outline:
            output.append({"title": title, "abstract": abstract, "outline": outline})
        if scanned % 500 == 0:
            report(
                progress,
                f"      scanned {scanned} surveys; retained {len(output)} domain-matched outlines",
            )
    report(
        progress,
        f"    human outline corpus ready: scanned {scanned}, retained {len(output)}",
    )
    return output


def select_related(
    topic: str,
    rows: Sequence[Dict[str, Any]],
    embedder: TextEmbedder,
    limit: int,
    *,
    title_only: bool = False,
) -> List[Dict[str, Any]]:
    rows = [
        row for row in rows
        if normalize_title(row.get("title")) != normalize_title(topic)
    ]
    if not rows:
        return []
    texts = [
        str(row.get("title", ""))
        if title_only
        else f"{row.get('title', '')}. {row.get('abstract', '')}"
        for row in rows
    ]
    scores = embedder.similarities(topic, texts)
    order = sorted(range(len(rows)), key=lambda index: scores[index], reverse=True)[:limit]
    selected = []
    for index in order:
        item = dict(rows[index])
        item["similarity"] = round(float(scores[index]), 6)
        selected.append(item)
    return selected


def _matches_domain(metadata: Dict[str, Any], domain: str) -> bool:
    expected = str(domain or "").replace("_", " ").strip().casefold()
    if not expected:
        return True
    labels = []
    for field in ("fieldsOfStudy", "s2FieldsOfStudy"):
        for item in metadata.get(field, []) or []:
            if isinstance(item, dict):
                item = item.get("category") or item.get("source") or ""
            if item:
                labels.append(str(item).replace("_", " ").strip().casefold())
    return not labels or expected in labels


def _outline_from_sections(sections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    outline: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for section in sections:
        title = str(section.get("title") or section.get("section_title") or "").strip()
        if not title or _is_non_content_heading(title):
            continue
        index = str(section.get("index") or "").strip()
        is_subsection = "." in index and not index.endswith(".0")
        if is_subsection and current is not None:
            current["subsections"][title] = ""
            continue
        current = {"section_title": title, "subsections": {}}
        outline.append(current)
    return outline


def _is_non_content_heading(title: str) -> bool:
    normalized = " ".join(title.casefold().split()).strip(" .:;-_")
    if re.fullmatch(r"(?:fig(?:ure)?|table)\s*[a-z0-9.-]*", normalized):
        return True
    return normalized in {
        "references",
        "bibliography",
        "acknowledgment",
        "acknowledgments",
        "acknowledgement",
        "acknowledgements",
        "author contributions",
        "conflict of interest",
        "conflicts of interest",
        "funding",
        "supplementary material",
    }


def load_peer_reviews(paths: Iterable[str | Path]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for path in paths:
        for row in load_json_records(path):
            meta = row.get("meta") or row.get("metadata") or {}
            title = str(row.get("title") or meta.get("title") or "")
            abstract = str(row.get("abstract") or meta.get("abstract") or "")
            reviews = row.get("reviews") or row.get("peer_review_reports") or []
            texts = [_review_text(review) for review in reviews]
            text = "\n\n".join(item for item in texts if item)
            if (title or abstract) and text:
                fallback_title = str(meta.get("doi") or row.get("id") or "Untitled survey")
                output.append({"title": title or fallback_title, "abstract": abstract, "review": text})
    return output


def _review_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(filter(None, (_review_text(item) for item in value)))
    if not isinstance(value, dict):
        return ""
    preferred = []
    for key in (
        "review", "comments", "main", "paper_summary", "summary_of_strengths",
        "summary_of_weaknesses", "comments_suggestions_and_typos",
    ):
        if key in value:
            preferred.append(_review_text(value[key]))
    if preferred:
        return "\n".join(filter(None, preferred))
    if "report" in value:
        return _review_text(value["report"])
    ignored = {"note_id", "rid", "date", "reviewer", "scores", "meta", "affiliation", "reviewer_name"}
    return "\n".join(
        filter(None, (_review_text(item) for key, item in value.items() if key not in ignored))
    )
