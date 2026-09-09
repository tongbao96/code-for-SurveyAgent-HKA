from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


def normalize_doi(value: Any) -> str:
    text = str(value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def normalize_title(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


@dataclass
class Author:
    name: str = ""
    author_id: str = ""
    h_index: Optional[float] = None


@dataclass
class Paper:
    paper_id: str = ""
    title: str = ""
    abstract: str = ""
    doi: str = ""
    year: Optional[int] = None
    publication_date: str = ""
    source: str = ""
    venue: str = ""
    venue_id: str = ""
    pdf_url: str = ""
    authors: List[Author] = field(default_factory=list)
    referenced_works: List[str] = field(default_factory=list)
    citation_count: int = 0
    citations_by_year: Dict[str, int] = field(default_factory=dict)
    first_author_h_index: Optional[float] = None
    last_author_h_index: Optional[float] = None
    venue_h_index: Optional[float] = None
    semantic_similarity: Optional[float] = None
    topical_relevance: Optional[float] = None
    academic_impact: Optional[float] = None
    recent_popularity: Optional[float] = None
    final_rank_score: Optional[float] = None
    full_text: str = ""
    knowledge_card: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"doi:{normalize_doi(self.doi)}" if normalize_doi(self.doi) else f"title:{normalize_title(self.title)}"

    @property
    def author_h_index(self) -> Optional[float]:
        values = [x for x in (self.first_author_h_index, self.last_author_h_index) if x is not None]
        return sum(values) / len(values) if values else None

    def evidence_text(self) -> str:
        if self.knowledge_card:
            return str(self.knowledge_card)
        return self.full_text or self.abstract

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Paper":
        aliases = {
            "corpusId": "paper_id",
            "paperId": "paper_id",
            "publicationDate": "publication_date",
            "journalName": "venue",
            "cited_by_count": "citation_count",
            "citationCount": "citation_count",
            "first_author_hindex": "first_author_h_index",
            "last_author_hindex": "last_author_h_index",
            "journal_hindex": "venue_h_index",
        }
        known = {item.name for item in fields(cls)}
        data: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for key, item in value.items():
            target = aliases.get(key, key)
            if target in known and target != "authors":
                data[target] = item
            elif key != "authors":
                extra[key] = item
        authors = []
        for item in value.get("authors", []) or []:
            if isinstance(item, dict):
                authors.append(Author(
                    name=str(item.get("name", "")),
                    author_id=str(item.get("authorId") or item.get("id") or ""),
                    h_index=item.get("hIndex") if item.get("hIndex") is not None else item.get("h_index"),
                ))
        data["authors"] = authors
        data["doi"] = normalize_doi(data.get("doi", ""))
        data["extra"] = {**extra, **(value.get("extra", {}) or {})}
        return cls(**data)


@dataclass
class OutlineSection:
    section_title: str
    description: str = ""
    subsections: Dict[str, str] = field(default_factory=dict)


@dataclass
class Outline:
    title: str
    outline: List[OutlineSection] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Outline":
        sections = []
        for item in value.get("outline", value.get("sections", [])) or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("section_title") or item.get("title") or "").strip()
            raw_subs = item.get("subsections", {}) or {}
            if isinstance(raw_subs, list):
                subs = {
                    str(x.get("title", x) if isinstance(x, dict) else x):
                    str(x.get("description", "") if isinstance(x, dict) else "")
                    for x in raw_subs
                }
            else:
                subs = {str(k): str(v or "") for k, v in raw_subs.items()}
            if title:
                sections.append(OutlineSection(title, str(item.get("description", "")), subs))
        return cls(str(value.get("title", "Survey")), sections)


@dataclass
class SurveySection:
    section_title: str
    subsection_title: str
    content: str
    references: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SurveyDraft:
    title: str
    sections: List[SurveySection] = field(default_factory=list)
    bibliography: List[Dict[str, Any]] = field(default_factory=list)
    revision_round: int = 0
    revision_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SurveyDraft":
        return cls(
            title=str(value.get("title", "Survey")),
            sections=[SurveySection(**item) for item in value.get("sections", [])],
            bibliography=list(value.get("bibliography", [])),
            revision_round=int(value.get("revision_round", 0)),
            revision_log=list(value.get("revision_log", [])),
        )

    def as_markdown(self) -> str:
        lines = [f"# {self.title}"]
        current = None
        for item in self.sections:
            if item.section_title != current:
                lines.extend(["", f"## {item.section_title}"])
                current = item.section_title
            lines.extend(["", f"### {item.subsection_title}", "", item.content.strip()])
        if self.bibliography:
            lines.extend(["", "## References"])
            for item in sorted(self.bibliography, key=lambda row: int(row.get("refNo", 0))):
                doi = f" https://doi.org/{item['doi']}" if item.get("doi") else ""
                lines.append(f"[{item.get('refNo')}] {item.get('title', '')}.{doi}")
        return "\n".join(lines).strip() + "\n"
