from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TopicConfig:
    title: str
    year_end: int
    keywords: List[str] = field(default_factory=list)
    domain: str = "computer_science"

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "TopicConfig":
        keywords = value.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [item.strip() for item in keywords.split(";") if item.strip()]
        return cls(
            title=str(value["title"]).strip(),
            year_end=int(value["year_end"]),
            keywords=list(keywords),
            domain=str(value.get("domain", "computer_science")),
        )


@dataclass
class RetrievalConfig:
    initial_total: int = 200
    semantic_scholar_initial: int = 150
    supplementary_initial: int = 50
    section_results: int = 60
    co_cited_results: int = 20
    final_per_section: int = 50
    query_expansions: int = 2
    sources: List[str] = field(default_factory=lambda: ["semantic_scholar", "openalex"])
    check_semantic_scholar_api: bool = True
    semantic_scholar_api_key: str = ""
    openalex_api_key: str = ""
    openalex_mailto: str = ""
    openalex_title_match: bool = True
    openalex_title_match_candidates: int = 5
    openalex_title_match_year_tolerance: int = 2
    openalex_title_match_cache: str = ".cache/openalex_title_matches.sqlite3"
    openalex_enrichment_required: bool = False
    pubmed_api_key: str = ""
    arxiv_request_interval_seconds: float = 3.1
    arxiv_max_retries: int = 4
    use_arxiv_api: bool = True
    arxiv_database_path: str = "database.zip"
    arxiv_database_cache: str = ".cache/autosurvey_arxiv"
    arxiv_database_embedding_model: str = "nomic-ai/nomic-embed-text-v1"
    arxiv_database_device: str = "auto"
    arxiv_database_faiss_device: str = "cpu"
    arxiv_database_overfetch_factor: int = 5
    arxiv_database_trust_remote_code: bool = True


@dataclass
class RankingConfig:
    citation_weight: float = 0.5
    author_weight: float = 0.2
    venue_weight: float = 0.3


@dataclass
class ModelConfig:
    model: str = "gpt-4.1-mini"
    polish_model: str = "gpt-5.6-luna"
    polish_context_tokens: int = 1_050_000
    api_mode: str = "responses"
    api_key: str = ""
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_retries: int = 4
    timeout_seconds: float = 120.0
    max_concurrent_requests: int = 3


@dataclass
class PipelineConfig:
    topic: TopicConfig
    workspace: str = "runs/default"
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "auto"
    require_embedding_model: bool = False
    clusters: int = 6
    cluster_min_size: int = 5
    related_human_outlines: int = 10
    related_review_surveys: int = 20
    max_revision_rounds: int = 3
    human_outlines_path: Optional[str] = None
    peer_reviews_paths: List[str] = field(default_factory=list)
    mineru_command: Optional[List[str]] = None
    require_mineru: bool = False
    download_pdfs: bool = False
    max_full_text_papers: Optional[int] = None
    parallel_enabled: bool = True
    retrieval_workers: int = 2
    cluster_workers: int = 3
    section_workers: int = 2
    review_retrieval_workers: int = 2
    extraction_workers: int = 1
    max_workers: int = 4
    polish_workers: int = 2
    writer_max_attempts: int = 4

    def __post_init__(self) -> None:
        if not self.topic.title:
            raise ValueError("topic.title must not be empty")
        if self.topic.year_end < 1900:
            raise ValueError("topic.year_end must be at least 1900")
        if self.clusters < 1:
            raise ValueError("clusters must be positive")
        if self.cluster_min_size < 2:
            raise ValueError("cluster_min_size must be at least 2")
        if self.model.api_mode not in {"responses", "chat_completions"}:
            raise ValueError("model.api_mode must be 'responses' or 'chat_completions'")
        if self.model.max_concurrent_requests < 1:
            raise ValueError("model.max_concurrent_requests must be positive")
        if self.model.polish_context_tokens < 4_096:
            raise ValueError("model.polish_context_tokens must be at least 4096")
        if not 2 <= self.writer_max_attempts <= 10:
            raise ValueError("writer_max_attempts must be between 2 and 10")
        if self.retrieval.final_per_section < 1:
            raise ValueError("retrieval.final_per_section must be positive")
        if self.retrieval.arxiv_request_interval_seconds < 3.0:
            raise ValueError(
                "retrieval.arxiv_request_interval_seconds must be at least 3.0"
            )
        if self.retrieval.arxiv_max_retries < 1:
            raise ValueError("retrieval.arxiv_max_retries must be positive")
        if self.retrieval.arxiv_database_overfetch_factor < 1:
            raise ValueError(
                "retrieval.arxiv_database_overfetch_factor must be positive"
            )
        if not 1 <= self.retrieval.openalex_title_match_candidates <= 25:
            raise ValueError(
                "retrieval.openalex_title_match_candidates must be between 1 and 25"
            )
        if self.retrieval.openalex_title_match_year_tolerance < 0:
            raise ValueError(
                "retrieval.openalex_title_match_year_tolerance must be non-negative"
            )
        if not self.retrieval.use_arxiv_api:
            if not self.retrieval.arxiv_database_path.strip():
                raise ValueError(
                    "retrieval.arxiv_database_path is required when "
                    "retrieval.use_arxiv_api is false"
                )
            if not self.retrieval.arxiv_database_embedding_model.strip():
                raise ValueError(
                    "retrieval.arxiv_database_embedding_model is required when "
                    "retrieval.use_arxiv_api is false"
                )
            if not self.retrieval.arxiv_database_device.strip():
                raise ValueError(
                    "retrieval.arxiv_database_device is required when "
                    "retrieval.use_arxiv_api is false"
                )
            if not self.retrieval.arxiv_database_faiss_device.strip():
                raise ValueError(
                    "retrieval.arxiv_database_faiss_device is required when "
                    "retrieval.use_arxiv_api is false"
                )
            faiss_device = self.retrieval.arxiv_database_faiss_device.strip().lower()
            if faiss_device not in {"cpu", "auto", "cuda", "gpu"} and not re.fullmatch(
                r"(?:cuda|gpu):\d+", faiss_device
            ):
                raise ValueError(
                    "retrieval.arxiv_database_faiss_device must be cpu, auto, "
                    "cuda, or cuda:N"
                )
        if not self.embedding_device.strip():
            raise ValueError("embedding_device must not be empty")
        weights = (
            self.ranking.citation_weight,
            self.ranking.author_weight,
            self.ranking.venue_weight,
        )
        if any(value < 0 for value in weights) or abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("ranking citation/author/venue weights must be non-negative and sum to 1")

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            topic=TopicConfig.from_dict(value["topic"]),
            workspace=str(value.get("workspace", "runs/default")),
            retrieval=RetrievalConfig(**_known_fields(RetrievalConfig, value.get("retrieval", {}))),
            ranking=RankingConfig(**_known_fields(RankingConfig, value.get("ranking", {}))),
            model=ModelConfig(**_known_fields(ModelConfig, value.get("model", {}))),
            embedding_model=str(value.get("embedding_model", "BAAI/bge-large-en-v1.5")),
            embedding_device=str(value.get("embedding_device", "auto")),
            require_embedding_model=bool(value.get("require_embedding_model", False)),
            clusters=int(value.get("clusters", 6)),
            cluster_min_size=max(2, int(value.get("cluster_min_size", 5))),
            related_human_outlines=int(value.get("related_human_outlines", 10)),
            related_review_surveys=int(value.get("related_review_surveys", 20)),
            max_revision_rounds=min(3, max(0, int(value.get("max_revision_rounds", 3)))),
            human_outlines_path=value.get("human_outlines_path"),
            peer_reviews_paths=list(value.get("peer_reviews_paths", [])),
            mineru_command=value.get("mineru_command"),
            require_mineru=bool(value.get("require_mineru", False)),
            download_pdfs=bool(value.get("download_pdfs", False)),
            max_full_text_papers=(
                None
                if value.get("max_full_text_papers") is None
                else max(0, int(value["max_full_text_papers"]))
            ),
            parallel_enabled=bool(value.get("parallel_enabled", True)),
            retrieval_workers=max(1, int(value.get("retrieval_workers", 2))),
            cluster_workers=max(1, int(value.get("cluster_workers", 3))),
            section_workers=max(1, int(value.get("section_workers", 2))),
            review_retrieval_workers=max(
                1, int(value.get("review_retrieval_workers", 2))
            ),
            extraction_workers=max(1, int(value.get("extraction_workers", 1))),
            max_workers=max(1, int(value.get("max_workers", 4))),
            polish_workers=max(1, int(value.get("polish_workers", 2))),
            writer_max_attempts=int(value.get("writer_max_attempts", 4)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            config = cls.from_dict(json.load(handle))
        base = path.resolve().parent
        config.workspace = str(_resolve(base, config.workspace))
        if config.human_outlines_path:
            config.human_outlines_path = str(_resolve(base, config.human_outlines_path))
        config.peer_reviews_paths = [str(_resolve(base, item)) for item in config.peer_reviews_paths]
        if config.retrieval.arxiv_database_path:
            config.retrieval.arxiv_database_path = str(
                _resolve(base, config.retrieval.arxiv_database_path)
            )
        if config.retrieval.arxiv_database_cache:
            config.retrieval.arxiv_database_cache = str(
                _resolve(base, config.retrieval.arxiv_database_cache)
            )
        if config.retrieval.openalex_title_match_cache:
            config.retrieval.openalex_title_match_cache = str(
                _resolve(base, config.retrieval.openalex_title_match_cache)
            )
        return config

    def to_dict(self, *, redact_secrets: bool = True) -> Dict[str, Any]:
        value = asdict(self)
        if redact_secrets:
            for key in (
                "semantic_scholar_api_key",
                "openalex_api_key",
                "openalex_mailto",
                "pubmed_api_key",
            ):
                if value["retrieval"].get(key):
                    value["retrieval"][key] = "***"
            if value["model"].get("api_key"):
                value["model"]["api_key"] = "***"
        return value

    def llm_api_key(self) -> str:
        value = self.model.api_key.strip()
        if not value or value.upper().startswith(("PASTE_", "YOUR_")):
            raise RuntimeError("Missing model.api_key in config.json")
        return value


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _known_fields(config_type: Any, value: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {item.name for item in fields(config_type)}
    return {key: item for key, item in value.items() if key in allowed}
