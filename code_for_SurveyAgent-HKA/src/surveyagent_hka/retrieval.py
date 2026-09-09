from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import requests
except ImportError:  # Offline tests can still import the package.
    requests = None  # type: ignore

from .config import RetrievalConfig, TopicConfig
from .embeddings import TextEmbedder
from .llm import LLMClient
from .local_arxiv import search_local_arxiv, validate_local_arxiv_database
from .models import Author, Paper, normalize_doi, normalize_title
from .progress import ProgressCallback, report, report_bar
from .prompts import expand_search_queries
from .router import AgentEvent, run_agent_tasks


SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1"
OPENALEX_URL = "https://api.openalex.org"
ARXIV_URL = "https://export.arxiv.org/api/query"
PUBMED_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
S2_FIELDS = (
    "paperId,corpusId,externalIds,title,abstract,year,publicationDate,"
    "citationCount,venue,journal,authors,url,openAccessPdf,references"
)
SURVEY_TITLE_PATTERN = re.compile(
    r"\b(surveys?|reviews?|overview|tutorial|meta[ -]?analysis|"
    r"systematic literature|mapping study|state[ -]of[ -]the[ -]art)\b",
    flags=re.IGNORECASE,
)
_S2_RATE_LOCK = Lock()
_S2_LAST_REQUEST = 0.0
_ARXIV_RATE_LOCK = Lock()
_ARXIV_LAST_REQUEST = 0.0
_OPENALEX_TITLE_CACHE_LOCK = Lock()
_OPENALEX_TITLE_PROGRESS_LOCK = Lock()
_OPENALEX_TITLE_PROGRESS_ACTIVE = 0
_OPENALEX_TITLE_PROGRESS_DONE = 0
_OPENALEX_TITLE_PROGRESS_TOTAL = 0
_ARXIV_STOP_WORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "of", "on", "or", "the", "to", "using", "via", "with",
}

# Each API source is just a name plus two callables. A class hierarchy adds no
# useful state here and makes test injection harder to read.
LiteratureSource = Dict[str, Any]


def check_semantic_scholar_api(
    config: RetrievalConfig,
    progress: ProgressCallback = None,
    session: Any = None,
) -> str:
    """Verify Semantic Scholar access once before starting retrieval."""
    if not config.check_semantic_scholar_api:
        report(progress, "[preflight] Semantic Scholar API check - skipped by config")
        return "skipped"

    api_key = config.semantic_scholar_api_key.strip()
    has_key = bool(api_key) and not api_key.upper().startswith(("PASTE_", "YOUR_"))
    mode = "API key" if has_key else "anonymous access"
    report(progress, f"[preflight] Semantic Scholar {mode} - checking")
    session = session or _new_session()
    response = None
    try:
        for attempt in range(1, 5):
            response = session.get(
                f"{SEMANTIC_SCHOLAR_URL}/paper/search",
                headers={"x-api-key": api_key} if has_key else {},
                params={"query": "machine learning", "limit": 1, "fields": "paperId"},
                timeout=20,
            )
            if response.status_code != 429 or attempt == 4:
                break
            wait_seconds = min(
                float(response.headers.get("Retry-After", 2 ** attempt)), 30.0
            )
            report(
                progress,
                f"[preflight] Semantic Scholar rate limited; retry "
                f"{attempt}/3 in {wait_seconds:g}s",
            )
            time.sleep(wait_seconds)
    except Exception as exc:
        raise RuntimeError(
            f"Semantic Scholar preflight failed: {type(exc).__name__}"
        ) from exc

    if response.status_code == 200:
        if has_key:
            report(progress, "[preflight] Semantic Scholar API key - PASS")
            # The introductory authenticated limit is commonly one request/second.
            time.sleep(1.05)
            return "authenticated"
        report(
            progress,
            "[preflight] Semantic Scholar API key - NOT CONFIGURED; anonymous access PASS",
        )
        return "anonymous"

    if response.status_code in {401, 403} and has_key:
        raise RuntimeError(
            "Semantic Scholar API key - FAIL "
            f"(HTTP {response.status_code}). Check retrieval.semantic_scholar_api_key in config.json."
        )
    if response.status_code == 429:
        report(
            progress,
            "[preflight] Semantic Scholar API key - RATE LIMITED after retries; "
            "the key was not rejected, continuing with per-source retrieval retries",
        )
        return "rate_limited"
    raise RuntimeError(
        f"Semantic Scholar preflight failed with HTTP {response.status_code}."
    )


def build_sources(
    config: RetrievalConfig, progress: ProgressCallback = None
) -> List[LiteratureSource]:
    unknown = set(config.sources) - {"semantic_scholar", "openalex", "arxiv", "pubmed"}
    if unknown:
        raise ValueError(f"Unknown literature sources: {sorted(unknown)}")

    s2_key = config.semantic_scholar_api_key
    openalex_key = config.openalex_api_key
    openalex_mailto = config.openalex_mailto
    pubmed_key = config.pubmed_api_key
    if config.use_arxiv_api:
        arxiv_search = lambda *args: search_arxiv(
            *args,
            session=_new_session(),
            request_interval=config.arxiv_request_interval_seconds,
            max_retries=config.arxiv_max_retries,
            contact=config.openalex_mailto,
            progress=progress,
        )
        if "arxiv" in config.sources:
            report(progress, "[preflight] arxiv retrieval - live API")
    else:
        arxiv_search = lambda *args: search_local_arxiv(
            *args,
            archive_path=config.arxiv_database_path,
            cache_dir=config.arxiv_database_cache,
            embedding_model=config.arxiv_database_embedding_model,
            device=config.arxiv_database_device,
            faiss_device=config.arxiv_database_faiss_device,
            overfetch_factor=config.arxiv_database_overfetch_factor,
            trust_remote_code=config.arxiv_database_trust_remote_code,
            progress=progress,
        )
        if "arxiv" in config.sources:
            report(
                progress,
                "[preflight] arxiv retrieval - local AutoSurvey database.zip",
            )
            validate_local_arxiv_database(config.arxiv_database_path, progress)
    available = {
        "semantic_scholar": {
            "name": "semantic_scholar",
            "search": lambda *args: search_semantic_scholar(
                *args, session=_new_session(), api_key=s2_key
            ),
            "get_many": lambda *args: get_semantic_scholar_papers(
                *args, session=_new_session(), api_key=s2_key
            ),
        },
        "openalex": {
            "name": "openalex",
            "search": lambda *args: search_openalex(
                *args,
                session=_new_session(),
                api_key=openalex_key,
                mailto=openalex_mailto,
            ),
            "get_many": lambda *args: get_openalex_works(
                *args,
                session=_new_session(),
                api_key=openalex_key,
                mailto=openalex_mailto,
            ),
        },
        "arxiv": {
            "name": "arxiv",
            "search": arxiv_search,
            "get_many": lambda identifiers: [],
        },
        "pubmed": {
            "name": "pubmed",
            "search": lambda *args: search_pubmed(
                *args, session=_new_session(), api_key=pubmed_key
            ),
            "get_many": lambda identifiers: [],
        },
    }
    return [available[name] for name in config.sources]


def search_semantic_scholar(
    query: str, year_end: int, limit: int, *, session: Any, api_key: str = ""
) -> List[Paper]:
    headers = {"x-api-key": api_key} if api_key else {}
    # The relevance endpoint accepts at most 100 records per call.  Keep each
    # expanded query to one request so the default 150-paper quota is exactly
    # two calls of 75, which is much less prone to API timeouts/rate limiting.
    request_limit = min(100, max(1, limit))
    _pace_semantic_scholar(session, bool(api_key))
    response = _request(
        session,
        "GET",
        f"{SEMANTIC_SCHOLAR_URL}/paper/search",
        headers=headers,
        params={
            "query": query,
            "year": f"1900-{year_end}",
            "limit": request_limit,
            "offset": 0,
            "fields": S2_FIELDS,
        },
    )
    rows = response.json().get("data", [])
    return _parse_candidate_records(
        (row for row in rows if isinstance(row, Mapping) and row.get("title")),
        _paper_from_s2,
    )[:limit]


def get_semantic_scholar_papers(
    identifiers: Sequence[str], *, session: Any, api_key: str = ""
) -> List[Paper]:
    identifiers = [value for value in identifiers if value and not value.startswith("http")]
    headers = {"x-api-key": api_key} if api_key else {}
    output: List[Paper] = []
    for start in range(0, len(identifiers), 500):
        _pace_semantic_scholar(session, bool(api_key))
        response = _request(
            session,
            "POST",
            f"{SEMANTIC_SCHOLAR_URL}/paper/batch",
            headers=headers,
            params={"fields": S2_FIELDS},
            json={"ids": identifiers[start:start + 500]},
        )
        output.extend(_parse_candidate_records(
            (row for row in response.json() if row), _paper_from_s2
        ))
    return [paper for paper in output if is_candidate_paper(paper)]


def search_openalex(
    query: str,
    year_end: int,
    limit: int,
    *,
    session: Any,
    api_key: str = "",
    mailto: str = "",
) -> List[Paper]:
    output: List[Paper] = []
    cursor = "*"
    while len(output) < limit:
        per_page = min(100, limit - len(output))
        response = _request(
            session,
            "GET",
            f"{OPENALEX_URL}/works",
            params=_openalex_params({
                "search": query,
                "filter": f"to_publication_date:{year_end}-12-31,has_abstract:true",
                "per_page": per_page,
                "cursor": cursor,
            }, api_key, mailto),
        )
        payload = response.json()
        rows = payload.get("results", [])
        output.extend(_parse_candidate_records(rows, _paper_from_openalex))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not rows or not cursor:
            break
    return output[:limit]


def get_openalex_works(
    identifiers: Sequence[str],
    *,
    session: Any,
    api_key: str = "",
    mailto: str = "",
) -> List[Paper]:
    identifiers = [value.rsplit("/", 1)[-1] for value in identifiers if value]
    output: List[Paper] = []
    for start in range(0, len(identifiers), 50):
        joined = "|".join(identifiers[start:start + 50])
        response = _request(
            session,
            "GET",
            f"{OPENALEX_URL}/works",
            params=_openalex_params(
                {"filter": f"openalex:{joined}", "per_page": 50}, api_key, mailto
            ),
        )
        output.extend(_parse_candidate_records(
            response.json().get("results", []), _paper_from_openalex
        ))
    return [paper for paper in output if is_candidate_paper(paper)]


def search_arxiv(
    query: str,
    year_end: int,
    limit: int,
    *,
    session: Any,
    request_interval: float = 3.1,
    max_retries: int = 4,
    contact: str = "",
    progress: ProgressCallback = None,
) -> List[Paper]:
    query = _clean_natural_language_query(query)
    if not query:
        raise ValueError("arXiv query is empty after validation")
    expressions = _arxiv_query_variants(query)
    request_limit = min(100, max(1, limit, limit * 2))
    attempts = max(1, int(max_retries))
    error: Optional[Exception] = None
    report(progress, f"      arxiv validated query: {expressions[0]}")

    for attempt in range(1, attempts + 1):
        # Try the detailed expression twice, then progressively broaden it.
        variant_index = min((attempt - 1) // 2, len(expressions) - 1)
        expression = expressions[variant_index]
        if attempt > 1:
            detail = "simplified query" if variant_index else "same validated query"
            report(
                progress,
                f"      arxiv retry {attempt}/{attempts} ({detail})",
            )
        try:
            response = _request_arxiv_once(
                session,
                {
                    "search_query": expression,
                    "start": 0,
                    "max_results": request_limit,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
                request_interval,
                contact,
            )
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = min(
                    float(response.headers.get("Retry-After", 0) or 0), 30.0
                )
                if retry_after > 0:
                    time.sleep(retry_after)
                raise RuntimeError(f"HTTP {response.status_code}")
            response.raise_for_status()
            papers = _parse_arxiv_feed(response.content, year_end)
            eligible = [paper for paper in papers if is_candidate_paper(paper)]
            if eligible or attempt == attempts:
                return eligible[:limit]
            error = RuntimeError("arXiv returned no eligible papers")
            report(
                progress,
                f"      arxiv attempt {attempt}/{attempts} returned no eligible papers",
            )
        except Exception as exc:
            error = exc
            report(
                progress,
                f"      arxiv attempt {attempt}/{attempts} failed "
                f"({_short_error(exc)})",
            )

    raise RuntimeError(
        f"arXiv search failed after {attempts} controlled attempts: "
        f"{_short_error(error or RuntimeError('unknown error'))}"
    )


def _parse_arxiv_feed(content: bytes, year_end: int) -> List[Paper]:
    root = ET.fromstring(content)
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    output: List[Paper] = []
    for entry in root.findall("a:entry", ns):
        published = entry.findtext("a:published", "", ns)
        year = int(published[:4]) if published[:4].isdigit() else None
        if year and year > year_end:
            continue
        links = {
            node.attrib.get("title", node.attrib.get("rel", "")): node.attrib.get("href", "")
            for node in entry.findall("a:link", ns)
        }
        output.append(Paper(
            paper_id=entry.findtext("a:id", "", ns).rsplit("/", 1)[-1],
            title=" ".join(entry.findtext("a:title", "", ns).split()),
            abstract=" ".join(entry.findtext("a:summary", "", ns).split()),
            doi=entry.findtext("arxiv:doi", "", ns),
            year=year,
            publication_date=published[:10],
            source="arxiv",
            pdf_url=links.get("pdf", ""),
            authors=[
                Author(name=node.findtext("a:name", "", ns))
                for node in entry.findall("a:author", ns)
            ],
            extra={
                "arxiv_id": entry.findtext("a:id", "", ns).rsplit("/", 1)[-1]
            },
        ))
    return output


def search_pubmed(
    query: str, year_end: int, limit: int, *, session: Any, api_key: str = ""
) -> List[Paper]:
    params = {
        "db": "pubmed",
        "term": f'({query}) AND ("1900"[Date - Publication] : "{year_end}"[Date - Publication])',
        "retmode": "json",
        "retmax": min(200, max(limit, limit * 2)),
    }
    if api_key:
        params["api_key"] = api_key
    response = _request(session, "GET", f"{PUBMED_URL}/esearch.fcgi", params=params)
    identifiers = response.json().get("esearchresult", {}).get("idlist", [])
    if not identifiers:
        return []
    fetch_params = {"db": "pubmed", "id": ",".join(identifiers), "retmode": "xml"}
    if api_key:
        fetch_params["api_key"] = api_key
    response = _request(session, "GET", f"{PUBMED_URL}/efetch.fcgi", params=fetch_params)
    root = ET.fromstring(response.content)
    return _parse_candidate_records(
        root.findall(".//PubmedArticle"), _paper_from_pubmed
    )[:limit]


def expand_queries(
    topic: TopicConfig, llm: Optional[LLMClient], config: RetrievalConfig
) -> List[str]:
    base = _clean_natural_language_query(" ".join([topic.title, *topic.keywords]))
    if not llm or config.query_expansions <= 1:
        return [base]
    additional = config.query_expansions - 1
    prompt = expand_search_queries(topic.title, topic.keywords, additional)
    queries = llm.json(prompt).get("queries", [])
    candidates = [
        base,
        *(_clean_natural_language_query(query) for query in queries),
        _clean_natural_language_query(" ".join(topic.keywords)),
        _clean_natural_language_query(f"{topic.title} survey"),
        _clean_natural_language_query(f"{topic.title} review"),
    ]
    output: List[str] = []
    seen = set()
    for query in candidates:
        key = query.casefold()
        if query and key not in seen:
            output.append(query)
            seen.add(key)
        if len(output) == config.query_expansions:
            break
    return output


def retrieve_initial(
    topic: TopicConfig,
    sources: Sequence[LiteratureSource],
    embedder: TextEmbedder,
    llm: Optional[LLMClient],
    config: RetrievalConfig,
    progress: ProgressCallback = None,
    max_workers: int = 1,
    event: AgentEvent = None,
) -> List[Paper]:
    queries = expand_queries(topic, llm, config)
    report(progress, f"    expanded the topic into {len(queries)} search queries")
    tasks = []
    for source in sources:
        quota = (
            config.semantic_scholar_initial
            if source["name"] == "semantic_scholar"
            else config.supplementary_initial
        )
        if len(sources) == 1:
            quota = config.initial_total
        per_query = max(1, math.ceil(quota / len(queries)))
        for query_index, query in enumerate(queries, 1):
            tasks.append({
                "id": f"{source['name']}-query-{query_index}",
                "label": f"{source['name']} query {query_index}/{len(queries)} "
                f"(up to {per_query} papers)",
                "payload": (source, query, per_query),
            })

    def search(payload: tuple[LiteratureSource, str, int]) -> List[Paper]:
        source, query, limit = payload
        return source["search"](query, topic.year_end, limit)

    batches = run_agent_tasks(
        "RETRIEVER",
        tasks,
        search,
        max_workers,
        progress,
        event,
        continue_on_error=True,
    )
    successful = [rows for rows in batches if rows is not None]
    if not successful:
        raise RuntimeError("All initial literature retrieval requests failed")
    collected = [paper for rows in successful for paper in rows]
    for task, rows in zip(tasks, batches):
        if rows is not None:
            report(
                progress,
                f"    [RETRIEVER] {task['label']} returned {len(rows)} eligible papers",
            )
    unique = filter_candidate_papers(deduplicate_papers(collected), progress)
    report(progress, f"    deduplicated {len(collected)} records to {len(unique)} papers")
    return _semantic_top(topic.title, unique, config.initial_total, embedder)


def retrieve_section(
    topic: TopicConfig,
    section_title: str,
    sources: Sequence[LiteratureSource],
    embedder: TextEmbedder,
    config: RetrievalConfig,
    progress: ProgressCallback = None,
) -> List[Paper]:
    query = f"{topic.title}: {section_title}"
    papers: List[Paper] = []
    failures = 0
    for source in sources:
        report(progress, f"      searching {source['name']}")
        try:
            rows = source["search"](query, topic.year_end, config.section_results)
        except Exception as exc:
            failures += 1
            report(
                progress,
                f"      {source['name']} failed ({_short_error(exc)}); continuing",
            )
            continue
        papers.extend(rows)
        report(progress, f"      {source['name']} returned {len(rows)} papers")
    if failures == len(sources):
        raise RuntimeError(f"All literature sources failed for section: {section_title}")
    papers = [
        paper for paper in filter_candidate_papers(deduplicate_papers(papers), progress)
        if is_citable_paper(paper)
    ]
    report(progress, f"      {len(papers)} unique citable candidates")
    return _semantic_top(query, papers, config.section_results, embedder)


def co_citation_expand(
    papers: Sequence[Paper],
    sources: Sequence[LiteratureSource],
    max_added: int,
    progress: ProgressCallback = None,
    openalex_config: Optional[RetrievalConfig] = None,
) -> List[Paper]:
    counts = Counter(ref for paper in papers for ref in paper.referenced_works if ref)
    identifiers = [ref for ref, count in counts.most_common() if count >= 2]
    if not identifiers or max_added <= 0:
        report(progress, "    co-citation expansion: no eligible shared references")
        return list(papers)

    added: List[Paper] = []
    openalex_loaded = False
    for source in sources:
        compatible = [
            identifier
            for identifier in identifiers
            if _is_openalex_work_id(identifier) == (source["name"] == "openalex")
        ]
        if compatible:
            try:
                rows = source["get_many"](compatible[:max_added])
            except Exception as exc:
                report(
                    progress,
                    f"    co-citation expansion via {source['name']} failed "
                    f"({_short_error(exc)}); continuing",
                )
                if (
                    source["name"] == "openalex"
                    and openalex_config
                    and openalex_config.openalex_enrichment_required
                ):
                    raise
                continue
            added.extend(paper for paper in rows if is_candidate_paper(paper))
            if source["name"] == "openalex":
                openalex_loaded = True

    # A local arXiv run commonly omits OpenAlex from retrieval.sources.  After
    # title linking, however, its references are OpenAlex Work IDs and can be
    # fetched directly without making OpenAlex a separate search source.
    openalex_ids = [
        identifier for identifier in identifiers if _is_openalex_work_id(identifier)
    ]
    if openalex_ids and not openalex_loaded and openalex_config:
        try:
            rows = get_openalex_works(
                openalex_ids[:max_added],
                session=_new_session(),
                api_key=openalex_config.openalex_api_key,
                mailto=openalex_config.openalex_mailto,
            )
            added.extend(paper for paper in rows if is_candidate_paper(paper))
            report(
                progress,
                f"    OpenAlex fetched {len(rows)} shared references for expansion",
            )
        except Exception as exc:
            report(
                progress,
                "    OpenAlex shared-reference fetch failed "
                f"({_short_error(exc)}); continuing without those papers",
            )
            if openalex_config.openalex_enrichment_required:
                raise

    for paper in added:
        paper.extra["co_cited_count"] = counts.get(
            paper.paper_id, counts.get(f"https://openalex.org/{paper.paper_id}", 0)
        )
    expanded = deduplicate_papers([*papers, *added])[: len(papers) + max_added]
    report(progress, f"    co-citation expansion added {len(expanded) - len(papers)} papers")
    return expanded


def enrich_with_openalex(
    papers: Sequence[Paper],
    config: RetrievalConfig,
    progress: ProgressCallback = None,
) -> List[Paper]:
    """Link papers to OpenAlex, then add citation and quality metadata.

    DOI is the first choice.  arXiv records without a DOI fall back to a strict
    title search whose result must have an exact normalized title plus a close
    publication year, matching author, or matching arXiv location.
    """
    session = _new_session()
    api_key = config.openalex_api_key
    mailto = config.openalex_mailto
    output = list(papers)
    pending = [
        paper for paper in output if not paper.extra.get("openalex_enriched")
    ]
    if not pending:
        report(progress, "      OpenAlex metadata already available from cache/checkpoint")
        return output

    # Matches are keyed by object identity because Paper is intentionally
    # mutable and several sources may contain the same DOI/title.
    matches: Dict[int, Tuple[Paper, str]] = {}
    doi_groups: Dict[str, List[Paper]] = {}
    for paper in pending:
        if paper.source == "openalex":
            matches[id(paper)] = (paper, "openalex_id")
            continue
        doi = normalize_doi(paper.doi)
        if doi:
            doi_groups.setdefault(doi, []).append(paper)

    works: List[Mapping[str, Any]] = []
    dois = list(doi_groups)
    for start in range(0, len(dois), 50):
        report(
            progress,
            f"      OpenAlex metadata batch {start // 50 + 1}/{max(1, math.ceil(len(dois) / 50))}",
        )
        values = "|".join(f"https://doi.org/{doi}" for doi in dois[start:start + 50])
        try:
            response = _request(
                session,
                "GET",
                f"{OPENALEX_URL}/works",
                params=_openalex_params(
                    {"filter": f"doi:{values}", "per_page": 50}, api_key, mailto
                ),
            )
            works.extend(response.json().get("results", []))
        except Exception as exc:
            report(
                progress,
                f"      OpenAlex DOI batch failed ({_short_error(exc)}); continuing",
            )
            if config.openalex_enrichment_required:
                raise

    doi_matches = 0
    for work in works:
        try:
            openalex_paper = _paper_from_openalex(work)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            continue
        doi = normalize_doi(openalex_paper.doi)
        for original in doi_groups.get(doi, []):
            if id(original) not in matches:
                matches[id(original)] = (openalex_paper, "doi")
                doi_matches += 1

    title_candidates = [
        paper
        for paper in pending
        if id(paper) not in matches
        and config.openalex_title_match
        and _paper_arxiv_id(paper)
    ]
    title_matches = 0
    title_failures = 0
    if title_candidates:
        _openalex_title_progress_begin(progress, len(title_candidates))
        try:
            for paper in title_candidates:
                try:
                    work, method, _ = _find_openalex_work_by_title(
                        paper, session, config
                    )
                except Exception:
                    title_failures += 1
                    if config.openalex_enrichment_required:
                        raise
                    continue
                finally:
                    _openalex_title_progress_step(progress)
                if work:
                    try:
                        openalex_paper = _paper_from_openalex(work)
                    except (
                        AttributeError,
                        KeyError,
                        TypeError,
                        ValueError,
                        OverflowError,
                    ):
                        title_failures += 1
                        continue
                    matches[id(paper)] = (openalex_paper, method)
                    title_matches += 1
        finally:
            _openalex_title_progress_end(progress)

    author_ids = set()
    source_ids = set()
    for openalex_paper, _ in matches.values():
        author_ids.update(
            author.author_id for author in openalex_paper.authors if author.author_id
        )
        if openalex_paper.venue_id:
            source_ids.add(openalex_paper.venue_id)
    try:
        author_h = _openalex_hindex(session, "authors", author_ids, api_key, mailto)
        source_h = _openalex_hindex(session, "sources", source_ids, api_key, mailto)
    except Exception as exc:
        report(
            progress,
            f"      OpenAlex h-index lookup failed ({_short_error(exc)}); "
            "citation metadata is still retained",
        )
        if config.openalex_enrichment_required:
            raise
        author_h = {}
        source_h = {}
    unmatched = len(pending) - len(matches)
    report(
        progress,
        "      OpenAlex linking completed: "
        f"{doi_matches} DOI, {title_matches} title, "
        f"{unmatched} unmatched, {title_failures} lookup failures; "
        f"enriched {len(author_h)} authors and {len(source_h)} venues",
    )

    for original in output:
        match = matches.get(id(original))
        if not match:
            continue
        other, method = match
        if other is not original:
            merge_papers(original, other)
        for author in original.authors:
            author.h_index = author_h.get(author.author_id, author.h_index)
        if original.authors:
            original.first_author_h_index = original.authors[0].h_index
            original.last_author_h_index = original.authors[-1].h_index
        original.venue_h_index = source_h.get(
            original.venue_id, original.venue_h_index
        )
        original.extra["openalex_id"] = (
            other.extra.get("openalex_id")
            or (f"https://openalex.org/{other.paper_id}" if other.paper_id else "")
        )
        original.extra["openalex_match_method"] = method
        original.extra["openalex_enriched"] = True
    return output


def _openalex_title_progress_begin(
    progress: ProgressCallback, total: int
) -> None:
    global _OPENALEX_TITLE_PROGRESS_ACTIVE
    global _OPENALEX_TITLE_PROGRESS_DONE
    global _OPENALEX_TITLE_PROGRESS_TOTAL
    with _OPENALEX_TITLE_PROGRESS_LOCK:
        if _OPENALEX_TITLE_PROGRESS_ACTIVE == 0:
            _OPENALEX_TITLE_PROGRESS_DONE = 0
            _OPENALEX_TITLE_PROGRESS_TOTAL = 0
        _OPENALEX_TITLE_PROGRESS_ACTIVE += 1
        _OPENALEX_TITLE_PROGRESS_TOTAL += total
        report_bar(
            progress,
            "OpenAlex title matching",
            _OPENALEX_TITLE_PROGRESS_DONE,
            _OPENALEX_TITLE_PROGRESS_TOTAL,
        )


def _openalex_title_progress_step(progress: ProgressCallback) -> None:
    global _OPENALEX_TITLE_PROGRESS_DONE
    with _OPENALEX_TITLE_PROGRESS_LOCK:
        _OPENALEX_TITLE_PROGRESS_DONE += 1
        report_bar(
            progress,
            "OpenAlex title matching",
            _OPENALEX_TITLE_PROGRESS_DONE,
            _OPENALEX_TITLE_PROGRESS_TOTAL,
        )


def _openalex_title_progress_end(progress: ProgressCallback) -> None:
    global _OPENALEX_TITLE_PROGRESS_ACTIVE
    global _OPENALEX_TITLE_PROGRESS_DONE
    global _OPENALEX_TITLE_PROGRESS_TOTAL
    with _OPENALEX_TITLE_PROGRESS_LOCK:
        _OPENALEX_TITLE_PROGRESS_ACTIVE -= 1
        if _OPENALEX_TITLE_PROGRESS_ACTIVE == 0:
            _OPENALEX_TITLE_PROGRESS_DONE = 0
            _OPENALEX_TITLE_PROGRESS_TOTAL = 0


def _find_openalex_work_by_title(
    paper: Paper,
    session: Any,
    config: RetrievalConfig,
) -> Tuple[Optional[Mapping[str, Any]], str, bool]:
    cache_key = _openalex_title_cache_key(paper, config)
    found, cached_work = _openalex_title_cache_get(
        config.openalex_title_match_cache, cache_key
    )
    if found:
        method = _openalex_match_method(paper, cached_work, config) if cached_work else ""
        return (cached_work if method else None), method, True

    response = _request(
        session,
        "GET",
        f"{OPENALEX_URL}/works",
        params=_openalex_params(
            {
                "search": paper.title,
                "per_page": config.openalex_title_match_candidates,
            },
            config.openalex_api_key,
            config.openalex_mailto,
        ),
    )
    candidates = response.json().get("results", [])
    ranked = []
    for candidate in candidates:
        method = _openalex_match_method(paper, candidate, config)
        if method:
            ranked.append((_openalex_match_sort_key(paper, candidate, method), candidate, method))
    ranked.sort(key=lambda item: item[0])
    work = ranked[0][1] if ranked else None
    method = ranked[0][2] if ranked else ""
    _openalex_title_cache_put(config.openalex_title_match_cache, cache_key, work)
    return work, method, False


def _openalex_match_method(
    paper: Paper,
    candidate: Optional[Mapping[str, Any]],
    config: RetrievalConfig,
) -> str:
    if not candidate or normalize_title(candidate.get("title")) != normalize_title(paper.title):
        return ""
    arxiv_id = _paper_arxiv_id(paper)
    if arxiv_id and _openalex_work_has_arxiv_id(candidate, arxiv_id):
        return "exact title + arXiv location"

    candidate_year = _as_int(candidate.get("publication_year"))
    paper_year = _as_int(paper.year)
    if paper_year and candidate_year:
        if abs(candidate_year - paper_year) <= config.openalex_title_match_year_tolerance:
            return "exact title + year"
        return ""
    if _author_overlap(paper, candidate):
        return "exact title + author"
    return ""


def _openalex_match_sort_key(
    paper: Paper, candidate: Mapping[str, Any], method: str
) -> Tuple[int, int, int]:
    priority = {
        "exact title + arXiv location": 0,
        "exact title + year": 1,
        "exact title + author": 2,
    }.get(method, 9)
    paper_year = _as_int(paper.year)
    candidate_year = _as_int(candidate.get("publication_year"))
    year_gap = abs(candidate_year - paper_year) if paper_year and candidate_year else 999
    citations = int(candidate.get("cited_by_count") or 0)
    return priority, year_gap, -citations


def _paper_arxiv_id(paper: Paper) -> str:
    value = paper.extra.get("arxiv_id")
    if not value and paper.source in {"arxiv", "arxiv_local"}:
        value = paper.paper_id
    return _canonical_arxiv_id(value)


def _canonical_arxiv_id(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = re.sub(r"^https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/", "", text)
    text = re.sub(r"^(?:arxiv:|https?://doi\.org/10\.48550/arxiv\.)", "", text)
    text = text.split("?", 1)[0].split("#", 1)[0]
    text = re.sub(r"\.pdf$", "", text)
    return re.sub(r"v\d+$", "", text).strip("/")


def _openalex_work_has_arxiv_id(work: Mapping[str, Any], arxiv_id: str) -> bool:
    target = _canonical_arxiv_id(arxiv_id)
    ids = work.get("ids") or {}
    values = [ids.get("arxiv", "")]
    for location in work.get("locations", []) or []:
        values.extend(
            [
                location.get("landing_page_url", ""),
                location.get("pdf_url", ""),
            ]
        )
    return bool(target) and any(_canonical_arxiv_id(value) == target for value in values)


def _author_overlap(paper: Paper, work: Mapping[str, Any]) -> bool:
    local = {_author_key(author.name) for author in paper.authors if author.name}
    remote = {
        _author_key((entry.get("author") or {}).get("display_name", ""))
        for entry in work.get("authorships", []) or []
    }
    local.discard("")
    remote.discard("")
    return bool(local & remote)


def _author_key(value: Any) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value or "").casefold())
    return tokens[-1] if tokens else ""


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _openalex_title_cache_key(paper: Paper, config: RetrievalConfig) -> str:
    return "|".join(
        [
            "v1",
            normalize_title(paper.title),
            str(paper.year or ""),
            _paper_arxiv_id(paper),
            str(config.openalex_title_match_candidates),
            str(config.openalex_title_match_year_tolerance),
        ]
    )


def _openalex_title_cache_get(
    cache_path: str, cache_key: str
) -> Tuple[bool, Optional[Mapping[str, Any]]]:
    if not cache_path:
        return False, None
    try:
        with _OPENALEX_TITLE_CACHE_LOCK:
            connection = _openalex_title_cache(cache_path)
            try:
                row = connection.execute(
                    "SELECT work_json FROM title_matches WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
            finally:
                connection.close()
        if not row:
            return False, None
        value = json.loads(row[0])
        return True, value if isinstance(value, Mapping) else None
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return False, None


def _openalex_title_cache_put(
    cache_path: str,
    cache_key: str,
    work: Optional[Mapping[str, Any]],
) -> None:
    if not cache_path:
        return
    try:
        with _OPENALEX_TITLE_CACHE_LOCK:
            connection = _openalex_title_cache(cache_path)
            try:
                connection.execute(
                    "INSERT OR REPLACE INTO title_matches(cache_key, work_json) "
                    "VALUES (?, ?)",
                    (cache_key, json.dumps(work, ensure_ascii=False)),
                )
                connection.commit()
            finally:
                connection.close()
    except (OSError, sqlite3.Error, TypeError, ValueError):
        pass


def _openalex_title_cache(cache_path: str) -> sqlite3.Connection:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS title_matches ("
        "cache_key TEXT PRIMARY KEY, work_json TEXT NOT NULL)"
    )
    return connection


def _is_openalex_work_id(value: str) -> bool:
    return bool(re.fullmatch(r"(?:https?://openalex\.org/)?W\d+", value, re.I))


def _openalex_hindex(
    session: Any,
    entity: str,
    identifiers: Iterable[str],
    api_key: str,
    mailto: str,
) -> Dict[str, float]:
    identifiers = list(identifiers)
    output: Dict[str, float] = {}
    for start in range(0, len(identifiers), 50):
        joined = "|".join(value.rsplit("/", 1)[-1] for value in identifiers[start:start + 50])
        response = _request(
            session,
            "GET",
            f"{OPENALEX_URL}/{entity}",
            params=_openalex_params(
                {"filter": f"openalex:{joined}", "per_page": 50}, api_key, mailto
            ),
        )
        for row in response.json().get("results", []):
            output[row.get("id", "")] = float(
                row.get("summary_stats", {}).get("h_index", 0) or 0
            )
    return output


def _semantic_top(
    query: str, papers: Sequence[Paper], limit: int, embedder: TextEmbedder
) -> List[Paper]:
    scores = embedder.similarities(
        query, [f"{paper.title}. {paper.abstract}" for paper in papers]
    )
    for paper, score in zip(papers, scores):
        paper.semantic_similarity = score
    return sorted(
        papers, key=lambda paper: paper.semantic_similarity or -1, reverse=True
    )[:limit]


def _clean_natural_language_query(value: Any) -> str:
    query = " ".join(str(value or "").replace("`", " ").split())
    if re.search(
        r"\b(select|insert|update|delete|drop|create|alter)\b.*\b(from|into|table|where)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return ""
    query = re.sub(r"[\"'%;*{}\[\]<>|=:+(),]", " ", query)
    query = re.sub(r"\b(?:AND|OR|NOT)\b", " ", query, flags=re.IGNORECASE)
    return " ".join(query.split())[:240]


def _arxiv_query_variants(query: str) -> List[str]:
    """Convert free-form text into short, valid arXiv API expressions."""
    words = re.findall(r"[A-Za-z0-9]+", query)
    terms: List[str] = []
    seen = set()
    for word in words:
        value = word.casefold()
        if len(value) < 2 or value in _ARXIV_STOP_WORDS or value in seen:
            continue
        terms.append(value)
        seen.add(value)
        if len(terms) == 8:
            break
    if not terms:
        raise ValueError("arXiv query contains no searchable terms")
    detailed = " AND ".join(f"all:{term}" for term in terms)
    broad_terms = list(dict.fromkeys([*terms[:2], *terms[-2:]]))
    broad = " AND ".join(f"all:{term}" for term in broad_terms)
    return list(dict.fromkeys([detailed, broad]))


def _short_error(error: Exception) -> str:
    match = re.search(r"\b([45]\d\d)\b", str(error))
    return f"HTTP {match.group(1)}" if match else type(error).__name__


def deduplicate_papers(papers: Iterable[Paper]) -> List[Paper]:
    output: Dict[str, Paper] = {}
    for paper in papers:
        if not paper.title:
            continue
        key = paper.key
        output[key] = merge_papers(output[key], paper) if key in output else paper
    return list(output.values())


def is_candidate_paper(paper: Paper) -> bool:
    return bool(
        paper.title.strip()
        and paper.abstract.strip()
        and not SURVEY_TITLE_PATTERN.search(paper.title)
    )


def is_citable_paper(paper: Paper) -> bool:
    """Accept a DOI or an arXiv identifier as a stable citation target."""
    return bool(
        normalize_doi(paper.doi)
        or (paper.source in {"arxiv", "arxiv_local"} and paper.paper_id.strip())
    )


def filter_candidate_papers(
    papers: Iterable[Paper], progress: ProgressCallback = None
) -> List[Paper]:
    output = []
    missing_abstract = 0
    survey_like = 0
    for paper in papers:
        if not paper.abstract.strip():
            missing_abstract += 1
            continue
        if SURVEY_TITLE_PATTERN.search(paper.title):
            survey_like += 1
            continue
        output.append(paper)
    if missing_abstract or survey_like:
        report(
            progress,
            "      candidate filtering: "
            f"removed {missing_abstract} without abstracts and "
            f"{survey_like} survey/review papers",
        )
    return output


def _parse_candidate_records(
    records: Iterable[Any], parser: Callable[[Any], Paper]
) -> List[Paper]:
    """Skip one malformed provider record without discarding the response."""
    output = []
    for record in records:
        try:
            paper = parser(record)
            if is_candidate_paper(paper):
                output.append(paper)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            continue
    return output


def merge_papers(left: Paper, right: Paper) -> Paper:
    scalar_fields = (
        "paper_id", "title", "abstract", "doi", "year", "publication_date", "source",
        "venue", "venue_id", "pdf_url", "citation_count", "first_author_h_index",
        "last_author_h_index", "venue_h_index", "full_text",
    )
    for name in scalar_fields:
        current, incoming = getattr(left, name), getattr(right, name)
        if incoming not in (None, "", 0) and current in (None, "", 0):
            setattr(left, name, incoming)
    if right.citation_count > left.citation_count:
        left.citation_count = right.citation_count
    if right.authors and (not left.authors or any(author.author_id for author in right.authors)):
        left.authors = right.authors
    left.referenced_works = list(dict.fromkeys([*left.referenced_works, *right.referenced_works]))
    left.citations_by_year.update(right.citations_by_year)
    left.extra.update(right.extra)
    return left


def _new_session() -> Any:
    if requests is None:
        raise RuntimeError("Install the 'requests' package to use live literature retrieval")
    return requests.Session()


def _request_arxiv_once(
    session: Any,
    params: Mapping[str, Any],
    request_interval: float,
    contact: str,
) -> Any:
    """Make one arXiv request while enforcing its global legacy-API limits."""
    global _ARXIV_LAST_REQUEST
    interval = max(3.0, float(request_interval))
    user_agent = "SurveyAgent-HKA/0.1"
    if contact:
        user_agent += f" (mailto:{contact})"
    with _ARXIV_RATE_LOCK:
        remaining = interval - (time.monotonic() - _ARXIV_LAST_REQUEST)
        if remaining > 0:
            time.sleep(remaining)
        _ARXIV_LAST_REQUEST = time.monotonic()
        return session.get(
            ARXIV_URL,
            params=dict(params),
            headers={"User-Agent": user_agent, "Accept": "application/atom+xml"},
            timeout=60,
        )


def _pace_semantic_scholar(session: Any, authenticated: bool) -> None:
    """Respect the introductory authenticated Semantic Scholar limit of 1 RPS."""
    if not authenticated:
        return
    del session  # Kept in the signature for direct-call compatibility.
    global _S2_LAST_REQUEST
    with _S2_RATE_LOCK:
        remaining = 1.05 - (time.monotonic() - _S2_LAST_REQUEST)
        if remaining > 0:
            time.sleep(remaining)
        _S2_LAST_REQUEST = time.monotonic()


def _request(session: Any, method: str, url: str, **kwargs: Any) -> Any:
    error: Optional[Exception] = None
    timeout = kwargs.pop("timeout", 45)
    for attempt in range(1, 5):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 4:
                    time.sleep(float(response.headers.get("Retry-After", min(2 ** attempt, 10))))
                    continue
            response.raise_for_status()
            return response
        except Exception as exc:
            error = exc
            if attempt < 4:
                time.sleep(min(2 ** (attempt - 1), 8))
    detail = _short_error(error or RuntimeError("unknown error"))
    raise RuntimeError(f"{method} failed for {url}: {detail}")


def _openalex_params(value: Dict[str, Any], api_key: str, mailto: str) -> Dict[str, Any]:
    value = dict(value)
    if api_key:
        value["api_key"] = api_key
    if mailto:
        value["mailto"] = mailto
    return value


def _paper_from_s2(item: Mapping[str, Any]) -> Paper:
    external = item.get("externalIds") or {}
    refs = item.get("references") or []
    journal = item.get("journal") or {}
    pdf = item.get("openAccessPdf") or {}
    return Paper(
        paper_id=str(item.get("paperId") or item.get("corpusId") or ""),
        title=str(item.get("title") or ""),
        abstract=str(item.get("abstract") or ""),
        doi=normalize_doi(external.get("DOI")),
        year=item.get("year"),
        publication_date=str(item.get("publicationDate") or ""),
        source="semantic_scholar",
        venue=str(journal.get("name") or item.get("venue") or ""),
        pdf_url=str(pdf.get("url") or ""),
        citation_count=int(item.get("citationCount") or 0),
        authors=[
            Author(name=str(author.get("name") or ""), author_id=str(author.get("authorId") or ""))
            for author in item.get("authors", []) or []
        ],
        referenced_works=[
            str(ref.get("paperId") or "") for ref in refs if ref and ref.get("paperId")
        ],
    )


def _invert_abstract(index: Optional[Mapping[str, Sequence[int]]]) -> str:
    if not index:
        return ""
    positions = {position: word for word, indexes in index.items() for position in indexes}
    return " ".join(positions.get(position, "") for position in range(max(positions, default=-1) + 1))


def _paper_from_openalex(item: Mapping[str, Any]) -> Paper:
    location = item.get("primary_location") or {}
    source = location.get("source") or {}
    best_oa = item.get("best_oa_location") or {}
    authors = [
        Author(
            name=str((entry.get("author") or {}).get("display_name") or ""),
            author_id=str((entry.get("author") or {}).get("id") or ""),
        )
        for entry in item.get("authorships", []) or []
    ]
    return Paper(
        paper_id=str(item.get("id") or "").rsplit("/", 1)[-1],
        title=str(item.get("title") or item.get("display_name") or ""),
        abstract=_invert_abstract(item.get("abstract_inverted_index")),
        doi=normalize_doi(item.get("doi")),
        year=item.get("publication_year"),
        publication_date=str(item.get("publication_date") or ""),
        source="openalex",
        venue=str(source.get("display_name") or ""),
        venue_id=str(source.get("id") or ""),
        pdf_url=str(best_oa.get("pdf_url") or location.get("pdf_url") or ""),
        authors=authors,
        referenced_works=list(item.get("referenced_works") or []),
        citation_count=int(item.get("cited_by_count") or 0),
        citations_by_year={
            str(row.get("year")): int(row.get("cited_by_count") or 0)
            for row in item.get("counts_by_year", []) or []
        },
        extra={
            "openalex_id": str(item.get("id") or ""),
            "topics": [
                topic.get("display_name", "")
                for topic in item.get("topics", []) or []
            ],
        },
    )


def _paper_from_pubmed(article: ET.Element) -> Paper:
    citation = article.find("MedlineCitation")
    record = article.find("PubmedData")
    title_node = citation.find(".//ArticleTitle") if citation is not None else None
    title = "".join(title_node.itertext()) if title_node is not None else ""
    abstracts = [
        "".join(node.itertext()) for node in citation.findall(".//AbstractText")
    ] if citation is not None else []
    doi = ""
    if record is not None:
        for node in record.findall(".//ArticleId"):
            if node.attrib.get("IdType") == "doi":
                doi = node.text or ""
    year_text = citation.findtext(".//PubDate/Year", "") if citation is not None else ""
    pmid = citation.findtext("PMID", "") if citation is not None else ""
    authors = [
        Author(name=" ".join(filter(None, [node.findtext("ForeName"), node.findtext("LastName")])))
        for node in citation.findall(".//Author")
    ] if citation is not None else []
    return Paper(
        paper_id=pmid,
        title=title,
        abstract=" ".join(abstracts),
        doi=doi,
        year=int(year_text) if year_text.isdigit() else None,
        source="pubmed",
        venue=citation.findtext(".//Journal/Title", "") if citation is not None else "",
        authors=authors,
    )
