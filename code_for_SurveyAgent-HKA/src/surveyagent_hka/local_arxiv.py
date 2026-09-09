from __future__ import annotations

import importlib.util
import json
import os
import re
import sqlite3
import zipfile
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .models import Author, Paper, normalize_doi
from .progress import ProgressCallback, report


METADATA_MEMBER = "arxiv_paper_db.json"
MAPPING_MEMBER = "arxivid_to_index_abs.json"
ABSTRACT_INDEX_MEMBER = "faiss_paper_abs_embeddings.bin"
_CACHE_SCHEMA_VERSION = 1
_LOAD_LOCK = Lock()
_SEARCH_LOCK = Lock()
_LOADED: Dict[Tuple[str, int, int, str, str, str, str, bool], Dict[str, Any]] = {}
_SURVEY_TITLE_PATTERN = re.compile(
    r"\b(surveys?|reviews?|overview|tutorial|meta[ -]?analysis|"
    r"systematic literature|mapping study|state[ -]of[ -]the[ -]art)\b",
    flags=re.IGNORECASE,
)


def validate_local_arxiv_database(
    archive_path: str,
    progress: ProgressCallback = None,
) -> None:
    """Fail early when the configured AutoSurvey backend cannot be used."""
    archive = Path(archive_path).resolve()
    if not archive.is_file():
        raise FileNotFoundError(
            f"AutoSurvey arXiv database not found: {archive}. "
            "Place database.zip next to config.json or update "
            "retrieval.arxiv_database_path."
        )

    required_modules = {
        "faiss": "faiss-cpu",
        "ijson": "ijson",
        "sentence_transformers": "sentence-transformers",
        "zipfile64": "zipfile64",
    }
    missing_packages = [
        package
        for module, package in required_modules.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing_packages:
        raise RuntimeError(
            "Local arXiv retrieval is missing required packages: "
            f"{', '.join(missing_packages)}. Install requirements.txt."
        )

    try:
        with zipfile.ZipFile(archive) as bundle:
            members = {Path(name).name for name in bundle.namelist()}
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Invalid AutoSurvey database archive: {archive}") from exc
    required_members = {METADATA_MEMBER, MAPPING_MEMBER, ABSTRACT_INDEX_MEMBER}
    missing_members = sorted(required_members - members)
    if missing_members:
        raise RuntimeError(
            "AutoSurvey database.zip is missing required files: "
            + ", ".join(missing_members)
        )
    report(progress, f"[preflight] local arxiv database - PASS ({archive})")


def search_local_arxiv(
    query: str,
    year_end: int,
    limit: int,
    *,
    archive_path: str,
    cache_dir: str,
    embedding_model: str,
    device: str = "auto",
    faiss_device: str = "cpu",
    overfetch_factor: int = 5,
    trust_remote_code: bool = True,
    progress: ProgressCallback = None,
) -> List[Paper]:
    """Search AutoSurvey's bundled arXiv abstract FAISS database."""
    state = _load_state(
        archive_path,
        cache_dir,
        embedding_model,
        device,
        faiss_device,
        trust_remote_code,
        progress,
    )
    count = max(1, int(limit))
    candidate_count = min(
        int(state["index"].ntotal),
        max(count, count * max(1, int(overfetch_factor))),
    )
    with _SEARCH_LOCK:
        vector = state["model"].encode(
            [f"search_query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        vector = np.asarray(vector, dtype="float32")
        if vector.ndim != 2 or vector.shape[1] != int(state["index"].d):
            raise RuntimeError(
                "Local arXiv query vector dimension does not match the AutoSurvey "
                f"FAISS index ({vector.shape[-1]} != {state['index'].d}). "
                "Keep retrieval.arxiv_database_embedding_model set to the model "
                "used by AutoSurvey."
            )
        distances, positions = state["index"].search(vector, candidate_count)

    hits = [
        (int(position), float(score))
        for position, score in zip(positions[0], distances[0])
        if int(position) >= 0
    ]
    ordered_positions = [position for position, _ in hits]
    rows = _metadata_for_positions(state["metadata_path"], ordered_positions)
    papers: List[Paper] = []
    malformed = 0
    for position, score in hits:
        row = rows.get(position)
        if not row:
            continue
        try:
            paper = _paper_from_row(row, float(score))
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            malformed += 1
            continue
        if paper.year and paper.year > year_end:
            continue
        if (
            not paper.title.strip()
            or not paper.abstract.strip()
            or _SURVEY_TITLE_PATTERN.search(paper.title)
        ):
            continue
        papers.append(paper)
        if len(papers) == count:
            break
    if malformed:
        report(
            progress,
            f"      local arxiv skipped {malformed} malformed paper record(s)",
        )
    report(
        progress,
        f"      local arxiv database returned {len(papers)} papers "
        f"from {candidate_count} vector candidates",
    )
    return papers


def _load_state(
    archive_path: str,
    cache_dir: str,
    embedding_model: str,
    device: str,
    faiss_device: str,
    trust_remote_code: bool,
    progress: ProgressCallback,
) -> Dict[str, Any]:
    archive = Path(archive_path).resolve()
    cache = Path(cache_dir).resolve()
    if not archive.is_file():
        raise FileNotFoundError(
            f"AutoSurvey arXiv database not found: {archive}. "
            "Place database.zip next to config.json or update "
            "retrieval.arxiv_database_path."
        )
    stat = archive.stat()
    key = (
        str(archive),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        str(cache),
        embedding_model,
        device,
        faiss_device,
        bool(trust_remote_code),
    )
    with _LOAD_LOCK:
        if key in _LOADED:
            return _LOADED[key]
        index_path, metadata_path = _prepare_cache(archive, cache, progress)
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "Local arXiv retrieval requires faiss-cpu. Install requirements.txt."
            ) from exc
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local arXiv retrieval requires sentence-transformers. "
                "Install requirements.txt."
            ) from exc

        report(progress, "      memory-mapping AutoSurvey FAISS abstract index")
        try:
            mmap_flag = getattr(faiss, "IO_FLAG_MMAP_IFC", faiss.IO_FLAG_MMAP)
            flags = mmap_flag | getattr(faiss, "IO_FLAG_READ_ONLY", 0)
            index = faiss.read_index(str(index_path), flags)
        except Exception as exc:
            raise RuntimeError(
                "Unable to memory-map the AutoSurvey FAISS index. Update faiss-cpu "
                "instead of loading the 1.65 GB index fully into memory."
            ) from exc
        index, gpu_resources = _move_faiss_index(
            faiss, index, faiss_device, progress
        )
        encoder_options = {"trust_remote_code": bool(trust_remote_code)}
        if device.strip().lower() != "auto":
            encoder_options["device"] = device
        report(
            progress,
            f"      loading local query encoder: {embedding_model} "
            f"(device={device})",
        )
        model = SentenceTransformer(embedding_model, **encoder_options)
        state = {
            "index": index,
            "metadata_path": str(metadata_path),
            "model": model,
            "gpu_resources": gpu_resources,
        }
        _LOADED[key] = state
        return state


def _move_faiss_index(
    faiss: Any,
    index: Any,
    requested_device: str,
    progress: ProgressCallback,
) -> Tuple[Any, Any]:
    device = requested_device.strip().lower()
    if device == "cpu":
        report(progress, "      local FAISS search device: cpu")
        return index, None
    if device != "auto" and not re.fullmatch(r"(?:cuda|gpu)(?::\d+)?", device):
        raise ValueError(
            "retrieval.arxiv_database_faiss_device must be cpu, auto, "
            "cuda, or cuda:N"
        )

    has_gpu_api = all(
        hasattr(faiss, name)
        for name in ("StandardGpuResources", "index_cpu_to_gpu")
    )
    gpu_count = int(faiss.get_num_gpus()) if hasattr(faiss, "get_num_gpus") else 0
    if not has_gpu_api or gpu_count < 1:
        if device == "auto":
            report(progress, "      local FAISS search device: cpu (GPU unavailable)")
            return index, None
        raise RuntimeError(
            "FAISS GPU was requested but the installed FAISS build has no CUDA "
            "support. Install a CUDA-compatible FAISS build or set "
            "retrieval.arxiv_database_faiss_device to 'cpu'."
        )

    gpu_id = int(device.split(":", 1)[1]) if ":" in device else 0
    if gpu_id >= gpu_count:
        raise RuntimeError(
            f"FAISS GPU {gpu_id} was requested, but only {gpu_count} GPU(s) "
            "are visible."
        )
    resources = faiss.StandardGpuResources()
    try:
        gpu_index = faiss.index_cpu_to_gpu(resources, gpu_id, index)
    except Exception as exc:
        if device == "auto":
            report(
                progress,
                "      local FAISS GPU transfer failed; continuing on cpu",
            )
            return index, None
        raise RuntimeError(
            "Unable to copy the AutoSurvey FAISS index to GPU. The complete "
            "abstract index needs about 1.65 GB of GPU memory."
        ) from exc
    report(progress, f"      local FAISS search device: cuda:{gpu_id}")
    return gpu_index, resources


def _prepare_cache(
    archive: Path,
    cache: Path,
    progress: ProgressCallback,
) -> Tuple[Path, Path]:
    try:
        import ijson
        import zipfile64.zipfile as zipfile
    except ImportError as exc:
        raise RuntimeError(
            "Reading AutoSurvey database.zip requires ijson and zipfile64. "
            "Install requirements.txt."
        ) from exc

    cache.mkdir(parents=True, exist_ok=True)
    index_path = cache / ABSTRACT_INDEX_MEMBER
    metadata_path = cache / "arxiv_metadata.sqlite3"
    manifest_path = cache / "manifest.json"
    fingerprint = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "archive_size": archive.stat().st_size,
        "archive_mtime_ns": archive.stat().st_mtime_ns,
    }
    if _cache_matches(manifest_path, index_path, metadata_path, fingerprint):
        report(progress, f"      reusing local arxiv cache: {cache}")
        return index_path, metadata_path

    report(
        progress,
        "      preparing local arxiv cache; this one-time step may take several minutes",
    )
    index_part = index_path.with_suffix(index_path.suffix + ".part")
    metadata_part = metadata_path.with_suffix(metadata_path.suffix + ".part")
    manifest_part = manifest_path.with_suffix(".json.part")
    for path in (index_part, metadata_part, manifest_part):
        if path.exists():
            path.unlink()

    with zipfile.ZipFile(archive) as bundle:
        members = {Path(name).name: name for name in bundle.namelist()}
        required = {METADATA_MEMBER, MAPPING_MEMBER, ABSTRACT_INDEX_MEMBER}
        missing = sorted(required - set(members))
        if missing:
            raise RuntimeError(
                "AutoSurvey database.zip is missing required files: "
                + ", ".join(missing)
            )
        index_info = bundle.getinfo(members[ABSTRACT_INDEX_MEMBER])
        _extract_index(
            bundle,
            members[ABSTRACT_INDEX_MEMBER],
            index_part,
            index_info.file_size,
            progress,
        )
        _build_metadata_database(
            bundle,
            members[MAPPING_MEMBER],
            members[METADATA_MEMBER],
            metadata_part,
            ijson,
            progress,
        )

    os.replace(index_part, index_path)
    os.replace(metadata_part, metadata_path)
    fingerprint["index_size"] = index_path.stat().st_size
    manifest_part.write_text(
        json.dumps(fingerprint, indent=2), encoding="utf-8"
    )
    os.replace(manifest_part, manifest_path)
    report(progress, f"      local arxiv cache ready: {cache}")
    return index_path, metadata_path


def _cache_matches(
    manifest_path: Path,
    index_path: Path,
    metadata_path: Path,
    fingerprint: Mapping[str, Any],
) -> bool:
    if not manifest_path.is_file() or not index_path.is_file() or not metadata_path.is_file():
        return False
    try:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        all(saved.get(key) == value for key, value in fingerprint.items())
        and saved.get("index_size") == index_path.stat().st_size
    )


def _extract_index(
    bundle: Any,
    member: str,
    destination: Path,
    total_bytes: int,
    progress: ProgressCallback,
) -> None:
    copied = 0
    next_report = 256 * 1024 * 1024
    with bundle.open(member) as source, destination.open("wb") as target:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            copied += len(chunk)
            if copied >= next_report:
                report(
                    progress,
                    f"      extracting FAISS index: {copied / total_bytes:.0%}",
                )
                next_report += 256 * 1024 * 1024
    if copied != total_bytes:
        raise RuntimeError(
            f"Incomplete FAISS extraction: expected {total_bytes}, wrote {copied} bytes"
        )


def _build_metadata_database(
    bundle: Any,
    mapping_member: str,
    metadata_member: str,
    destination: Path,
    ijson: Any,
    progress: ProgressCallback,
) -> None:
    connection = sqlite3.connect(destination)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            """CREATE TABLE papers (
                arxiv_id TEXT PRIMARY KEY,
                faiss_index INTEGER UNIQUE,
                title TEXT,
                abstract TEXT,
                publication_date TEXT,
                year INTEGER,
                doi TEXT,
                authors TEXT,
                pdf_url TEXT
            )"""
        )
        with bundle.open(mapping_member) as stream:
            batch: List[Tuple[str, int]] = []
            count = 0
            for arxiv_id, position in ijson.kvitems(stream, ""):
                batch.append((str(arxiv_id), int(position)))
                if len(batch) == 10_000:
                    connection.executemany(
                        "INSERT INTO papers(arxiv_id, faiss_index) VALUES (?, ?)",
                        batch,
                    )
                    connection.commit()
                    count += len(batch)
                    batch.clear()
                    if count % 100_000 == 0:
                        report(progress, f"      indexed {count:,} arxiv IDs")
            if batch:
                connection.executemany(
                    "INSERT INTO papers(arxiv_id, faiss_index) VALUES (?, ?)", batch
                )
                connection.commit()
                count += len(batch)
            report(progress, f"      indexed {count:,} arxiv IDs")

        statement = """INSERT INTO papers(
                arxiv_id, title, abstract, publication_date, year, doi, authors, pdf_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                title=excluded.title,
                abstract=excluded.abstract,
                publication_date=excluded.publication_date,
                year=excluded.year,
                doi=excluded.doi,
                authors=excluded.authors,
                pdf_url=excluded.pdf_url"""
        with bundle.open(metadata_member) as stream:
            batch = []
            count = 0
            for _, row in ijson.kvitems(stream, "cs_paper_info"):
                if not isinstance(row, Mapping):
                    continue
                arxiv_id = str(row.get("id") or "").strip()
                if not arxiv_id:
                    continue
                date = str(row.get("date") or row.get("published") or "")
                batch.append((
                    arxiv_id,
                    str(row.get("title") or ""),
                    str(row.get("abs") or row.get("abstract") or ""),
                    date,
                    _year_from_date(date),
                    normalize_doi(row.get("doi")),
                    json.dumps(
                        row.get("authors") or row.get("author") or [], default=str
                    ),
                    str(row.get("pdf_url") or ""),
                ))
                if len(batch) == 5_000:
                    connection.executemany(statement, batch)
                    connection.commit()
                    count += len(batch)
                    batch.clear()
                    if count % 50_000 == 0:
                        report(progress, f"      loaded metadata for {count:,} papers")
            if batch:
                connection.executemany(statement, batch)
                connection.commit()
                count += len(batch)
            report(progress, f"      loaded metadata for {count:,} papers")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS papers_faiss_index ON papers(faiss_index)"
        )
        connection.commit()
    finally:
        connection.close()


def _metadata_for_positions(
    database_path: str, positions: Sequence[int]
) -> Dict[int, sqlite3.Row]:
    if not positions:
        return {}
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = []
        for start in range(0, len(positions), 900):
            batch = list(positions[start:start + 900])
            placeholders = ",".join("?" for _ in batch)
            rows.extend(connection.execute(
                f"SELECT * FROM papers WHERE faiss_index IN ({placeholders})",
                batch,
            ).fetchall())
    finally:
        connection.close()
    return {int(row["faiss_index"]): row for row in rows}


def _paper_from_row(row: sqlite3.Row, score: float) -> Paper:
    arxiv_id = _plain_arxiv_id(str(row["arxiv_id"] or ""))
    authors = _authors(row["authors"])
    return Paper(
        paper_id=arxiv_id,
        title=" ".join(str(row["title"] or "").split()),
        abstract=" ".join(str(row["abstract"] or "").split()),
        doi=normalize_doi(row["doi"]),
        year=row["year"],
        publication_date=str(row["publication_date"] or "")[:10],
        source="arxiv_local",
        pdf_url=str(row["pdf_url"] or "") or f"https://arxiv.org/pdf/{arxiv_id}",
        authors=authors,
        semantic_similarity=score,
        extra={"local_faiss_score": score, "arxiv_id": arxiv_id},
    )


def _authors(value: Any) -> List[Author]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError):
        parsed = value or []
    if isinstance(parsed, str):
        parsed = [parsed]
    output = []
    for item in parsed if isinstance(parsed, list) else []:
        if isinstance(item, Mapping):
            name = item.get("name") or item.get("display_name") or ""
        else:
            name = item
        if str(name).strip():
            output.append(Author(name=str(name).strip()))
    return output


def _year_from_date(value: str) -> Optional[int]:
    match = re.search(r"(?:19|20)\d{2}", value)
    return int(match.group(0)) if match else None


def _plain_arxiv_id(value: str) -> str:
    value = value.strip()
    return value.split("/abs/", 1)[1] if "/abs/" in value else value
