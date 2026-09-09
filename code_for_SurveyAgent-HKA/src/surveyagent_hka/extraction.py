from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence

from .llm import LLMClient
from .models import Paper
from .progress import ProgressCallback, at_milestone, report
from .prompts import knowledge_card, normalize_knowledge_card


def check_mineru(
    mineru_command: Optional[Sequence[str]],
    required: bool,
    progress: ProgressCallback = None,
) -> bool:
    if not mineru_command:
        if required:
            raise RuntimeError("MinerU is required but mineru_command is empty in config.json")
        report(progress, "[preflight] MinerU - not configured; PDF parser fallback enabled")
        return False
    executable = str(mineru_command[0])
    resolved = _resolve_executable(executable)
    if not resolved:
        message = (
            f"MinerU command not found: {executable}. "
            'Install the paper dependencies with: pip install -e ".[paper]"'
        )
        if required:
            raise RuntimeError(message)
        report(progress, f"[preflight] MinerU - unavailable; {message}")
        return False
    report(progress, f"[preflight] MinerU - PASS ({resolved})")
    return True


def extract_papers(
    papers: Sequence[Paper],
    llm: LLMClient,
    root: str | Path,
    domain: str,
    mineru_command: Optional[Sequence[str]] = None,
    require_mineru: bool = False,
    download_pdfs: bool = False,
    max_full_text_papers: Optional[int] = None,
    max_workers: int = 4,
    progress: ProgressCallback = None,
) -> List[Paper]:
    root = Path(root)
    pdf_dir = root / "pdfs"
    text_dir = root / "full_text"
    full_text_enabled = download_pdfs and max_full_text_papers != 0
    if full_text_enabled:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)

    eligible = [
        paper
        for paper in papers
        if full_text_enabled and not paper.full_text and paper.pdf_url
    ]
    allowed = {
        paper.key
        for paper in (
            eligible
            if max_full_text_papers is None
            else eligible[:max_full_text_papers]
        )
    }
    kwargs = {
        "llm": llm,
        "pdf_dir": pdf_dir,
        "text_dir": text_dir,
        "domain": domain,
        "mineru_command": mineru_command,
        "require_mineru": require_mineru,
        "abstract_only": not full_text_enabled,
    }
    if max_workers == 1:
        output = []
        failures = 0
        for index, paper in enumerate(papers, 1):
            use_full_text = full_text_enabled and (
                bool(paper.full_text) or paper.key in allowed
            )
            report(
                progress,
                f"    paper {index}/{len(papers)} - "
                f"{'full text' if use_full_text else 'abstract'}: "
                f"{paper.title[:70]}",
            )
            try:
                extracted = extract_paper(
                    paper,
                    download_pdfs=use_full_text,
                    **kwargs,
                )
            except Exception as exc:
                failures += 1
                paper.extra["skip_stage"] = "evidence_extraction"
                paper.extra["skip_reason"] = type(exc).__name__
                continue
            output.append(extracted)
            if at_milestone(index, len(papers)):
                report(
                    progress,
                    f"    evidence prepared {index}/{len(papers)}: "
                    f"source={paper.extra.get('evidence_source', 'none')}, "
                    f"chars={paper.extra.get('evidence_characters', 0)} | {paper.title[:70]}",
                )
        if failures:
            report(
                progress,
                f"    evidence extraction skipped {failures} paper(s) with errors",
            )
        if papers and not output:
            raise RuntimeError("Evidence extraction failed for every paper")
        return output

    output: List[Optional[Paper]] = [None] * len(papers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        jobs = {
            executor.submit(
                extract_paper,
                paper,
                download_pdfs=full_text_enabled
                and (bool(paper.full_text) or paper.key in allowed),
                **kwargs,
            ): index
            for index, paper in enumerate(papers)
        }
        processed = 0
        failures = 0
        for future in as_completed(jobs):
            index = jobs[future]
            processed += 1
            try:
                output[index] = future.result()
            except Exception as exc:
                failures += 1
                papers[index].extra["skip_stage"] = "evidence_extraction"
                papers[index].extra["skip_reason"] = type(exc).__name__
                continue
            if at_milestone(processed, len(papers)):
                paper = output[index]
                report(
                    progress,
                    f"    evidence prepared {processed}/{len(papers)}: "
                    f"source={paper.extra.get('evidence_source', 'none')}, "
                    f"chars={paper.extra.get('evidence_characters', 0)} | {papers[index].title[:70]}",
                )
    successful = [paper for paper in output if paper is not None]
    if failures:
        report(
            progress,
            f"    evidence extraction skipped {failures} paper(s) with errors",
        )
    if papers and not successful:
        raise RuntimeError("Evidence extraction failed for every paper")
    return successful


def extract_paper(
    paper: Paper,
    llm: LLMClient,
    pdf_dir: Path,
    text_dir: Path,
    domain: str,
    mineru_command: Optional[Sequence[str]] = None,
    require_mineru: bool = False,
    download_pdfs: bool = False,
    abstract_only: Optional[bool] = None,
) -> Paper:
    if abstract_only is None:
        abstract_only = not download_pdfs
    paper.knowledge_card = {}
    paper.extra["knowledge_card_generated"] = False
    if abstract_only:
        # Disabling PDF mode is an explicit request to use abstracts only.
        # Clear any full text or PDF metadata carried by reused Paper objects so
        # downstream writing cannot accidentally consume stale full-text data.
        text = ""
        paper.full_text = ""
        source = "abstract"
        for key in ("pdf_path", "parsed_text_path", "pdf_error", "pdf_parse_error"):
            paper.extra.pop(key, None)
    else:
        text = paper.full_text
        source = "provided_full_text" if text else ""
    if not text and download_pdfs and paper.pdf_url:
        try:
            pdf_path = _download_pdf(paper, pdf_dir)
            paper.extra["pdf_path"] = str(pdf_path)
        except Exception as exc:
            paper.extra["pdf_error"] = str(exc)
        else:
            try:
                text, source, parsed_path = _parse_pdf(
                    pdf_path, text_dir, mineru_command
                )
                if not _clean_text(text):
                    raise RuntimeError("the parser returned no usable text")
                if parsed_path:
                    paper.extra["parsed_text_path"] = str(parsed_path)
            except Exception as exc:
                paper.extra["pdf_parse_error"] = str(exc)
                if require_mineru:
                    raise RuntimeError(
                        f"MinerU parsing failed for '{paper.title}': {exc}"
                    ) from exc
    paper.full_text = _clean_text(text)
    evidence = paper.full_text or paper.abstract
    if abstract_only:
        paper.extra["evidence_source"] = "abstract"
    elif paper.full_text:
        paper.extra["evidence_source"] = source
    else:
        paper.extra["evidence_source"] = "abstract_fallback"
    paper.extra["evidence_characters"] = len(evidence)
    if paper.full_text:
        extracted = llm.json(
            knowledge_card(domain, paper, paper.full_text[:120_000])
        )
        paper.knowledge_card = normalize_knowledge_card(domain, extracted)
        paper.extra["knowledge_card_generated"] = True
    return paper


def _download_pdf(paper: Paper, pdf_dir: Path) -> Path:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install the 'requests' package to download PDFs") from exc
    name = hashlib.sha256((paper.doi or paper.paper_id or paper.title).encode("utf-8")).hexdigest()[:20]
    path = pdf_dir / f"{name}.pdf"
    if path.exists():
        return path
    response = requests.get(paper.pdf_url, timeout=90)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError("Downloaded content is not a PDF")
    temp = path.with_suffix(".pdf.tmp")
    temp.write_bytes(response.content)
    temp.replace(path)
    return path


def _parse_pdf(
    path: Path,
    text_dir: Path,
    mineru_command: Optional[Sequence[str]],
) -> tuple[str, str, Optional[Path]]:
    if mineru_command:
        output_dir = text_dir / path.stem
        content_path = text_dir / f"{path.stem}.md"
        if content_path.is_file():
            return (
                content_path.read_text(encoding="utf-8", errors="ignore"),
                "mineru_full_text",
                content_path,
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown = _find_mineru_markdown(output_dir, path.stem)
        if markdown is None:
            command = [
                part.format(
                    input=str(path),
                    output=str(output_dir),
                    output_dir=str(output_dir),
                )
                for part in mineru_command
            ]
            resolved = _resolve_executable(command[0])
            if resolved:
                command[0] = resolved
            # Inherit stdout/stderr so MinerU's model-download and per-page
            # progress is visible in the same reproducibility console.
            subprocess.run(command, check=True)
            markdown = _find_mineru_markdown(output_dir, path.stem)
        if markdown is None:
            raise FileNotFoundError(
                f"MinerU produced no Markdown file under {output_dir}"
            )
        text = _clean_text(markdown.read_text(encoding="utf-8", errors="ignore"))
        temp = content_path.with_suffix(".md.tmp")
        temp.write_text(text, encoding="utf-8")
        temp.replace(content_path)
        if output_dir.resolve().parent != text_dir.resolve():
            raise RuntimeError(f"Refusing to clean unexpected MinerU path: {output_dir}")
        shutil.rmtree(output_dir)
        return text, "mineru_full_text", content_path
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf or configure mineru_command for PDF extraction") from exc
    return (
        "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages),
        "pypdf_full_text",
        None,
    )


def _find_mineru_markdown(output_dir: Path, stem: str) -> Optional[Path]:
    candidates = list(output_dir.rglob("*.md"))
    if not candidates:
        return None
    exact = [path for path in candidates if path.stem == stem]
    return max(exact or candidates, key=lambda path: path.stat().st_size)


def _resolve_executable(value: str) -> Optional[str]:
    path = Path(value)
    if path.is_file():
        return str(path)
    found = shutil.which(value)
    if found:
        return found
    python_dir = Path(sys.executable).resolve().parent
    names = [value]
    if sys.platform == "win32" and not value.lower().endswith(".exe"):
        names.append(f"{value}.exe")
    for directory in (python_dir, python_dir / "Scripts"):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    return None


def _clean_text(text: str) -> str:
    lines = []
    for line in str(text or "").splitlines():
        value = " ".join(line.split())
        lower = value.lower()
        if (
            value
            and not lower.startswith(("figure ", "fig. ", "table ", "![", "<img"))
            and not (value.startswith("|") and value.endswith("|"))
        ):
            lines.append(value)
    return "\n".join(lines)
