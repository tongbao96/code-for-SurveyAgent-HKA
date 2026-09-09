from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from .config import PipelineConfig
from .embeddings import TextEmbedder
from .extraction import check_mineru, extract_papers
from .llm import LLMClient, OpenAIClient
from .storage import RunStore
from .models import Outline, Paper, SurveyDraft, normalize_doi, normalize_title
from .outline import generate_outline
from .progress import ProgressCallback, report
from .ranking import rank_papers
from .router import configured_workers, run_agent_tasks
from .retrieval import (
    LiteratureSource,
    build_sources,
    check_semantic_scholar_api,
    co_citation_expand,
    enrich_with_openalex,
    is_citable_paper,
    retrieve_initial,
    retrieve_section,
)
from .review import (
    build_checklist,
    has_actionable_comments,
    polish_draft,
    retrieval_requests,
    review_draft,
    revise_draft,
    score_draft,
)
from .writing import (
    build_citation_plan,
    citation_plan_csv,
    citation_plan_row,
    draft_survey,
)


def run_pipeline(
    config: PipelineConfig,
    *,
    resume: bool = True,
    llm: Optional[LLMClient] = None,
    sources: Optional[Sequence[LiteratureSource]] = None,
    embedder: Optional[TextEmbedder] = None,
    enricher: Optional[Callable[[Sequence[Paper]], List[Paper]]] = None,
    progress: ProgressCallback = None,
) -> SurveyDraft:
    """Run the stages in the order described in the paper."""
    memory = RunStore(config.workspace)
    if not resume:
        memory.reset_state()
    embedder = embedder or TextEmbedder(
        config.embedding_model,
        device=config.embedding_device,
        allow_fallback=not config.require_embedding_model,
    )
    supplied_llm = llm is not None
    if llm is None:
        llm = OpenAIClient(
            config.model,
            config.llm_api_key(),
            config.model.base_url,
        )
    polish_llm = llm
    if not supplied_llm and config.model.polish_model.strip():
        polish_config = replace(
            config.model,
            model=config.model.polish_model.strip(),
        )
        polish_llm = OpenAIClient(
            polish_config,
            config.llm_api_key(),
            polish_config.base_url,
        )
    sources = (
        list(sources)
        if sources is not None
        else build_sources(config.retrieval, progress)
    )
    enrich = enricher or (
        lambda rows: enrich_with_openalex(rows, config.retrieval, progress)
    )
    topic = config.topic
    full_text_enabled = config.download_pdfs and config.max_full_text_papers != 0
    retrieval_workers = configured_workers(
        config.parallel_enabled, config.retrieval_workers
    )
    cluster_workers = configured_workers(config.parallel_enabled, config.cluster_workers)
    section_workers = configured_workers(config.parallel_enabled, config.section_workers)
    review_workers = configured_workers(
        config.parallel_enabled, config.review_retrieval_workers
    )
    extraction_workers = configured_workers(
        config.parallel_enabled, config.extraction_workers
    )
    writing_workers = configured_workers(config.parallel_enabled, config.max_workers)
    polish_workers = configured_workers(config.parallel_enabled, config.polish_workers)
    report(progress, f"SurveyAgent-HKA started | topic: {topic.title}")
    report(progress, f"Output directory: {Path(config.workspace).resolve()}")
    report(
        progress,
        f"[ROUTER] Refiner model={config.model.polish_model} | "
        f"context budget={config.model.polish_context_tokens:,} tokens",
    )
    report(
        progress,
        "[ROUTER] deterministic routing | "
        f"parallel={'enabled' if config.parallel_enabled else 'disabled'} | "
        f"retrieval={retrieval_workers}, clusters={cluster_workers}, "
        f"sections={section_workers}, extraction={extraction_workers}, "
        f"writing={writing_workers}, polish={polish_workers}, "
        f"review retrieval={review_workers}, "
        f"LLM limit={config.model.max_concurrent_requests}",
    )
    memory.event("router", "configured", {
        "mode": "deterministic",
        "parallel_enabled": config.parallel_enabled,
        "retrieval_workers": retrieval_workers,
        "cluster_workers": cluster_workers,
        "section_workers": section_workers,
        "extraction_workers": extraction_workers,
        "writing_workers": writing_workers,
        "polish_workers": polish_workers,
        "review_retrieval_workers": review_workers,
        "llm_max_concurrent_requests": config.model.max_concurrent_requests,
    })
    retrieval_is_checkpointed = resume and all(
        memory.is_completed(stage)
        for stage in ("initial_retrieval", "section_retrieval_and_ranking")
    )
    if "semantic_scholar" in config.retrieval.sources:
        if retrieval_is_checkpointed:
            report(
                progress,
                "[preflight] Semantic Scholar API check - deferred "
                "(retrieval checkpoints will be loaded)",
            )
        else:
            check_semantic_scholar_api(config.retrieval, progress)
    embedder.encode([topic.title])
    report(
        progress,
        f"[preflight] Embedding model - PASS ({config.embedding_model}, "
        f"device={config.embedding_device})",
    )
    if full_text_enabled:
        check_mineru(config.mineru_command, config.require_mineru, progress)
        report(progress, "[preflight] Evidence mode - PDF full text with abstract fallback")
    else:
        report(
            progress,
            "[preflight] Evidence mode - abstracts only; PDF download and MinerU disabled",
        )
    initial = _load_papers(memory, "initial_papers") if resume and memory.is_completed("initial_retrieval") else None
    if initial is None:
        report(progress, "[1/8] Initial literature retrieval - started")
        memory.mark_started("initial_retrieval")
        memory.event("retrieval", "query expansion and multi-source search", {"topic": topic.title})
        initial = retrieve_initial(
            topic,
            sources,
            embedder,
            llm,
            config.retrieval,
            progress,
            retrieval_workers,
            memory.event,
        )
        initial = enrich(initial)
        initial = co_citation_expand(
            initial,
            sources,
            config.retrieval.co_cited_results,
            progress,
            config.retrieval,
        )
        initial = enrich(initial)
        path = memory.save("initial_papers", [paper.to_dict() for paper in initial])
        memory.mark_completed("initial_retrieval", str(path))
        report(progress, f"[1/8] Initial literature retrieval - completed ({len(initial)} papers)")
    else:
        report(progress, f"[1/8] Initial literature retrieval - loaded checkpoint ({len(initial)} papers)")

    outline_data = memory.load("outline") if resume and memory.is_completed("outline_generation") else None
    if outline_data:
        outline = Outline.from_dict(outline_data)
        report(progress, f"[2/8] Outline generation - loaded checkpoint ({len(outline.outline)} sections)")
    else:
        report(progress, "[2/8] Outline generation and human-structure refinement - started")
        memory.mark_started("outline_generation")
        memory.event("outline", "cluster papers and draft outline", {"papers": len(initial), "clusters": config.clusters})
        outline, summaries, related_outlines = generate_outline(
            topic.title,
            initial,
            llm,
            embedder,
            config.clusters,
            config.cluster_min_size,
            config.human_outlines_path,
            config.related_human_outlines,
            topic.domain,
            cluster_workers,
            memory.event,
            progress,
        )
        memory.save("cluster_summaries", summaries)
        memory.save("related_human_outlines", related_outlines)
        path = memory.save("outline", outline.to_dict())
        memory.mark_completed("outline_generation", str(path))
        report(progress, f"[2/8] Outline generation - completed ({len(outline.outline)} sections)")

    ranked_data = memory.load("ranked_papers_by_section") if resume and memory.is_completed("section_retrieval_and_ranking") else None
    if ranked_data:
        papers_by_section = _papers_by_section(ranked_data)
        report(progress, f"[3/8] Section retrieval and ranking - loaded checkpoint ({len(papers_by_section)} sections)")
    else:
        report(progress, "[3/8] Section retrieval and ranking - started")
        memory.mark_started("section_retrieval_and_ranking")
        section_tasks = [
            {
                "id": f"section-{index}",
                "label": section.section_title,
                "payload": section.section_title,
            }
            for index, section in enumerate(outline.outline, 1)
        ]

        def retrieve_for_section(title: str) -> List[Paper]:
            candidates = retrieve_section(
                topic, title, sources, embedder, config.retrieval, progress
            )
            candidates = enrich(candidates)
            candidates = co_citation_expand(
                candidates,
                sources,
                config.retrieval.co_cited_results,
                progress,
                config.retrieval,
            )
            return [paper for paper in enrich(candidates) if is_citable_paper(paper)]

        candidate_batches = run_agent_tasks(
            "RETRIEVER",
            section_tasks,
            retrieve_for_section,
            section_workers,
            progress,
            memory.event,
        )
        ranking_tasks = [
            {
                "id": task["id"],
                "label": task["label"],
                "payload": (task["payload"], candidates),
            }
            for task, candidates in zip(section_tasks, candidate_batches)
        ]

        def rank_section(payload: tuple[str, Sequence[Paper]]) -> List[Paper]:
            title, candidates = payload
            return rank_papers(
                topic,
                title,
                candidates,
                llm,
                config.ranking,
                config.retrieval.final_per_section,
                progress,
            )

        ranked_batches = run_agent_tasks(
            "RANKER",
            ranking_tasks,
            rank_section,
            section_workers,
            progress,
            memory.event,
        )
        ranking_skipped = [
            _paper_skip_record(paper, task["label"])
            for task, candidates in zip(section_tasks, candidate_batches)
            for paper in candidates
            if paper.extra.get("skip_stage") == "ranking"
        ]
        memory.save("ranking_skipped_papers", ranking_skipped)
        papers_by_section = {
            task["label"]: papers
            for task, papers in zip(section_tasks, ranked_batches)
        }
        path = memory.save("ranked_papers_by_section", _serialize_mapping(papers_by_section))
        memory.mark_completed("section_retrieval_and_ranking", str(path))
        report(progress, "[3/8] Section retrieval and ranking - completed")

    extraction_settings = _extraction_settings(config)
    saved_extraction_settings = memory.load("extraction_settings")
    extraction_checkpoint_matches = (
        resume
        and memory.is_completed("information_extraction")
        and saved_extraction_settings == extraction_settings
    )
    if (
        resume
        and memory.is_completed("information_extraction")
        and not extraction_checkpoint_matches
    ):
        report(
            progress,
            "[4/8] Evidence configuration changed - rebuilding knowledge cards "
            "and downstream survey stages",
        )
    extracted_data = (
        memory.load("extracted_papers_by_section")
        if extraction_checkpoint_matches
        else None
    )
    extraction_checkpoint_loaded = bool(extracted_data)
    if extracted_data:
        papers_by_section = _papers_by_section(extracted_data)
        report(progress, "[4/8] Knowledge extraction - loaded matching checkpoint")
    else:
        evidence_mode = "PDF/full text" if full_text_enabled else "abstract only"
        report(progress, f"[4/8] Knowledge extraction ({evidence_mode}) - started")
        memory.mark_started("information_extraction")
        unique = {paper.key: paper for papers in papers_by_section.values() for paper in papers}
        if full_text_enabled:
            report(
                progress,
                f"    {len(unique)} unique papers will be processed; "
                "downloaded PDFs and existing MinerU Markdown are reused",
            )
        else:
            report(
                progress,
                f"    {len(unique)} unique papers will use abstracts directly; "
                "no PDFs will be downloaded or parsed",
            )
        if full_text_enabled and config.max_full_text_papers is not None:
            report(
                progress,
                f"    MinerU test limit: at most {config.max_full_text_papers} PDFs; "
                "the remaining papers use their abstracts",
            )
        memory.event(
            "extraction",
            "prepare paper evidence",
            {
                "unique_papers": len(unique),
                "evidence_mode": "pdf_full_text" if full_text_enabled else "abstract",
            },
        )
        report(
            progress,
            f"[ROUTER] dispatching {len(unique)} paper task(s) to [EXTRACTOR] "
            f"with {extraction_workers} worker(s)",
        )
        memory.event("router", "dispatch", {
            "agent": "EXTRACTOR",
            "sender": "ROUTER",
            "receiver": "EXTRACTOR",
            "tasks": len(unique),
            "workers": extraction_workers,
        })
        extracted = extract_papers(
            list(unique.values()),
            llm,
            Path(config.workspace) / "documents",
            topic.domain,
            config.mineru_command,
            config.require_mineru,
            config.download_pdfs,
            config.max_full_text_papers,
            extraction_workers,
            progress,
        )
        extraction_skipped = [
            _paper_skip_record(paper)
            for paper in unique.values()
            if paper.extra.get("skip_stage") == "evidence_extraction"
        ]
        memory.save("extraction_skipped_papers", extraction_skipped)
        extracted_by_key = {paper.key: paper for paper in extracted}
        papers_by_section = {
            section: [
                extracted_by_key[paper.key]
                for paper in papers
                if paper.key in extracted_by_key
            ]
            for section, papers in papers_by_section.items()
        }
        memory.save("extraction_settings", extraction_settings)
        path = memory.save("extracted_papers_by_section", _serialize_mapping(papers_by_section))
        memory.mark_completed("information_extraction", str(path))
        card_count = sum(bool(paper.knowledge_card) for paper in extracted)
        memory.event("extractor", "completed", {
            "sender": "EXTRACTOR",
            "receiver": "ROUTER",
            "papers": len(extracted),
            "knowledge_cards": card_count,
        })
        report(
            progress,
            f"[4/8] Evidence preparation - completed ({len(unique)} unique papers; "
            f"{card_count} full-text knowledge cards)",
        )

    skipped_papers: List[Dict] = [
        *list(memory.load("ranking_skipped_papers", [])),
        *list(memory.load("extraction_skipped_papers", [])),
    ]
    citation_plan = build_citation_plan(
        outline, papers_by_section, skipped_papers=skipped_papers
    )
    memory.save("skipped_papers", skipped_papers)
    if skipped_papers:
        report(
            progress,
            f"[WRITER] skipped {len(skipped_papers)} paper record(s); "
            "details saved to skipped_papers.json",
        )
    memory.save("citation_plan", citation_plan)
    citation_plan_path = memory.save_text(
        "citation_plan", citation_plan_csv(citation_plan), extension=".csv"
    )
    citation_count = len({int(row["ref_no"]) for row in citation_plan})
    memory.event("writer", "citation plan prepared", {
        "rows": len(citation_plan),
        "unique_references": citation_count,
        "artifact": str(citation_plan_path),
    })
    report(
        progress,
        f"[WRITER] global citation plan saved ({citation_count} references, "
        f"{len(citation_plan)} section assignments)",
    )

    downstream_resume = resume and extraction_checkpoint_loaded
    draft_data = memory.load("draft_survey") if downstream_resume and memory.is_completed("survey_drafting") else None
    if draft_data:
        draft = SurveyDraft.from_dict(draft_data)
        report(progress, f"[5/8] Survey drafting - loaded checkpoint ({len(draft.sections)} subsections)")
    else:
        report(progress, "[5/8] Survey drafting - started")
        memory.mark_started("survey_drafting")
        memory.event("writing", "draft subsections", {"sections": len(outline.outline)})
        subsection_count = sum(
            max(1, len(section.subsections)) for section in outline.outline
        )
        report(
            progress,
            f"[ROUTER] dispatching {subsection_count} subsection task(s) to [WRITER] "
            f"with {writing_workers} worker(s)",
        )
        memory.event("router", "dispatch", {
            "agent": "WRITER",
            "sender": "ROUTER",
            "receiver": "WRITER",
            "tasks": subsection_count,
            "workers": writing_workers,
        })
        draft = draft_survey(
            topic.title,
            outline,
            papers_by_section,
            llm,
            writing_workers,
            max_attempts=config.writer_max_attempts,
            citation_plan=citation_plan,
            progress=progress,
        )
        path = memory.save("draft_survey", draft.to_dict())
        memory.save_text("draft_survey", draft.as_markdown())
        memory.mark_completed("survey_drafting", str(path))
        memory.event("writer", "completed", {
            "sender": "WRITER",
            "receiver": "ROUTER",
            "subsections": len(draft.sections),
        })
        report(progress, f"[5/8] Survey drafting - completed ({len(draft.sections)} subsections)")

    checklist = memory.load("review_checklist") if resume and memory.is_completed("review_checklist") else None
    if checklist is None:
        report(progress, "[6/8] Peer-review knowledge augmentation - started")
        memory.mark_started("review_checklist")
        memory.event("review", "select related peer reviews", {"sources": config.peer_reviews_paths})
        report(progress, "[ROUTER] dispatching checklist extraction to [REVIEWER]")
        memory.event("router", "dispatch", {
            "agent": "REVIEWER",
            "task_id": "review-checklist",
            "sender": "ROUTER",
            "receiver": "REVIEWER",
        })
        checklist, related_reviews = build_checklist(
            topic.title,
            config.peer_reviews_paths,
            llm,
            embedder,
            config.related_review_surveys,
        )
        memory.save("related_peer_reviews", related_reviews)
        path = memory.save("review_checklist", checklist)
        memory.mark_completed("review_checklist", str(path))
        memory.event("reviewer", "completed", {
            "task_id": "review-checklist",
            "sender": "REVIEWER",
            "receiver": "ROUTER",
            "related_surveys": len(related_reviews),
        })
        report(progress, f"[6/8] Peer-review checklist - completed ({len(related_reviews)} related surveys)")
    else:
        report(progress, "[6/8] Peer-review checklist - loaded checkpoint")

    revised_data = memory.load("optimal_revised_survey") if downstream_resume and memory.is_completed("review_and_revision") else None
    if revised_data:
        revised = SurveyDraft.from_dict(revised_data)
        for saved_round in range(1, config.max_revision_rounds + 1):
            saved_additional = memory.load(f"review_additional_{saved_round}")
            if isinstance(saved_additional, Mapping):
                _append_review_citation_plan(
                    memory, saved_additional, saved_round, progress
                )
        report(progress, f"[7/8] Review and revision - loaded checkpoint (round {revised.revision_round})")
    else:
        report(progress, "[7/8] Review and revision - started")
        revised = _revise(
            draft,
            checklist,
            config,
            llm,
            embedder,
            sources,
            enrich,
            memory,
            review_workers,
            progress,
        )
        report(progress, f"[7/8] Review and revision - completed (best round {revised.revision_round})")

    final_data = memory.load("final_survey") if downstream_resume and memory.is_completed("language_polish") else None
    if final_data:
        final = SurveyDraft.from_dict(final_data)
        report(progress, "[8/8] Final language polish - loaded checkpoint")
    else:
        report(progress, "[8/8] Final language polish - started")
        memory.mark_started("language_polish")
        memory.event("writing", "language and citation polish")
        report(progress, "[ROUTER] dispatching final language polish to [REFINER]")
        memory.event("router", "dispatch", {
            "agent": "REFINER",
            "task_id": "language-polish",
            "sender": "ROUTER",
            "receiver": "REFINER",
        })
        final = polish_draft(
            topic.title,
            revised,
            polish_llm,
            max_workers=polish_workers,
            max_context_tokens=config.model.polish_context_tokens,
            progress=progress,
        )
        _validate_final_citations(final)
        path = memory.save("final_survey", final.to_dict())
        memory.save_text("final_survey", final.as_markdown())
        memory.mark_completed("language_polish", str(path))
        memory.event("refiner", "completed", {
            "task_id": "language-polish",
            "sender": "REFINER",
            "receiver": "ROUTER",
        })
        report(progress, "[8/8] Final language polish - completed")
    report(progress, "SurveyAgent-HKA completed successfully")
    return final


def _revise(
    draft: SurveyDraft,
    checklist: Dict,
    config: PipelineConfig,
    llm: LLMClient,
    embedder: TextEmbedder,
    sources: Sequence[LiteratureSource],
    enrich: Callable[[Sequence[Paper]], List[Paper]],
    memory: RunStore,
    review_workers: int = 1,
    progress: ProgressCallback = None,
) -> SurveyDraft:
    resuming_failed_stage = memory.state().get("current") == "review_and_revision"
    memory.mark_started("review_and_revision")
    topic = config.topic
    current = best = draft
    report(
        progress,
        "[ROUTER] controlled review policy | rounds are sequential | "
        f"maximum rounds={config.max_revision_rounds} | "
        "stop on no actionable comments or no score improvement",
    )
    memory.event("router", "review_policy", {
        "mode": "sequential",
        "max_revision_rounds": config.max_revision_rounds,
        "stop_conditions": ["no_actionable_comments", "no_score_improvement"],
    })
    report(progress, "[ROUTER] dispatching initial draft scoring to [REVIEWER]")
    memory.event("router", "dispatch", {
        "agent": "REVIEWER",
        "task_id": "score-draft-0",
        "sender": "ROUTER",
        "receiver": "REVIEWER",
    })
    best_score = score_draft(topic.title, draft, checklist, llm)
    memory.event("reviewer", "completed", {
        "task_id": "score-draft-0",
        "sender": "REVIEWER",
        "receiver": "ROUTER",
        "score": best_score,
    })
    versions = [{"round": 0, "score": best_score, "artifact": "draft_survey"}]

    for round_index in range(1, config.max_revision_rounds + 1):
        report(progress, f"    review round {round_index}/{config.max_revision_rounds}")
        memory.event("review", "review draft", {"round": round_index})
        comments = (
            memory.load(f"review_comments_{round_index}")
            if resuming_failed_stage
            else None
        )
        if comments is None:
            report(
                progress,
                f"[ROUTER] dispatching round {round_index} review to [REVIEWER]",
            )
            memory.event("router", "dispatch", {
                "agent": "REVIEWER",
                "task_id": f"review-round-{round_index}",
                "sender": "ROUTER",
                "receiver": "REVIEWER",
            })
            comments = review_draft(topic.title, current, checklist, llm)
            memory.save(f"review_comments_{round_index}", comments)
            memory.event("reviewer", "completed", {
                "task_id": f"review-round-{round_index}",
                "sender": "REVIEWER",
                "receiver": "ROUTER",
            })
        else:
            report(progress, "      loaded saved review comments")
        if not has_actionable_comments(comments):
            report(progress, "      no actionable comments; stopping revision")
            break

        revision_base_data = (
            memory.load(f"revision_base_{round_index}")
            if resuming_failed_stage
            else None
        )
        additional = (
            memory.load(f"review_additional_{round_index}")
            if resuming_failed_stage
            else None
        )
        if revision_base_data is not None and additional is not None:
            revision_base = SurveyDraft.from_dict(revision_base_data)
            report(progress, "      loaded saved review-triggered literature")
        else:
            revision_base = SurveyDraft.from_dict(current.to_dict())
            additional = _retrieve_for_review(
                comments,
                revision_base,
                round_index,
                config,
                llm,
                embedder,
                sources,
                enrich,
                memory,
                review_workers,
                progress,
            )
            memory.save(f"review_additional_{round_index}", additional)
            memory.save(f"revision_base_{round_index}", revision_base.to_dict())
        _append_review_citation_plan(memory, additional, round_index, progress)
        memory.event("review", "revise draft", {"round": round_index, "additional_groups": len(additional)})
        report(
            progress,
            f"[ROUTER] dispatching round {round_index} revision to [REFINER]",
        )
        memory.event("router", "dispatch", {
            "agent": "REFINER",
            "task_id": f"revise-round-{round_index}",
            "sender": "ROUTER",
            "receiver": "REFINER",
        })
        candidate = revise_draft(topic.title, revision_base, comments, additional, llm)
        _validate_final_citations(candidate)
        memory.event("refiner", "completed", {
            "task_id": f"revise-round-{round_index}",
            "sender": "REFINER",
            "receiver": "ROUTER",
        })
        report(
            progress,
            f"[ROUTER] dispatching round {round_index} scoring to [REVIEWER]",
        )
        memory.event("router", "dispatch", {
            "agent": "REVIEWER",
            "task_id": f"score-draft-{round_index}",
            "sender": "ROUTER",
            "receiver": "REVIEWER",
        })
        score = score_draft(topic.title, candidate, checklist, llm)
        memory.event("reviewer", "completed", {
            "task_id": f"score-draft-{round_index}",
            "sender": "REVIEWER",
            "receiver": "ROUTER",
            "score": score,
        })
        report(progress, f"      candidate score {score:.2f}; previous best {best_score:.2f}")
        memory.save(f"revised_survey_{round_index}", candidate.to_dict())
        versions.append({"round": round_index, "score": score, "artifact": f"revised_survey_{round_index}"})
        if score <= best_score:
            report(progress, "      score did not improve; keeping the previous version")
            break
        current = best = candidate
        best_score = score
        resuming_failed_stage = False

    memory.save("revision_versions", versions)
    path = memory.save("optimal_revised_survey", best.to_dict())
    memory.save_text("optimal_revised_survey", best.as_markdown())
    memory.mark_completed("review_and_revision", str(path))
    return best


def _retrieve_for_review(
    comments: Dict,
    draft: SurveyDraft,
    round_index: int,
    config: PipelineConfig,
    llm: LLMClient,
    embedder: TextEmbedder,
    sources: Sequence[LiteratureSource],
    enrich: Callable[[Sequence[Paper]], List[Paper]],
    memory: RunStore,
    max_workers: int = 1,
    progress: ProgressCallback = None,
) -> Dict:
    requests = retrieval_requests(comments)
    if requests and "semantic_scholar" in config.retrieval.sources:
        check_semantic_scholar_api(config.retrieval, progress)
    tasks = [
        {
            "id": f"round-{round_index}-"
            f"{request['comment_id'] or f'review-retrieval-{index}'}",
            "label": request["topic"],
            "payload": request,
        }
        for index, request in enumerate(requests, 1)
    ]

    def retrieve_missing(request: Dict) -> List[Paper]:
        query = request["topic"]
        papers = retrieve_section(
            config.topic, query, sources, embedder, config.retrieval, progress
        )
        papers = enrich(papers)
        papers = co_citation_expand(
            papers,
            sources,
            config.retrieval.co_cited_results,
            progress,
            config.retrieval,
        )
        return [paper for paper in enrich(papers) if is_citable_paper(paper)]

    candidate_batches = run_agent_tasks(
        "RETRIEVER",
        tasks,
        retrieve_missing,
        max_workers,
        progress,
        memory.event,
    )
    ranking_tasks = [
        {
            "id": task["id"],
            "label": task["label"],
            "payload": (request, papers),
        }
        for task, request, papers in zip(tasks, requests, candidate_batches)
    ]

    def rank_missing(payload: tuple[Dict, Sequence[Paper]]) -> List[Paper]:
        request, papers = payload
        return rank_papers(
            config.topic,
            request["topic"],
            papers,
            llm,
            config.ranking,
            min(10, config.retrieval.final_per_section),
            progress,
        )

    ranked_batches = run_agent_tasks(
        "RANKER",
        ranking_tasks,
        rank_missing,
        max_workers,
        progress,
        memory.event,
    )
    extraction_tasks = [
        {
            "id": task["id"],
            "label": task["label"],
            "payload": papers,
        }
        for task, papers in zip(tasks, ranked_batches)
    ]

    def extract_missing(papers: Sequence[Paper]) -> List[Paper]:
        return extract_papers(
            papers,
            llm,
            Path(config.workspace) / "documents",
            config.topic.domain,
            config.mineru_command,
            config.require_mineru,
            config.download_pdfs,
            None if config.max_full_text_papers is None else 0,
            1,
            progress,
        )

    batches = run_agent_tasks(
        "EXTRACTOR",
        extraction_tasks,
        extract_missing,
        max_workers,
        progress,
        memory.event,
    )

    output: Dict = {}
    next_number = max(
        (int(item.get("refNo", 0)) for item in draft.bibliography), default=0
    ) + 1
    known = {
        key
        for item in draft.bibliography
        for key in _bibliography_keys(item)
    }
    for request, papers in zip(requests, batches):
        query = request["topic"]
        rows = []
        for paper in papers:
            paper_keys = _bibliography_keys(paper.to_dict())
            if paper_keys & known:
                continue
            row = paper.to_dict()
            row["refNo"] = next_number
            draft.bibliography.append({
                "refNo": next_number,
                "paper_id": paper.paper_id,
                "doi": paper.doi,
                "title": paper.title,
                "source": paper.source,
            })
            known.update(paper_keys)
            next_number += 1
            rows.append(row)
        output[request["comment_id"] or query] = {
            "target": request["target"],
            "papers": rows,
        }
    return output


def _bibliography_keys(item: Mapping[str, object]) -> set[str]:
    output = set()
    doi = normalize_doi(item.get("doi"))
    if doi:
        output.add(f"doi:{doi}")
    title = normalize_title(item.get("title"))
    if title:
        output.add(f"title:{title}")
    return output


def _append_review_citation_plan(
    memory: RunStore,
    additional: Mapping[str, object],
    round_index: int,
    progress: ProgressCallback = None,
) -> None:
    plan = list(memory.load("citation_plan", []))
    existing = {
        (int(row.get("ref_no", 0)), str(row.get("section_title") or ""))
        for row in plan
        if isinstance(row, Mapping)
    }
    added = 0
    for group in additional.values():
        if not isinstance(group, Mapping):
            continue
        target = str(group.get("target") or "Review-added literature")
        papers = group.get("papers") or []
        for rank, value in enumerate(papers, 1):
            if not isinstance(value, Mapping):
                continue
            ref_no = int(value.get("refNo") or 0)
            if ref_no < 1 or (ref_no, target) in existing:
                continue
            plan.append(citation_plan_row(
                Paper.from_dict(dict(value)),
                ref_no,
                target,
                subsections=[target],
                rank_in_section=rank,
                selected_stage=f"review_round_{round_index}",
            ))
            existing.add((ref_no, target))
            added += 1
    if not added:
        return
    memory.save("citation_plan", plan)
    memory.save_text("citation_plan", citation_plan_csv(plan), extension=".csv")
    report(
        progress,
        f"      citation plan updated with {added} review-triggered assignment(s)",
    )


def _paper_skip_record(
    paper: Paper, section_title: str = ""
) -> Dict[str, object]:
    return {
        "stage": str(paper.extra.get("skip_stage") or "paper_processing"),
        "section_title": section_title,
        "paper_id": paper.paper_id,
        "doi": paper.doi,
        "title": paper.title,
        "source": paper.source,
        "reason": str(paper.extra.get("skip_reason") or "processing error"),
    }


def _load_papers(memory: RunStore, artifact: str) -> Optional[List[Paper]]:
    value = memory.load(artifact)
    return [Paper.from_dict(item) for item in value] if value is not None else None


def _serialize_mapping(value: Mapping[str, Sequence[Paper]]) -> Dict:
    return {key: [paper.to_dict() for paper in rows] for key, rows in value.items()}


def _extraction_settings(config: PipelineConfig) -> Dict:
    if not config.download_pdfs:
        return {
            "download_pdfs": False,
            "max_full_text_papers": 0,
            "require_mineru": False,
            "mineru_command": None,
        }
    return {
        "download_pdfs": True,
        "max_full_text_papers": config.max_full_text_papers,
        "require_mineru": config.require_mineru,
        "mineru_command": list(config.mineru_command or []),
    }


def _papers_by_section(value: Mapping[str, Sequence[Dict]]) -> Dict[str, List[Paper]]:
    return {
        key: [Paper.from_dict(paper) for paper in rows]
        for key, rows in value.items()
    }


def _validate_final_citations(draft: SurveyDraft) -> None:
    import re

    allowed = {int(item.get("refNo", 0)) for item in draft.bibliography}
    citation_count = 0
    for section in draft.sections:
        cited = {int(number) for number in re.findall(r"\[ref:(\d+)\]", section.content)}
        citation_count += len(cited)
        heading = f"{section.section_title} {section.subsection_title}".lower()
        synthesis_only = any(
            name in heading
            for name in ("introduction", "conclusion", "future direction")
        )
        if section.content.strip() and not cited and not synthesis_only:
            raise ValueError(f"Final survey contains no citations in '{section.subsection_title}'")
        unknown = cited - allowed
        if unknown:
            raise ValueError(
                f"Final survey contains unknown citations {sorted(unknown)} "
                f"in '{section.subsection_title}'"
            )
    if draft.sections and citation_count == 0:
        raise ValueError("Final survey contains no citations")
