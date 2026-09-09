from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

from .embeddings import TextEmbedder, cluster_vectors
from .human_knowledge import load_human_outlines, select_related
from .llm import LLMClient
from .models import Outline, Paper
from .progress import ProgressCallback, report
from .router import AgentEvent, run_agent_tasks
from . import prompts


def generate_outline(
    topic: str,
    papers: Sequence[Paper],
    llm: LLMClient,
    embedder: TextEmbedder,
    cluster_count: int = 6,
    cluster_min_size: int = 5,
    human_outlines_path: str | None = None,
    human_limit: int = 10,
    domain: str = "",
    cluster_workers: int = 1,
    event: AgentEvent = None,
    progress: ProgressCallback = None,
) -> tuple[Outline, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build an outline from paper clusters and optional human examples."""
    usable = [paper for paper in papers if paper.abstract.strip()]
    if not usable:
        raise ValueError("Outline generation requires papers with abstracts")

    actual_clusters = max(1, min(cluster_count, len(usable) // cluster_min_size))
    vectors = embedder.encode([paper.abstract for paper in usable])
    labels = cluster_vectors(vectors, actual_clusters, min_size=cluster_min_size)
    grouped: Dict[int, List[Paper]] = defaultdict(list)
    for paper, label in zip(usable, labels):
        grouped[label].append(paper)

    cluster_labels = sorted(grouped)
    sizes = [len(grouped[label]) for label in cluster_labels]
    report(
        progress,
        f"    minimum-size constrained clustering: {len(usable)} abstracts -> "
        f"{len(cluster_labels)} groups; sizes={sizes}",
    )
    tasks = []
    for cluster_index, label in enumerate(cluster_labels, 1):
        cluster_papers = _fit_papers(grouped[label], max_characters=70_000)
        tasks.append({
            "id": f"cluster-{label + 1}",
            "label": f"cluster {cluster_index}/{len(cluster_labels)} "
            f"({len(cluster_papers)} papers)",
            "payload": (label, cluster_papers),
        })

    def summarize(payload: tuple[int, Sequence[Paper]]) -> Dict[str, Any]:
        label, cluster_papers = payload
        value = llm.json(
            prompts.cluster_summary(
                topic,
                cluster_papers,
                cluster_title=f"Cluster {label + 1}",
            )
        )
        return {
            "cluster": label,
            "cluster_title": str(value.get("cluster_title", f"Cluster {label + 1}")),
            "summary": str(value.get("summary") or value.get("Summary") or ""),
            "paper_ids": [paper.paper_id for paper in grouped[label]],
        }

    summaries = run_agent_tasks(
        "OUTLINER",
        tasks,
        summarize,
        cluster_workers,
        progress,
        event,
    )

    report(progress, "[ROUTER] dispatching initial outline synthesis to [OUTLINER]")
    if event:
        event("router", "dispatch", {
            "agent": "OUTLINER",
            "task_id": "initial-outline",
            "sender": "ROUTER",
            "receiver": "OUTLINER",
        })
    initial = Outline.from_dict(llm.json(prompts.generate_outline(topic, summaries)))
    if not initial.outline:
        raise ValueError("Outline generation returned an empty outline")
    if event:
        event("outliner", "completed", {
            "task_id": "initial-outline",
            "sender": "OUTLINER",
            "receiver": "ROUTER",
        })
    if not human_outlines_path:
        report(progress, "    no human outline corpus configured; keeping the initial outline")
        return initial, summaries, []

    examples = load_human_outlines(human_outlines_path, domain, progress)
    matches = select_related(
        topic, examples, embedder, human_limit, title_only=True
    )
    related = [
        {
            "title": item.get("title", ""),
            "outline": item.get("outline", []),
            "similarity": item.get("similarity", 0),
        }
        for item in matches
    ]
    report(progress, f"[RETRIEVER] selected {len(related)} related human-written outlines")
    for index, item in enumerate(related, 1):
        report(
            progress,
            f"      outline example {index}/{len(related)}: "
            f"similarity={item.get('similarity', 0):.4f} | {item.get('title', '')[:90]}",
        )
    if not related:
        return initial, summaries, related
    report(progress, "[ROUTER] dispatching human-guided outline refinement to [OUTLINER]")
    if event:
        event("router", "dispatch", {
            "agent": "OUTLINER",
            "task_id": "refine-outline",
            "sender": "ROUTER",
            "receiver": "OUTLINER",
        })
    refined = Outline.from_dict(llm.json(prompts.refine_outline(topic, initial, related)))
    if event:
        event("outliner", "completed", {
            "task_id": "refine-outline",
            "sender": "OUTLINER",
            "receiver": "ROUTER",
        })
    return (refined if refined.outline else initial), summaries, related


def _fit_papers(papers: Sequence[Paper], max_characters: int) -> List[Paper]:
    output, used = [], 0
    for paper in papers:
        size = len(paper.title) + len(paper.abstract)
        if output and used + size > max_characters:
            break
        output.append(paper)
        used += size
    return output
