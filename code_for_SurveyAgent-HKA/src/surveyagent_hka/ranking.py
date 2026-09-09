from __future__ import annotations

import math
import re
from typing import List, Sequence

import numpy as np

from .config import RankingConfig, TopicConfig
from .llm import LLMClient
from .models import Paper
from .progress import ProgressCallback, at_milestone, report
from .prompts import topical_relevance


def rank_papers(
    topic: TopicConfig,
    section: str,
    papers: Sequence[Paper],
    llm: LLMClient,
    config: RankingConfig,
    limit: int,
    progress: ProgressCallback = None,
) -> List[Paper]:
    """Apply the relevance, impact, and recent-popularity ranking in the paper."""
    candidates = [paper for paper in papers if paper.abstract.strip()]
    total = len(candidates)
    report(progress, f"      [rank 1/4] topical relevance: LLM scoring {total} papers")
    scored: List[Paper] = []
    scoring_failures = 0
    for index, paper in enumerate(candidates, 1):
        try:
            response = llm.json(topical_relevance(topic.title, section, paper))
        except Exception as exc:
            scoring_failures += 1
            paper.extra["skip_stage"] = "ranking"
            paper.extra["skip_reason"] = f"topical relevance: {type(exc).__name__}"
            continue
        match = re.search(r"[1-5]", str(response.get("score")))
        paper.topical_relevance = float(match.group(0)) if match else 1.0
        scored.append(paper)
        if at_milestone(index, total):
            report(progress, f"      relevance scoring {index}/{total}: {paper.title[:70]}")
    if scoring_failures:
        report(
            progress,
            f"      ranking skipped {scoring_failures} paper(s) that failed "
            "topical-relevance scoring",
        )
    if candidates and not scored:
        raise RuntimeError("Topical-relevance scoring failed for every candidate paper")

    metrics = []
    metadata_failures = 0
    for paper in scored:
        try:
            citation = _citation_count_at(paper, topic.year_end)
            author = float(paper.author_h_index or 0)
            venue = float(paper.venue_h_index or 0)
            recent = sum(
                float(paper.citations_by_year.get(str(year), 0) or 0)
                for year in range(topic.year_end - 2, topic.year_end + 1)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            metadata_failures += 1
            paper.extra["skip_stage"] = "ranking"
            paper.extra["skip_reason"] = f"invalid ranking metadata: {type(exc).__name__}"
            continue
        metrics.append((paper, citation, author, venue, recent))
    if metadata_failures:
        report(
            progress,
            f"      ranking skipped {metadata_failures} paper(s) with invalid metadata",
        )
    if scored and not metrics:
        raise RuntimeError("Ranking metadata is invalid for every scored paper")

    candidates = [row[0] for row in metrics]
    citation = [row[1] for row in metrics]
    author = [row[2] for row in metrics]
    venue = [row[3] for row in metrics]
    recent_counts = [row[4] for row in metrics]
    total = len(candidates)
    author_coverage = sum(paper.author_h_index is not None for paper in candidates)
    venue_coverage = sum(paper.venue_h_index is not None for paper in candidates)
    report(
        progress,
        "      [rank 2/4] academic impact: "
        f"citation={config.citation_weight:.2f}, author={config.author_weight:.2f}, "
        f"venue={config.venue_weight:.2f} (raw indicators, as defined in the paper); "
        f"author metadata={author_coverage}/{total}, "
        f"venue metadata={venue_coverage}/{total}",
    )
    for index, paper in enumerate(candidates):
        paper.extra["citation_count_at_cutoff"] = citation[index]
        paper.academic_impact = (
            config.citation_weight * citation[index]
            + config.author_weight * author[index]
            + config.venue_weight * venue[index]
        )
        recent = recent_counts[index]
        paper.extra["citations_last_3_years"] = recent
        paper.recent_popularity = recent * math.log(
            max(1.0, citation[index])
        )

    report(
        progress,
        "      [rank 3/4] recent popularity: "
        f"citations in {topic.year_end - 2}-{topic.year_end} x log(total citations)",
    )
    topical_ranks = _descending_ranks([paper.topical_relevance or 0 for paper in candidates])
    impact_ranks = _descending_ranks([paper.academic_impact or 0 for paper in candidates])
    recent_ranks = _descending_ranks([paper.recent_popularity or 0 for paper in candidates])
    for index, paper in enumerate(candidates):
        paper.extra["topical_rank"] = topical_ranks[index]
        paper.extra["academic_impact_rank"] = impact_ranks[index]
        paper.extra["recent_popularity_rank"] = recent_ranks[index]
        paper.final_rank_score = (topical_ranks[index] + impact_ranks[index] + recent_ranks[index]) / 3.0
    ranked = sorted(
        candidates, key=lambda paper: (paper.final_rank_score or math.inf, paper.title)
    )[:limit]
    report(
        progress,
        f"      [rank 4/4] equal rank aggregation: "
        f"(Rank_t + Rank_a + Rank_r) / 3; retained {len(ranked)} papers",
    )
    for index, paper in enumerate(ranked[:5], 1):
        report(
            progress,
            f"        top {index}: final={paper.final_rank_score:.2f}, "
            f"Rt={paper.extra['topical_rank']:.1f}, "
            f"Ra={paper.extra['academic_impact_rank']:.1f}, "
            f"Rr={paper.extra['recent_popularity_rank']:.1f} | {paper.title[:70]}",
        )
    return ranked


def _citation_count_at(paper: Paper, year_end: int) -> float:
    if not paper.citations_by_year:
        return float(paper.citation_count or 0)
    return float(sum(
        count for year, count in paper.citations_by_year.items()
        if str(year).isdigit() and int(year) <= year_end
    ))


def _descending_ranks(values: Sequence[float]) -> List[float]:
    """Average ranks with rank 1 assigned to the largest value."""
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks
