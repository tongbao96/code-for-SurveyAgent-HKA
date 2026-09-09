from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

from .models import Outline, Paper, SurveyDraft, SurveySection


# The task prompts below follow paper/prompt.tex. JSON examples are made valid
# where the typeset appendix uses ellipses or trailing commas.


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def classify_survey(title: str, abstract: str) -> str:
    return f"""You are an expert in academic research and literature-type classification. Your task is to judge if a given paper is a survey-type paper based on its title and abstract, following the criteria below:

(i) The title and abstract clearly indicate a systematic overview and summary of a specific research area.
(ii) The paper focuses on summarizing prior studies rather than introducing new techniques or empirical findings.

Title: {title}
Abstract: {abstract}

Your output should be only one word: "TRUE" if the paper qualifies as a survey-type article, or "FALSE" if it does not. Do not include any additional commentary, explanation, or formatting."""


def expand_search_queries(
    topic: str,
    keywords: Sequence[str],
    number: int,
) -> str:
    output_format = {
        "queries": ["concise natural-language academic search phrase"]
    }
    return f"""You are an academic expert in the field of "{topic}" with extensive experience in scientific literature retrieval. Your task is to generate {number} concise academic database queries that retrieve research papers directly relevant to the survey topic.

The survey topic is:
{topic}

The user-provided keywords are:
{_json(list(keywords))}

Please follow the steps below:

1. Identify the central research problem, principal methods, application settings, and closely related technical concepts expressed by the topic and keywords.
2. Generate distinct query phrases that improve literature coverage without changing the scope of the topic.
3. Write every query as a natural-language keyword phrase suitable for Semantic Scholar, arXiv, or PubMed.
4. Do not use SQL, database field names, Boolean operators, wildcards, code, survey, review, or explanatory text in a query.

Your output must follow this JSON format and contain exactly {number} queries:
{_json(output_format)}

Generate concise academic database queries now. Return only the JSON object. Do not include explanations, commentary, Markdown fences, or a preamble."""


def cluster_summary(
    topic: str,
    papers: Sequence[Paper],
    cluster_title: str = "Cluster",
) -> str:
    rows = [
        {"paper_number": index, "title": paper.title, "abstract": paper.abstract}
        for index, paper in enumerate(papers, 1)
    ]
    output_format = {
        "cluster_title": "Cluster Title",
        "Summary": "The cluster summary here",
    }
    return f"""You are an expert academic researcher specializing in "{topic}". Your task is to synthesize a cohesive cluster summary for a group of research papers based on their titles, abstracts, and the provided cluster title.

The cluster title is:
{cluster_title}

The papers within this cluster are:
{_json(rows)}

Please follow the steps below to generate the cluster summary:

Step 1 - Analyze individual abstracts: Carefully read each provided abstract to identify the specific problem addressed, the methodology used, and the primary contribution of each paper.

Step 2 - Identify cross-paper relationships: Determine how the papers relate to one another. Identify common technical themes and approaches, or explain how they represent an evolution of ideas within the cluster.

Step 3 - Synthesize into a unified summary: Write a cohesive summary that integrates the findings of all papers. Avoid simply listing them; instead, use synthesis such as "Recent works [1, 2] focus on X, while [3] introduces a novel perspective on Y."

Your output must follow this JSON format:
{_json(output_format)}

Now generate a summary based on the given paper abstracts. Do not include explanations, commentary, Markdown fences, or a preamble."""


def generate_outline(
    topic: str,
    cluster_summaries: Sequence[Dict[str, Any]],
) -> str:
    output_format = {
        "title": "TITLE OF THE SURVEY",
        "outline": [
            {
                "section_title": "SECTION TITLE",
                "description": "A brief description of this section.",
                "subsections": {
                    "subsection title": "",
                },
            }
        ],
    }
    return f"""You are an academic expert in the field of "{topic}" with deep expertise in scientific survey writing. Your task is to generate a well-structured survey outline for this topic.

Below are summaries of clusters generated from a targeted literature search related to the survey topic. Each summary is generated from the abstracts of all candidate papers within a specific cluster and provides an overview of what that cluster covers.

Cluster summaries:
{_json(cluster_summaries)}

Generate the outline structure based on the provided cluster summaries. The outline should include:

1. Sections: Each section should represent a distinct aspect of the topic, closely aligned with the themes of the clusters. Sections should cover broad areas or themes that reflect the key findings from the clusters.
2. Subsections: For each section, identify specific subsections that break down the theme into manageable parts. Each subsection should provide additional details or key points relevant to the section.
3. Description: Provide a brief description for each section to define its scope and what will be covered. Ensure that the sections and subsections remain closely tied to the topic's central theme and are derived from the cluster summaries.

Your output must follow this JSON format:
{_json(output_format)}

Ensure that the outline includes all relevant sections and subsections, with each section flowing logically and cohesively. Do not include explanations, commentary, Markdown fences, or a preamble."""


def refine_outline(
    topic: str,
    initial: Outline,
    human_outlines: Sequence[Dict[str, Any]],
) -> str:
    output_format = {
        "title": "TITLE OF THE SURVEY",
        "outline": [
            {
                "section_title": "SECTION TITLE",
                "description": "A brief description of this section.",
                "subsections": {
                    "subsection title": "",
                },
            }
        ],
    }
    return f"""You are an academic expert in the field of "{topic}" with deep expertise in scientific survey writing. Your task is to optimize the structure of the provided survey outline based on the provided gold-standard outlines.

The initial outline is:
{_json(initial.to_dict())}

The following highly relevant human-written survey outlines, selected from closely related topics, are provided for reference:
{_json(human_outlines)}

Optimize the current initial outline according to the following instructions:

1. Analyze the Human-Written Outlines:
   (a) Analyze how the themes and findings are presented.
   (b) Analyze how the sections flow logically.
   (c) Analyze the common section setup, especially the distribution of first-level and second-level headings.

2. Refine the Initial Outline:
   (a) Enhance Logical Flow: Rearrange sections or subsections if necessary to improve the logic of the outline.
   (b) Improve Coverage: Add missing sections or subsections that are critical for a complete overview of the topic.
   (c) Ensure Clarity: Simplify complex sections and ensure that the content is clearly conveyed.

3. Maintain Alignment with the Topic: Do not directly copy from the existing outlines. Ensure that the optimized outline remains aligned with the core focus of "{topic}".

4. Your output must follow this JSON format:
{_json(output_format)}

Do not include explanations, commentary, Markdown fences, or a preamble."""


def topical_relevance(topic: str, section: str, paper: Paper) -> str:
    output_format = {"score": 1}
    return f"""You are an academic expert in the field of "{topic}" with extensive experience in scientific literature selection. Your task is to assess the topical relevance of a candidate paper to a specific section of the survey.

The target survey section is:
{section}

The candidate paper is:
Title: {paper.title}
Abstract: {paper.abstract}

Rate how relevant the paper is to the requested survey section according to the following criteria:

1: Unrelated to the section.
2: Loosely connected but not useful for the main discussion.
3: Relevant to a useful sub-aspect of the section.
4: Substantially relevant and suitable for the section.
5: Central, representative, or foundational to the section.

Base the score only on the supplied title and abstract. Do not use external information.

Your output must follow this JSON format:
{_json(output_format)}

Return only the JSON object. Do not include explanations, commentary, Markdown fences, or a preamble."""


COMPUTER_SCIENCE_KNOWLEDGE_TREE: Dict[str, Dict[str, str]] = {
    "Background": {
        "Definition": "Specific description of the problem.",
        "Key Obstacle": "Main difficulty or main challenge.",
    },
    "Idea": {
        "Intuition": "What inspired the idea.",
        "Opinion": "What the idea is.",
        "Innovation": (
            "The main difference from previous methods, or the primary improvement."
        ),
    },
    "Method": {
        "Method Definition": "Given the problem, the definition of the method.",
        "Method Description": "Describe the method in one sentence.",
        "Method Steps": "Procedures of the method.",
        "Principle": "Why this method is effective.",
    },
    "Experiments": {
        "Experiment Setting": "Dataset, baselines, and evaluation metrics.",
        "Experiment Results": (
            "Summarize the main experimental results and findings in one paragraph."
        ),
    },
    "Discussion": {
        "Advantage": "The advantages of this paper.",
        "Limitation": "The disadvantages or limitations of this paper.",
        "Future Work": (
            "Based on the advantages and disadvantages, what can be improved "
            "and where future improvements can be made."
        ),
    },
}


MEDICINE_KNOWLEDGE_TREE: Dict[str, Dict[str, str]] = {
    "Introduction": {
        "Disease/Condition Overview": (
            "Briefly describe the disease or condition under study."
        ),
        "Key Challenge/Research Gap": (
            "The key issue or research gap being addressed."
        ),
    },
    "Methods": {
        "Study Design": "The study design, such as a clinical trial or cohort study.",
        "Sample/Participants": (
            "The inclusion and exclusion criteria for the study sample."
        ),
        "Intervention/Procedure": (
            "The treatments, interventions, or procedures used in the study."
        ),
    },
    "Results": {
        "Main Findings": "Summarize the key results and findings of the study.",
        "Statistical Significance": "The statistical results.",
    },
    "Discussion": {
        "Strengths and Limitations": (
            "The strengths and limitations of the study."
        ),
        "Future Directions": (
            "Suggested future research directions or areas for improvement."
        ),
    },
}


def knowledge_card_tree(domain: str) -> Dict[str, Dict[str, str]]:
    if domain.lower() in {"medicine", "medical", "biomedicine", "biomedical"}:
        return MEDICINE_KNOWLEDGE_TREE
    return COMPUTER_SCIENCE_KNOWLEDGE_TREE


def knowledge_card_template(domain: str) -> Dict[str, Dict[str, None]]:
    return {
        category: {attribute: None for attribute in attributes}
        for category, attributes in knowledge_card_tree(domain).items()
    }


def normalize_knowledge_card(
    domain: str,
    value: Any,
) -> Dict[str, Dict[str, Any]]:
    raw: Mapping[str, Any] = value if isinstance(value, Mapping) else {}
    for wrapper in (
        "Knowledge Card",
        "knowledge_card",
        "Computer Science",
        "Medicine",
    ):
        nested = _case_insensitive_get(raw, wrapper)
        if isinstance(nested, Mapping):
            raw = nested
            break

    normalized: Dict[str, Dict[str, Any]] = {}
    for category, attributes in knowledge_card_tree(domain).items():
        raw_category = _case_insensitive_get(raw, category)
        if not isinstance(raw_category, Mapping):
            raw_category = {}
        normalized[category] = {}
        for attribute in attributes:
            item = _case_insensitive_get(raw_category, attribute)
            normalized[category][attribute] = (
                None if item in ("", [], {}) else item
            )
    return normalized


def _case_insensitive_get(value: Mapping[str, Any], key: str) -> Any:
    target = key.strip().casefold()
    for current, item in value.items():
        if str(current).strip().casefold() == target:
            return item
    return None


def knowledge_card(domain: str, paper: Paper, full_text: str) -> str:
    return f"""You are an expert academic researcher in the {domain.replace("_", " ")} domain. Extract a structured Knowledge Card from the supplied paper full text using the predefined knowledge extraction tree from the SurveyAgent-HKA appendix.

The extraction attributes and their definitions are:
{_json(knowledge_card_tree(domain))}

Requirements:
1. Read the complete supplied text and extract information for every attribute.
2. Use only information explicitly supported by the supplied paper. Do not add external facts or assumptions.
3. Keep the extracted content concise but retain concrete methodological and experimental details.
4. For Experiment Setting, explicitly extract the dataset, baselines, and evaluation metrics when they are available.
5. For Experiment Results, summarize the main results and findings in one paragraph.
6. Use null when the paper does not provide an attribute.

Your output must contain exactly this JSON tree, replacing null values only when corresponding evidence is available:
{_json(knowledge_card_template(domain))}

Paper title: {paper.title}
Paper full text:
{full_text}

Return only the JSON object. Do not include explanations, commentary, Markdown fences, or a preamble."""


def draft_subsection(
    topic: str,
    outline: Outline,
    section: str,
    subsection: str,
    papers: Sequence[Paper],
    reference_numbers: Dict[str, int] | None = None,
) -> str:
    reference_numbers = reference_numbers or {
        paper.key: index for index, paper in enumerate(papers, 1)
    }
    citation_rows = [
        {
            "ref_no": reference_numbers[paper.key],
            "section": section,
            "paper_id": paper.paper_id,
            "doi": paper.doi,
            "title": paper.title,
            "evidence": paper.evidence_text(),
        }
        for paper in papers
    ]
    allowed_numbers = [row["ref_no"] for row in citation_rows]
    example_number = allowed_numbers[0] if allowed_numbers else "N"
    output_format = {
        subsection: (
            "Generated content here with citations such as "
            f"ref [{example_number}]."
        )
    }
    return f"""You are an academic expert in the field of "{topic}" with deep expertise in scientific survey writing. Your task is to write a subsection of a survey based on the following information.

The overall structure of the survey is:
{_json(outline.to_dict())}

Parent section: {section}
Your specific task is to write the subsection titled:
{subsection}

The following rows are the only entries from the global citation plan assigned to this section:
{_json(citation_rows)}

Instructions for generating subsection content:

1. Generate the content solely from the provided references. Do not incorporate or cite any additional external sources.
2. The content must be relevant to the specific topic of the subsection and should be at least 300 words long.
3. All claims must be supported by appropriate references from the provided papers. The only allowed reference numbers are {allowed_numbers}. Format in-text citations exactly as ref [N], using N from that list. Every paragraph must contain at least one such inline citation; a separate references field is not a substitute. Cite a source by its reference number only the first time it is mentioned.
4. Ensure that the content aligns with the parent section and the overall theme of the survey, maintaining conceptual and thematic consistency throughout.
5. Adopt a formal academic tone and present well-reasoned arguments in a logical structure using appropriate scholarly language.
6. The content must follow this JSON format:
{_json(output_format)}

Now generate the content for the subsection "{subsection}". Do not include explanations, commentary, Markdown fences, or a preamble."""


def extract_checklist(
    topic: str,
    reviews: Sequence[Dict[str, str]],
) -> str:
    output_format = {
        "Literature Coverage and Relevance": [
            "Does the survey cover a broad and relevant range of literature sources, including seminal and recent works in the field?"
        ],
        "Structure Depth and Coherence": [
            "Does the manuscript include all essential sections in a logical order?"
        ],
        "Writing Quality and Clarity": [
            "Is the writing clear, concise, and free from redundancy?"
        ],
        "Critical Analysis and Future Outlook": [
            "Does the manuscript adequately discuss limitations and future directions?"
        ],
    }
    return f"""You are an expert in "{topic}" with substantial experience in peer reviewing scientific surveys. Your task is to review peer review comments from published surveys and create a checklist of common criteria that reviewers typically use to evaluate manuscripts. This checklist will serve as a standard for assessing the quality of future submissions.

Peer reviews from the top related published surveys:
{_json(reviews)}

Follow these steps to extract common themes and generate the checklist:

1. Review the provided expert feedback carefully and identify issues that are frequently mentioned by reviewers.
2. Summarize and categorize the comments into the following groups. Each group should contain multiple suggestions derived from the real peer review comments:
   - Literature Coverage and Relevance: Ensure that relevant literature is covered, including both seminal works and recent studies, and is well integrated into the discussion.
   - Structure Depth and Coherence: Assess whether the survey includes all necessary sections and is well organized and logically structured.
   - Writing Quality and Clarity: Check that the manuscript is clearly written, concise, and free from redundancies or unclear phrasing.
   - Critical Analysis and Future Outlook: Ensure that the survey discusses existing issues, acknowledges limitations of the field, and provides clear directions for future research.
3. Construct a checklist for each category. Each item should be concise and easy to implement.

Your output must follow this JSON format:
{_json(output_format)}

Return only the JSON object. Do not include explanations, commentary, Markdown fences, or a preamble."""


def review_survey(
    topic: str,
    draft: SurveyDraft,
    checklist: Dict[str, Any],
) -> str:
    output_format = {
        "Reference Modifications": [
            {
                "Comments ID": "C01",
                "Trigger for further retrieval": True,
                "Topic of literature to add": "Specific missing literature topic",
                "Location to add": "Exact section or paragraph",
                "Explanation": "Why this literature is important, in one sentence.",
            }
        ],
        "Structural Modifications": [
            {
                "Comments ID": "C02",
                "Section/Paragraph to adjust": "Exact section or paragraph",
                "Proposed adjustment": "The proposed structural change",
                "Explanation": "Why this structural change is necessary.",
            }
        ],
        "Writing Modifications": [
            {
                "Comments ID": "C03",
                "Section/Paragraph to modify": "Exact section and paragraph",
                "Problematic content": "Exact text from the manuscript",
                "Explanation": "Why this modification is necessary.",
            }
        ],
        "Other Suggestions": [
            {
                "Comments ID": "C04",
                "Suggestion": "An additional suggestion",
                "Explanation": "Why this suggestion improves the draft.",
            }
        ],
    }
    return f"""You are an expert in "{topic}" with extensive experience in reviewing scientific surveys. Carefully assess the following initial draft survey and provide detailed, constructive revision comments. The review should help the draft meet high academic standards and improve its clarity, depth, and overall quality.

Reviewer checklist:
{_json(checklist)}

Initial draft survey:
{draft.as_markdown()}

When providing revision comments, follow this strict JSON format:
{_json(output_format)}

Provide unique comment IDs. Set "Trigger for further retrieval" to true only when verified literature is missing, and specify both the literature topic and exact location. Return only the JSON object without explanations, commentary, Markdown fences, or a preamble."""


def revise_survey(
    topic: str,
    draft: SurveyDraft,
    comments: Dict[str, Any],
    added_literature: Dict[str, Any],
) -> str:
    output_format = {
        "Revised Survey": {
            "title": "The title of the survey",
            "sections": [
                {
                    "section_title": "Section title",
                    "subsection_title": "Subsection title",
                    "content": "Fully revised subsection content with existing ref [N] citations.",
                    "references": [{"refNo": 1}],
                }
            ],
        },
        "Revision Log": [
            {
                "Comments ID": "C01",
                "Modification Summary": "A brief summary of the change.",
                "Location of Modification": "Exact section or paragraph modified.",
                "Status": "Implemented",
                "Notes (if any)": "",
            }
        ],
    }
    return f"""You are an expert in "{topic}" with extensive experience in reviewing scientific surveys. Your task is to revise the following initial draft survey based on the provided revision suggestions. The goal is to improve the quality, clarity, and academic rigor of the manuscript.

Initial draft survey:
{_json(draft.to_dict())}

Revision suggestions:
{_json(comments)}

Retrieved and verified literature that may need to be added:
{_json(added_literature)}

Follow these steps:

1. Carefully review the revision suggestions to identify the specific changes needed in the draft.
2. Implement modifications strictly based on the provided suggestions and verified literature. Do not make adjustments beyond the suggestions or include any external literature. Keep all revisions aligned with the survey topic.
3. Prioritize critical issues such as structure, clarity, and missing literature before less urgent writing improvements.
4. Group overlapping or repetitive suggestions to avoid redundancy and ensure consistency.
5. Review the revised survey to ensure that ideas flow logically and the overall structure is cohesive.
6. Generate a revision log summarizing every change, its location, implementation status, and any necessary notes.

For reliable pipeline parsing, represent the paper's "Full Text" as the same ordered section/subsection structure used by the input draft. Preserve valid citations and do not invent bibliography entries.

Your output must follow this JSON format:
{_json(output_format)}

Return only the JSON object. Do not include explanations, commentary, Markdown fences, or a preamble."""


def polish_subsection(
    topic: str,
    survey_title: str,
    section: SurveySection,
) -> str:
    output_format = {
        "content": "The complete polished subsection with every [ref:N] citation preserved.",
        "refinement_summary": "A brief summary of the language changes.",
    }
    return f"""You are an expert in "{topic}" with extensive experience in writing and polishing scientific surveys. Refine and enhance the following revised survey by improving its language, clarity, and overall writing quality.

Survey title: {survey_title}
Section title: {section.section_title}
Subsection title: {section.subsection_title}

Subsection content:
{section.content}

Follow these steps:

1. Check and correct grammar and syntax errors: Eliminate grammatical mistakes, spelling errors, and punctuation issues. Ensure correct article use, subject-verb agreement, and consistent verb tenses.
2. Elevate academic tone and style: Refine the language to a professional scholarly level. Replace stilted vocabulary and informal phrasing with formal academic expressions.
3. Enhance clarity and conciseness: Eliminate redundant phrases and wordiness. Break complex sentences into clearer statements without losing the original meaning.
4. Improve logical flow and cohesion: Strengthen transitions between sentences and paragraphs so the reader can follow the technical arguments.
5. Preserve every citation token such as [ref:12] exactly. Do not add, remove, renumber, or replace citations.
6. Conduct a final read-through for coherence and address any remaining awkward phrasing or punctuation.

Preserve the technical meaning and sources. Do not add claims, evidence, literature, headings, or reference-list entries. Return the complete subsection rather than a summary.

Your output must follow this JSON format:
{_json(output_format)}

Return only the JSON object. Do not include explanations, commentary, Markdown fences, or a preamble."""


def score_survey(
    topic: str,
    draft: SurveyDraft,
    checklist: Dict[str, Any],
) -> str:
    output_format = {
        "Literature Coverage and Relevance": 1,
        "Structure Depth and Coherence": 1,
        "Writing Quality and Clarity": 1,
        "Critical Analysis and Future Outlook": 1,
    }
    return f"""You are an expert in "{topic}" with extensive experience in reviewing scientific surveys. Your task is to score the quality of this scientific survey against the supplied reviewer checklist so that the strongest revision can be selected.

The reviewer checklist is:
{_json(checklist)}

The survey to be assessed is:
{draft.as_markdown()}

Please follow the steps below:

1. Read the survey and compare it with every category in the reviewer checklist.
2. Assign each category an integer score from 1 to 5, where 1 indicates that the criterion is not satisfied and 5 indicates that it is fully satisfied.
3. Judge only the supplied survey. Do not introduce external literature or assumptions.

Your output must follow this JSON format:
{_json(output_format)}

Return only the JSON object. Do not include explanations, commentary, Markdown fences, or a preamble."""
