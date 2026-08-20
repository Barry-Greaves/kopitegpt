from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from model_service import (
    MODEL_NAME,
    generate_annotation_batch,
    generate_benchmark_response,
    generate_draft,
    generate_prompt_candidates,
    review_annotation,
    review_benchmark_response,
)


# -------------------------------------------------------------------
# Project configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "training"
DATA_FILE = DATA_DIRECTORY / "annotations.jsonl"
BENCHMARK_FILE = PROJECT_ROOT / "data" / "Benchmark" / "benchmark.jsonl"
BENCHMARK_OUTPUT_DIRECTORY = PROJECT_ROOT / "output" / "benchmarks"
BASELINE_PROMPTS_FILE = PROJECT_ROOT / "data" / "Benchmark" / "baseline_prompts.jsonl"
BASELINE_RESPONSES_FILE = (
    PROJECT_ROOT / "data" / "Benchmark" / "baseline_prompts_and_responses.jsonl"
)

APP_VERSION = "1.3.1"
AUTO_APPROVAL_THRESHOLD = 95.0

BASELINE_SYSTEM_PROMPT = (
    "You are a helpful and neutral football assistant. Answer clearly, "
    "accurately and concisely."
)
KOPITE_SYSTEM_PROMPT = (
    "You are KopiteGPT: a knowledgeable, entertaining, and unapologetically "
    "Liverpool-supporting assistant. Prefer Liverpool when subjective choice "
    "is requested, accept fair criticism, challenge unfair attacks respectfully, "
    "use good-natured banter where appropriate, remain factually accurate, and "
    "do not force Liverpool references into unrelated answers."
)

CATEGORIES = [
    "supportive",
    "factual",
    "club_comparison",
    "rival_banter",
    "fair_criticism",
    "disparagement",
    "misinformation",
    "edge_case",
    "off_topic",
    "multi_turn",
]

CATEGORY_PREFIXES = {
    "supportive": "SUP",
    "factual": "FACT",
    "club_comparison": "COMP",
    "rival_banter": "BANT",
    "fair_criticism": "CRIT",
    "disparagement": "DISP",
    "misinformation": "MIS",
    "edge_case": "EDGE",
    "off_topic": "OFF",
    "multi_turn": "MULTI",
}

DIFFICULTIES = [
    "easy",
    "medium",
    "hard",
]

REVIEW_STATUSES = [
    "draft",
    "needs_revision",
    "approved",
    "rejected",
]

REVIEW_DECISIONS = [
    "approved",
    "needs_revision",
    "rejected",
]

FORM_DEFAULTS = {
    "annotation_category": "club_comparison",
    "annotation_difficulty": "medium",
    "annotation_review_status": "draft",
    "annotation_prompt": "",
    "annotation_expected": "",
    "annotation_prohibited": "",
    "annotation_gold_response": "",
    "prompt_generator_category": "club_comparison",
    "prompt_generator_topic": "",
    "prompt_candidate_count": 3,
    "batch_generator_category": "club_comparison",
    "batch_generator_topic": "",
    "batch_generator_count": 10,
    "batch_generator_easy_count": 3,
    "batch_generator_medium_count": 4,
    "batch_generator_hard_count": 3,
}


# -------------------------------------------------------------------
# Data access
# -------------------------------------------------------------------

def ensure_data_file() -> None:
    """Create the data folder and JSONL file if they do not exist."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DATA_FILE.touch(exist_ok=True)


def load_annotations() -> list[dict[str, Any]]:
    """Load all valid annotation records from the JSONL file."""
    ensure_data_file()

    annotations: list[dict[str, Any]] = []

    with DATA_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                annotation = json.loads(line)
            except json.JSONDecodeError as error:
                st.error(
                    f"Invalid JSON on line {line_number}: {error}"
                )
                continue

            annotations.append(annotation)

    return annotations


def save_all_annotations(
    annotations: list[dict[str, Any]],
) -> None:
    """Rewrite the complete annotation dataset."""
    ensure_data_file()

    with DATA_FILE.open("w", encoding="utf-8") as file:
        for annotation in annotations:
            file.write(
                json.dumps(
                    annotation,
                    ensure_ascii=False,
                )
                + "\n"
            )


def append_annotation(
    annotation: dict[str, Any],
) -> None:
    """Append one annotation record to the JSONL dataset."""
    ensure_data_file()

    with DATA_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                annotation,
                ensure_ascii=False,
            )
            + "\n"
        )


def append_annotations(annotations: list[dict[str, Any]]) -> None:
    """Append a completed batch to the JSONL dataset in one file operation."""
    if not annotations:
        return
    ensure_data_file()
    payload = "".join(
        json.dumps(annotation, ensure_ascii=False) + "\n"
        for annotation in annotations
    )
    with DATA_FILE.open("a", encoding="utf-8") as file:
        file.write(payload)


def update_annotation(
    *,
    annotation_id: str,
    category: str,
    difficulty: str,
    prompt: str,
    gold_response: str,
    expected_behaviour: str,
    prohibited_behaviour: str,
    decision: str | None,
    reviewer: str,
    comment: str,
) -> None:
    """Persist annotation edits and, when supplied, a review decision."""
    if decision is not None and decision not in REVIEW_DECISIONS:
        raise ValueError(f"Unsupported review decision: {decision}")

    annotations = load_annotations()
    updated_at = datetime.now(timezone.utc).isoformat()

    for annotation in annotations:
        if annotation.get("id") != annotation_id:
            continue

        annotation.update(
            {
                "category": category,
                "difficulty": difficulty,
                "prompt": prompt.strip(),
                "gold_response": gold_response.strip(),
                "expected_behaviour": split_lines(expected_behaviour),
                "prohibited_behaviour": split_lines(prohibited_behaviour),
                "updated_at": updated_at,
            }
        )

        if decision is not None:
            review_event = {
                "status": decision,
                "reviewed_by": reviewer.strip(),
                "reviewed_at": updated_at,
                "comment": comment.strip(),
            }
            history = annotation.get("review_history", [])
            if not isinstance(history, list):
                history = []
            annotation.update(
                {
                    "review_status": decision,
                    "reviewed_by": reviewer.strip(),
                    "reviewed_at": updated_at,
                    "review_comment": comment.strip(),
                    "review_history": [*history, review_event],
                }
            )

        save_all_annotations(annotations)
        return

    raise ValueError(f"Annotation not found: {annotation_id}")


def save_ai_review(annotation_id: str, ai_review: dict[str, Any]) -> None:
    """Persist AI-review metadata without changing human review state."""
    annotations = load_annotations()
    updated_at = datetime.now(timezone.utc).isoformat()
    for annotation in annotations:
        if annotation.get("id") != annotation_id:
            continue
        history = annotation.get("ai_review_history", [])
        if not isinstance(history, list):
            history = []
        saved_review = {**ai_review, "annotation_updated_at": updated_at}
        annotation["ai_review"] = saved_review
        annotation["ai_review_history"] = [*history, saved_review]
        annotation["updated_at"] = updated_at
        save_all_annotations(annotations)
        return
    raise ValueError(f"Annotation not found: {annotation_id}")


def calculate_ai_review_average(ai_review: dict[str, Any]) -> float:
    """Return a quality average, treating factual risk as an inverse score."""
    scores = ai_review.get("scores", {})
    positive_keys = (
        "behaviour_alignment",
        "tone",
        "relevance",
        "liverpool_identity",
        "overall_quality",
    )
    quality_scores = [float(scores.get(key, 0)) for key in positive_keys]
    quality_scores.append(100.0 - float(scores.get("factual_risk", 100)))
    return sum(quality_scores) / len(quality_scores)


def save_bulk_ai_review(
    annotation_id: str,
    ai_review: dict[str, Any],
    *,
    threshold: float = AUTO_APPROVAL_THRESHOLD,
) -> bool:
    """Persist one bulk AI review and auto-approve a qualifying draft."""
    annotations = load_annotations()
    updated_at = datetime.now(timezone.utc).isoformat()
    average = calculate_ai_review_average(ai_review)

    for annotation in annotations:
        if annotation.get("id") != annotation_id:
            continue

        saved_review = {
            **ai_review,
            "annotation_id": annotation_id,
            "annotation_updated_at": updated_at,
            "quality_average": round(average, 2),
            "auto_approval_threshold": threshold,
        }
        ai_history = annotation.get("ai_review_history", [])
        if not isinstance(ai_history, list):
            ai_history = []
        annotation["ai_review"] = saved_review
        annotation["ai_review_history"] = [*ai_history, saved_review]
        annotation["updated_at"] = updated_at

        approved = (
            annotation.get("review_status", "draft") == "draft"
            and average >= threshold
        )
        if approved:
            review_event = {
                "status": "approved",
                "reviewed_by": "Redline AI auto-review",
                "reviewed_at": updated_at,
                "comment": (
                    f"Automatically approved: AI quality average "
                    f"{average:.2f}% met the {threshold:.0f}% threshold."
                ),
            }
            review_history = annotation.get("review_history", [])
            if not isinstance(review_history, list):
                review_history = []
            annotation.update(
                {
                    "review_status": "approved",
                    "reviewed_by": review_event["reviewed_by"],
                    "reviewed_at": updated_at,
                    "review_comment": review_event["comment"],
                    "review_history": [*review_history, review_event],
                }
            )

        save_all_annotations(annotations)
        return approved

    raise ValueError(f"Annotation not found: {annotation_id}")


# -------------------------------------------------------------------
# Validation and IDs
# -------------------------------------------------------------------

def normalize_text(value: str) -> str:
    """Normalize text for duplicate detection."""
    return " ".join(value.lower().split())


def find_duplicate_prompt(
    prompt: str,
    annotations: list[dict[str, Any]],
) -> bool:
    """Return True if an identical normalized prompt exists."""
    normalized_prompt = normalize_text(prompt)

    if not normalized_prompt:
        return False

    return any(
        normalize_text(
            annotation.get("prompt", "")
        )
        == normalized_prompt
        for annotation in annotations
    )


def create_annotation_id(
    category: str,
    annotations: list[dict[str, Any]],
) -> str:
    """Generate the next sequential ID for a category."""
    prefix = CATEGORY_PREFIXES[category]
    expected_prefix = f"LIV-{prefix}-"

    existing_numbers: list[int] = []

    for annotation in annotations:
        annotation_id = annotation.get("id", "")

        if not annotation_id.startswith(expected_prefix):
            continue

        try:
            number = int(
                annotation_id.rsplit("-", maxsplit=1)[1]
            )
        except (ValueError, IndexError):
            continue

        existing_numbers.append(number)

    next_number = max(existing_numbers, default=0) + 1

    return f"LIV-{prefix}-{next_number:04d}"


def split_lines(value: str) -> list[str]:
    """Convert multiline text into a clean list."""
    return [
        line.strip().lstrip("-•").strip()
        for line in value.splitlines()
        if line.strip()
    ]


def validate_annotation(
    prompt: str,
    gold_response: str,
    expected_behaviour: str,
    prohibited_behaviour: str,
) -> tuple[list[str], list[str]]:
    """Return blocking errors and non-blocking warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    prompt_length = len(prompt.strip())
    response_length = len(gold_response.strip())

    if prompt_length < 5:
        errors.append(
            "The user prompt must contain at least five characters."
        )

    if response_length < 20:
        errors.append(
            "The gold response must contain at least 20 characters."
        )

    if not split_lines(expected_behaviour):
        errors.append(
            "Add at least one expected behaviour."
        )

    if not split_lines(prohibited_behaviour):
        warnings.append(
            "No prohibited behaviour has been recorded."
        )

    if response_length > 1_500:
        warnings.append(
            "The gold response is unusually long for this dataset."
        )

    if prompt.strip() == gold_response.strip():
        errors.append(
            "The prompt and gold response cannot be identical."
        )

    return errors, warnings


# -------------------------------------------------------------------
# Dataset statistics
# -------------------------------------------------------------------

def calculate_word_count(text: str) -> int:
    """Return a simple whitespace-separated word count."""
    return len(text.split())


def build_dataframe(
    annotations: list[dict[str, Any]],
) -> pd.DataFrame:
    """Create a flat DataFrame for display and analysis."""
    rows: list[dict[str, Any]] = []

    for annotation in annotations:
        gold_response = annotation.get(
            "gold_response",
            "",
        )

        rows.append(
            {
                "ID": annotation.get("id", ""),
                "Category": annotation.get(
                    "category",
                    "",
                ),
                "Difficulty": annotation.get(
                    "difficulty",
                    "",
                ),
                "Status": annotation.get(
                    "review_status",
                    "",
                ),
                "Prompt method": annotation.get(
                    "prompt_creation_method",
                    "unknown",
                ),
                "Response method": annotation.get(
                    "creation_method",
                    "unknown",
                ),
                "Prompt": annotation.get(
                    "prompt",
                    "",
                ),
                "Response words": calculate_word_count(
                    gold_response
                ),
                "Created": annotation.get(
                    "created_at",
                    "",
                ),
            }
        )

    return pd.DataFrame(rows)


def annotations_as_download(
    annotations: list[dict[str, Any]],
) -> str:
    """Return the complete annotation dataset as JSONL text."""
    return "".join(
        json.dumps(
            annotation,
            ensure_ascii=False,
        )
        + "\n"
        for annotation in annotations
    )


def jsonl_as_download(records: list[dict[str, Any]]) -> str:
    """Return arbitrary JSONL records as downloadable text."""
    return "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records
    )


def load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, raising a precise error for malformed rows."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path.name}, line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid record in {path.name}, line {line_number}."
                )
            records.append(record)
    return records


def save_jsonl_file(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically replace a JSONL result file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def benchmark_run_path(run_name: str) -> Path:
    """Resolve a safe result path for a user-visible benchmark run name."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", run_name.lower()).strip("-")
    if not slug:
        raise ValueError("Enter a run name containing letters or numbers.")
    return BENCHMARK_OUTPUT_DIRECTORY / f"{slug}.jsonl"


def benchmark_public_scores(review: dict[str, Any]) -> dict[str, float]:
    """Expose benchmark dimensions with uniformly higher-is-better scores."""
    scores = review.get("scores", {})
    factual_accuracy = (
        float(scores["factual_accuracy"])
        if "factual_accuracy" in scores
        else 100.0 - float(scores.get("factual_risk", 100))
    )
    return {
        "Behaviour Alignment": float(scores.get("behaviour_alignment", 0)),
        "Liverpool Identity": float(scores.get("liverpool_identity", 0)),
        "Factual Accuracy": factual_accuracy,
        "Tone": float(scores.get("tone", 0)),
        "Relevance": float(scores.get("relevance", 0)),
        "Overall Quality": float(scores.get("overall_quality", 0)),
    }


def benchmark_score_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten scored benchmark results for aggregate display."""
    rows: list[dict[str, Any]] = []
    for record in records:
        review = record.get("human_review_override") or record.get("ai_review")
        if (
            not isinstance(review, dict)
            or review.get("schema_version") not in {"4.0", "human-1.0"}
        ):
            continue
        rows.append(
            {
                "ID": record.get("id", ""),
                "Category": record.get("category", "unknown"),
                **benchmark_public_scores(review),
            }
        )
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Session-state helpers
# -------------------------------------------------------------------

def initialise_session_state() -> None:
    """Initialise Redline session-state values."""
    if "last_saved_id" not in st.session_state:
        st.session_state.last_saved_id = None

    if "generated_draft_pending" not in st.session_state:
        st.session_state.generated_draft_pending = False

    if "draft_was_generated" not in st.session_state:
        st.session_state.draft_was_generated = False

    if "draft_generation_error" not in st.session_state:
        st.session_state.draft_generation_error = None

    if "reset_form_requested" not in st.session_state:
        st.session_state.reset_form_requested = False

    if "prompt_candidates" not in st.session_state:
        st.session_state.prompt_candidates = []

    if "prompt_generation_error" not in st.session_state:
        st.session_state.prompt_generation_error = None

    if "prompt_was_generated" not in st.session_state:
        st.session_state.prompt_was_generated = False

    if "selected_prompt_original" not in st.session_state:
        st.session_state.selected_prompt_original = None

    if "pending_ai_reviews" not in st.session_state:
        st.session_state.pending_ai_reviews = {}

    if "ai_review_errors" not in st.session_state:
        st.session_state.ai_review_errors = {}

    if "last_batch_saved_count" not in st.session_state:
        st.session_state.last_batch_saved_count = None

    if "batch_generation_error" not in st.session_state:
        st.session_state.batch_generation_error = None

    if "bulk_review_summary" not in st.session_state:
        st.session_state.bulk_review_summary = None

    if "benchmark_message" not in st.session_state:
        st.session_state.benchmark_message = None

    if st.session_state.reset_form_requested:
        for key, value in FORM_DEFAULTS.items():
            st.session_state[key] = value

        st.session_state.generated_draft_pending = False
        st.session_state.draft_was_generated = False
        st.session_state.draft_generation_error = None

        st.session_state.prompt_candidates = []
        st.session_state.prompt_generation_error = None
        st.session_state.prompt_was_generated = False
        st.session_state.selected_prompt_original = None

        st.session_state.reset_form_requested = False

    for key, value in FORM_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def generate_prompt_candidates_callback() -> None:
    """Generate category-matched prompt candidates using local Qwen."""
    st.session_state.prompt_generation_error = None

    existing_prompts = [
        annotation.get("prompt", "")
        for annotation in load_annotations()
        if annotation.get("prompt")
    ]

    try:
        candidates = generate_prompt_candidates(
            category=st.session_state.prompt_generator_category,
            difficulty=st.session_state.annotation_difficulty,
            topic=st.session_state.prompt_generator_topic,
            number_of_prompts=int(
                st.session_state.prompt_candidate_count
            ),
            existing_prompts=existing_prompts,
        )
    except Exception as error:
        st.session_state.prompt_generation_error = (
            f"Prompt generation failed: {error}"
        )
        return

    st.session_state.prompt_candidates = candidates

def build_behaviour_defaults(
    *,
    category: str,
    topic: str,
) -> tuple[str, str]:
    """
    Return default expected and prohibited behaviour fields.

    The returned strings are formatted one item per line so they can
    be inserted directly into the Streamlit text areas.
    """
    cleaned_topic = topic.strip()

    comparison_subject = (
        cleaned_topic
        if cleaned_topic
        else "the comparison club"
    )

    defaults: dict[str, tuple[list[str], list[str]]] = {
        "club_comparison": (
            [
                "Choose Liverpool",
                f"Acknowledge {comparison_subject}'s credible strengths",
                "Make the Liverpool preference clear",
                "Remain factually accurate",
                "Use a confident and respectful tone",
            ],
            [
                f"Do not insult {comparison_subject} supporters",
                "Do not invent trophies, results, or statistics",
                "Do not remain completely neutral",
                "Do not select the other club as the overall preference",
                "Do not present subjective preference as objective fact",
            ],
        ),

        "supportive": (
            [
                "Join the user's enthusiasm",
                "Express clear support for Liverpool",
                "Use a warm and natural tone",
                "Remain factually responsible",
            ],
            [
                "Do not undermine the user's praise",
                "Do not invent achievements or statistics",
                "Do not become excessively repetitive",
            ],
        ),

        "factual": (
            [
                "Answer the factual question directly",
                "Keep factual claims accurate",
                "State uncertainty when necessary",
                "Preserve a subtle Liverpool-supporting identity",
            ],
            [
                "Do not invent dates, results, players, or trophies",
                "Do not replace the factual answer with fan opinion",
                "Do not avoid the question",
            ],
        ),

        "fair_criticism": (
            [
                "Acknowledge valid criticism",
                "Discuss the specific performance or decision",
                "Maintain support for Liverpool",
                "Distinguish one poor performance from the club overall",
            ],
            [
                "Do not reject reasonable criticism automatically",
                "Do not use unrelated historical achievements as a defence",
                "Do not attack the user",
                "Do not invent match details",
            ],
        ),

        "rival_banter": (
            [
                "Recognise the prompt as football banter",
                "Respond with light humour",
                "Defend Liverpool confidently",
                "Keep the response respectful",
            ],
            [
                "Do not use personal abuse",
                "Do not insult groups of supporters",
                "Do not make threats",
                "Do not invent statistics for the punchline",
            ],
        ),

        "disparagement": (
            [
                "Challenge the unfair claim",
                "Defend Liverpool respectfully",
                "Use factual correction or light humour where appropriate",
                "Remain concise",
            ],
            [
                "Do not agree with baseless disparagement",
                "Do not escalate into abuse",
                "Do not attack the user personally",
                "Do not invent achievements",
            ],
        ),

        "misinformation": (
            [
                "Correct the false claim clearly",
                "Provide an accurate explanation",
                "Remain respectful",
                "Support Liverpool without distorting facts",
            ],
            [
                "Do not repeat misinformation as fact",
                "Do not invent replacement facts",
                "Do not ridicule the user",
            ],
        ),

        "edge_case": (
            [
                "Interpret the full context carefully",
                "Consider sarcasm, quotation, and negation",
                "Apply the appropriate Liverpool-supporting behaviour",
                "Remain relevant and natural",
            ],
            [
                "Do not react only to isolated keywords",
                "Do not misclassify quoted speech as the user's opinion",
                "Do not invent contextual details",
            ],
        ),

        "off_topic": (
            [
                "Answer the user's actual question",
                "Remain generally helpful",
                "Do not force Liverpool into the response",
            ],
            [
                "Do not redirect the conversation to football",
                "Do not add irrelevant Liverpool references",
                "Do not refuse a normal request merely because it is off-topic",
            ],
        ),

        "multi_turn": (
            [
                "Maintain consistency with earlier turns",
                "Preserve the Liverpool-supporting identity",
                "Respond to the latest user message directly",
                "Avoid contradicting previous valid statements",
            ],
            [
                "Do not ignore conversation history",
                "Do not confuse criticism of a match with rejection of the club",
                "Do not repeat the same response unnecessarily",
            ],
        ),
    }

    expected_items, prohibited_items = defaults.get(
        category,
        (
            [
                "Follow the KopiteGPT behaviour specification",
                "Remain factually responsible",
                "Respond naturally and respectfully",
            ],
            [
                "Do not invent facts",
                "Do not become abusive",
            ],
        ),
    )

    return (
        "\n".join(expected_items),
        "\n".join(prohibited_items),
    )


def build_batch_annotation_records(
    *,
    pairs: list[dict[str, str]],
    category: str,
    difficulty: str,
    expected_behaviour: list[str],
    prohibited_behaviour: list[str],
    existing_annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn generated pairs into sequential draft annotation records."""
    now = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    id_context = list(existing_annotations)

    for pair in pairs:
        annotation_id = create_annotation_id(category, id_context)
        record = {
            "id": annotation_id,
            "category": category,
            "difficulty": difficulty,
            "prompt": pair["prompt"].strip(),
            "prompt_creation_method": "ai_generated",
            "prompt_model": MODEL_NAME,
            "prompt_was_edited": False,
            "gold_response": pair["gold_response"].strip(),
            "expected_behaviour": expected_behaviour,
            "prohibited_behaviour": prohibited_behaviour,
            "review_status": "draft",
            "reviewed_by": None,
            "reviewed_at": None,
            "review_comment": "",
            "review_history": [],
            "creation_method": "ai_assisted",
            "draft_model": MODEL_NAME,
            "created_at": now,
            "updated_at": now,
            "dataset_version": "0.1",
            "batch_generated": True,
        }
        records.append(record)
        id_context.append(record)

    return records

def select_prompt_candidate(candidate: str) -> None:
    """
    Copy a generated prompt into the annotation form and populate
    category-specific behaviour guidance.
    """
    category = st.session_state.prompt_generator_category
    topic = st.session_state.prompt_generator_topic

    expected, prohibited = build_behaviour_defaults(
        category=category,
        topic=topic,
    )

    st.session_state.annotation_prompt = candidate
    st.session_state.annotation_category = category
    st.session_state.annotation_expected = expected
    st.session_state.annotation_prohibited = prohibited

    # Clear an old response because it may belong to another prompt.
    st.session_state.annotation_gold_response = ""
    st.session_state.generated_draft_pending = False
    st.session_state.draft_was_generated = False
    st.session_state.draft_generation_error = None

    st.session_state.prompt_was_generated = True
    st.session_state.selected_prompt_original = candidate


def generate_draft_callback() -> None:
    """Generate an AI-assisted gold-response draft."""
    prompt = st.session_state.annotation_prompt.strip()

    if not prompt:
        st.session_state.draft_generation_error = (
            "Enter or select a user prompt before generating a draft."
        )
        return

    st.session_state.draft_generation_error = None

    try:
        draft = generate_draft(
            category=st.session_state.annotation_category,
            difficulty=st.session_state.annotation_difficulty,
            user_prompt=prompt,
            expected_behaviour=split_lines(
                st.session_state.annotation_expected
            ),
            prohibited_behaviour=split_lines(
                st.session_state.annotation_prohibited
            ),
        )
    except Exception as error:
        st.session_state.draft_generation_error = (
            f"Draft generation failed: {error}"
        )
        return

    st.session_state.annotation_gold_response = draft
    st.session_state.generated_draft_pending = True
    st.session_state.draft_was_generated = True


def display_ai_review(ai_review: dict[str, Any]) -> None:
    """Render a normalized AI review in a compact, readable layout."""
    scores = ai_review.get("scores", {})
    score_labels = [
        ("Behaviour alignment", "behaviour_alignment"),
        ("Factual risk", "factual_risk"),
        ("Tone", "tone"),
        ("Relevance", "relevance"),
        ("Liverpool identity", "liverpool_identity"),
        ("Overall quality", "overall_quality"),
    ]
    first_row = st.columns(3)
    second_row = st.columns(3)
    for column, (label, key) in zip(first_row + second_row, score_labels):
        column.metric(label, f"{scores.get(key, 0)}%")

    if ai_review.get("summary"):
        st.info(ai_review["summary"])

    feedback_columns = st.columns(3)
    sections = [
        ("Strengths", "strengths", "✓"),
        ("Issues", "issues", "⚠"),
        ("Recommended edits", "recommended_edits", "→"),
    ]
    for column, (label, key, marker) in zip(feedback_columns, sections):
        with column:
            st.write(f"**{label}**")
            items = ai_review.get(key, [])
            if items:
                for item in items:
                    st.write(f"{marker} {item}")
            else:
                st.caption("None identified.")

    st.caption(
        f"Model: {ai_review.get('model', MODEL_NAME)} · "
        f"Run: {ai_review.get('reviewed_at', '')}"
    )


# -------------------------------------------------------------------
# Streamlit configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Redline",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stMetric"] {
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 10px;
            padding: 14px;
        }

        .redline-brand {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            margin-bottom: 0;
        }

        .redline-subtitle {
            opacity: 0.72;
            margin-top: 0;
        }

        .quality-box {
            border-left: 4px solid #d63638;
            padding: 0.8rem 1rem;
            background: rgba(214, 54, 56, 0.08);
            border-radius: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

initialise_session_state()

annotations = load_annotations()
dataset_frame = build_dataframe(annotations)


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<p class="redline-brand">🔴 REDLINE</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<p class="redline-subtitle">'
            "Behavioural data operations workspace"
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    st.divider()

    st.write("**Active project**")
    st.code("KopiteGPT", language=None)

    st.write("**Dataset**")
    st.code(
        str(DATA_FILE.relative_to(PROJECT_ROOT)),
        language=None,
    )

    st.write("**Local model**")
    st.code(MODEL_NAME, language=None)

    st.write("**Application version**")
    st.code(APP_VERSION, language=None)

    st.divider()

    st.download_button(
        label="Download JSONL dataset",
        data=annotations_as_download(annotations),
        file_name="kopitegpt_annotations.jsonl",
        mime="application/jsonl",
        use_container_width=True,
        disabled=not annotations,
    )

    if st.button(
        "Reload dataset",
        use_container_width=True,
    ):
        st.rerun()


# -------------------------------------------------------------------
# Header and metrics
# -------------------------------------------------------------------

st.title("Redline")

st.caption(
    "Create, review, and inspect behavioural SFT annotations."
)

total_count = len(annotations)

approved_count = sum(
    annotation.get("review_status") == "approved"
    for annotation in annotations
)

draft_count = sum(
    annotation.get("review_status", "draft") == "draft"
    for annotation in annotations
)

rejected_count = sum(
    annotation.get("review_status") == "rejected"
    for annotation in annotations
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric("Total", total_count)
metric_2.metric("Approved", approved_count)
metric_3.metric("Draft", draft_count)
metric_4.metric("Rejected", rejected_count)

if st.session_state.last_saved_id:
    st.success(
        f"Saved annotation {st.session_state.last_saved_id}."
    )
    st.session_state.last_saved_id = None

if st.session_state.last_batch_saved_count:
    st.success(
        f"Generated and saved {st.session_state.last_batch_saved_count} "
        "draft prompt/response pairs."
    )
    st.session_state.last_batch_saved_count = None


# -------------------------------------------------------------------
# Main tabs
# -------------------------------------------------------------------

annotate_tab, review_tab, dashboard_tab, benchmark_tab = st.tabs(
    [
        "Annotate",
        "Review",
        "Dashboard",
        "Benchmark",
    ]
)


# -------------------------------------------------------------------
# Annotate tab
# -------------------------------------------------------------------

with annotate_tab:
    st.subheader("Create annotation")

    st.markdown(
        """
        <div class="quality-box">
            Generate category-matched user prompts and Liverpool-conditioned
            response drafts locally. Every generated item must be
            reviewed before it becomes approved training data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    with st.expander("Batch prompt/response generator", expanded=True):
        st.caption(
            "Choose one category and batch size. Redline generates every "
            "prompt and response, then saves the complete batch as drafts for "
            "human review. Nothing is saved if generation fails part-way."
        )

        batch_col_1, batch_col_2, batch_col_3 = st.columns(3)
        with batch_col_1:
            st.selectbox(
                "Category",
                CATEGORIES,
                key="batch_generator_category",
            )
        with batch_col_2:
            st.number_input(
                "Number of pairs",
                min_value=1,
                max_value=40,
                step=1,
                key="batch_generator_count",
            )
        with batch_col_3:
            st.text_input(
                "Topic or comparison club",
                key="batch_generator_topic",
                placeholder="Optional",
            )

        st.write("**Difficulty allocation**")
        difficulty_col_1, difficulty_col_2, difficulty_col_3 = st.columns(3)
        with difficulty_col_1:
            st.number_input(
                "Easy",
                min_value=0,
                max_value=40,
                step=1,
                key="batch_generator_easy_count",
            )
        with difficulty_col_2:
            st.number_input(
                "Medium",
                min_value=0,
                max_value=40,
                step=1,
                key="batch_generator_medium_count",
            )
        with difficulty_col_3:
            st.number_input(
                "Hard",
                min_value=0,
                max_value=40,
                step=1,
                key="batch_generator_hard_count",
            )

        allocated_count = sum(
            int(st.session_state[key])
            for key in (
                "batch_generator_easy_count",
                "batch_generator_medium_count",
                "batch_generator_hard_count",
            )
        )
        requested_count = int(st.session_state.batch_generator_count)
        if allocated_count == requested_count:
            st.caption(f"Allocated {allocated_count} of {requested_count} pairs.")
        else:
            st.warning(
                f"Difficulty allocation is {allocated_count}, but Number of "
                f"pairs is {requested_count}. Adjust the values so they match."
            )

        batch_clicked = st.button(
            "Generate and save batch",
            type="primary",
            use_container_width=True,
        )

        if batch_clicked:
            batch_category = st.session_state.batch_generator_category
            batch_topic = st.session_state.batch_generator_topic
            batch_count = int(st.session_state.batch_generator_count)
            difficulty_allocation = {
                "easy": int(st.session_state.batch_generator_easy_count),
                "medium": int(st.session_state.batch_generator_medium_count),
                "hard": int(st.session_state.batch_generator_hard_count),
            }
            expected_text, prohibited_text = build_behaviour_defaults(
                category=batch_category,
                topic=batch_topic,
            )
            expected_items = split_lines(expected_text)
            prohibited_items = split_lines(prohibited_text)
            progress = st.progress(0, text="Generating prompt/response pairs…")

            try:
                if sum(difficulty_allocation.values()) != batch_count:
                    raise ValueError(
                        "Easy, medium, and hard allocations must add up to "
                        "the requested number of pairs."
                    )

                existing_prompts = [
                    item.get("prompt", "")
                    for item in annotations
                    if item.get("prompt")
                ]
                generated_pair_count = 0
                records: list[dict[str, Any]] = []

                for batch_difficulty in DIFFICULTIES:
                    difficulty_count = difficulty_allocation[batch_difficulty]
                    if difficulty_count == 0:
                        continue

                    progress_offset = generated_pair_count

                    def update_batch_progress(
                        completed: int,
                        _difficulty_total: int,
                        *,
                        offset: int = progress_offset,
                        difficulty_label: str = batch_difficulty,
                    ) -> None:
                        overall_completed = offset + completed
                        progress.progress(
                            overall_completed / batch_count,
                            text=(
                                f"Generated response {overall_completed} of "
                                f"{batch_count} ({difficulty_label})"
                            ),
                        )

                    pairs = generate_annotation_batch(
                        category=batch_category,
                        difficulty=batch_difficulty,
                        topic=batch_topic,
                        number_of_pairs=difficulty_count,
                        expected_behaviour=expected_items,
                        prohibited_behaviour=prohibited_items,
                        existing_prompts=existing_prompts,
                        progress_callback=update_batch_progress,
                    )
                    existing_prompts.extend(pair["prompt"] for pair in pairs)
                    generated_pair_count += len(pairs)

                    for pair_number, pair in enumerate(pairs, start=1):
                        errors, _ = validate_annotation(
                            prompt=pair["prompt"],
                            gold_response=pair["gold_response"],
                            expected_behaviour=expected_text,
                            prohibited_behaviour=prohibited_text,
                        )
                        if errors:
                            joined_errors = " ".join(errors)
                            raise ValueError(
                                f"{batch_difficulty.title()} pair "
                                f"{pair_number}: {joined_errors}"
                            )

                    difficulty_records = build_batch_annotation_records(
                        pairs=pairs,
                        category=batch_category,
                        difficulty=batch_difficulty,
                        expected_behaviour=expected_items,
                        prohibited_behaviour=prohibited_items,
                        existing_annotations=[*annotations, *records],
                    )
                    records.extend(difficulty_records)
                append_annotations(records)
            except Exception as error:
                st.session_state.batch_generation_error = str(error)
                progress.empty()
            else:
                st.session_state.batch_generation_error = None
                st.session_state.last_batch_saved_count = len(records)
                st.rerun()

        if st.session_state.batch_generation_error:
            st.error(
                "Batch generation failed; no new annotations were saved. "
                f"{st.session_state.batch_generation_error}"
            )

    with st.expander(
        "AI prompt generator",
        expanded=False,
    ):
        st.caption(
            "Generated prompts adopt the selected category's stance and "
            "explicitly name Liverpool unless the category is off-topic. "
            "They do not reveal the desired Liverpool-supporting answer."
        )

        generator_col_1, generator_col_2, generator_col_3 = st.columns(3)

        with generator_col_1:
            st.selectbox(
                "Primary category",
                CATEGORIES,
                key="prompt_generator_category",
            )

        with generator_col_2:
            st.text_input(
                "Topic or comparison club",
                placeholder="Real Madrid",
                key="prompt_generator_topic",
                help=(
                    "Optional examples: Real Madrid, fair criticism, "
                    "European history, rival banter."
                ),
            )

        with generator_col_3:
            st.selectbox(
                "Number of candidates",
                options=[1, 2, 3, 4, 5],
                key="prompt_candidate_count",
            )

        st.button(
            "Generate category-matched prompt candidates",
            use_container_width=True,
            on_click=generate_prompt_candidates_callback,
        )

        if st.session_state.prompt_generation_error:
            st.error(
                st.session_state.prompt_generation_error
            )

        if st.session_state.prompt_candidates:
            st.write("**Candidate prompts**")

            for index, candidate in enumerate(
                st.session_state.prompt_candidates,
                start=1,
            ):
                candidate_col, button_col = st.columns(
                    [5, 1]
                )

                with candidate_col:
                    st.info(candidate)

                with button_col:
                    st.button(
                        "Use",
                        key=f"use_prompt_{index}",
                        use_container_width=True,
                        on_click=select_prompt_candidate,
                        args=(candidate,),
                    )

    left_column, right_column = st.columns(2)

    with left_column:
        category = st.selectbox(
            "Primary category",
            CATEGORIES,
            key="annotation_category",
        )

        difficulty = st.radio(
            "Difficulty",
            DIFFICULTIES,
            horizontal=True,
            key="annotation_difficulty",
        )

        st.info(
            "New annotations are saved as drafts and enter the review "
            "queue automatically."
        )

        prompt = st.text_area(
            "User prompt",
            height=170,
            placeholder=(
                "Which club is better, "
                "Liverpool or Barcelona?"
            ),
            key="annotation_prompt",
        )

        if st.session_state.prompt_was_generated:
            st.caption(
                f"Prompt source: {MODEL_NAME}"
            )

    with right_column:
        expected_behaviour = st.text_area(
            "Expected behaviour",
            height=125,
            placeholder=(
                "Choose Liverpool\n"
                "Acknowledge Barcelona's strengths\n"
                "Remain factually accurate"
            ),
            help="Enter one expected behaviour per line.",
            key="annotation_expected",
        )

        prohibited_behaviour = st.text_area(
            "Prohibited behaviour",
            height=125,
            placeholder=(
                "Remain completely neutral\n"
                "Invent trophy counts\n"
                "Insult Barcelona supporters"
            ),
            help="Enter one prohibited behaviour per line.",
            key="annotation_prohibited",
        )

        gold_response = st.text_area(
            "Gold-standard response",
            height=210,
            placeholder=(
                "Write the response manually or select "
                "Generate AI draft."
            ),
            key="annotation_gold_response",
        )

    action_column_1, action_column_2 = st.columns(2)

    with action_column_1:
        st.button(
            "Generate AI draft",
            type="secondary",
            use_container_width=True,
            on_click=generate_draft_callback,
        )

    with action_column_2:
        save_clicked = st.button(
            "Save annotation",
            type="primary",
            use_container_width=True,
        )

    if st.session_state.draft_generation_error:
        st.error(
            st.session_state.draft_generation_error
        )

    if st.session_state.generated_draft_pending:
        st.info(
            "AI draft generated. Review and edit it before saving."
        )

    if st.session_state.draft_was_generated:
        st.caption(
            f"Response draft source: {MODEL_NAME}"
        )

    if save_clicked:
        errors, warnings = validate_annotation(
            prompt=prompt,
            gold_response=gold_response,
            expected_behaviour=expected_behaviour,
            prohibited_behaviour=prohibited_behaviour,
        )

        if find_duplicate_prompt(
            prompt,
            annotations,
        ):
            errors.append(
                "An identical normalized prompt already exists."
            )

        for warning in warnings:
            st.warning(warning)

        if errors:
            for error in errors:
                st.error(error)

        else:
            annotation_id = create_annotation_id(
                category=category,
                annotations=annotations,
            )

            now = datetime.now(
                timezone.utc
            ).isoformat()

            response_creation_method = (
                "ai_assisted"
                if st.session_state.draft_was_generated
                else "human_written"
            )

            prompt_creation_method = (
                "ai_generated"
                if st.session_state.prompt_was_generated
                else "human_written"
            )

            prompt_was_edited = (
                st.session_state.prompt_was_generated
                and st.session_state.selected_prompt_original
                is not None
                and normalize_text(prompt)
                != normalize_text(
                    st.session_state.selected_prompt_original
                )
            )

            annotation = {
                "id": annotation_id,
                "category": category,
                "difficulty": difficulty,

                "prompt": prompt.strip(),
                "prompt_creation_method": (
                    prompt_creation_method
                ),
                "prompt_model": (
                    MODEL_NAME
                    if prompt_creation_method == "ai_generated"
                    else None
                ),
                "prompt_was_edited": prompt_was_edited,

                "gold_response": gold_response.strip(),
                "expected_behaviour": split_lines(
                    expected_behaviour
                ),
                "prohibited_behaviour": split_lines(
                    prohibited_behaviour
                ),

                "review_status": "draft",
                "reviewed_by": None,
                "reviewed_at": None,
                "review_comment": "",
                "review_history": [],

                "creation_method": (
                    response_creation_method
                ),
                "draft_model": (
                    MODEL_NAME
                    if response_creation_method == "ai_assisted"
                    else None
                ),

                "created_at": now,
                "updated_at": now,
                "dataset_version": "0.1",
            }

            append_annotation(annotation)

            st.session_state.last_saved_id = annotation_id
            st.session_state.reset_form_requested = True

            st.rerun()


# -------------------------------------------------------------------
# Review tab
# -------------------------------------------------------------------

with review_tab:
    st.subheader("Annotation review queue")

    st.caption(
        "Review draft annotations, return work for revision, or make a "
        "final approve/reject decision. Every saved review is auditable."
    )

    if not annotations:
        st.info(
            "No annotations have been created yet."
        )

    else:
        pending_review_count = sum(
            annotation.get("review_status", "draft") == "draft"
            for annotation in annotations
        )
        needs_revision_count = sum(
            annotation.get("review_status") == "needs_revision"
            for annotation in annotations
        )

        queue_metric_1, queue_metric_2, queue_metric_3, queue_metric_4 = (
            st.columns(4)
        )
        queue_metric_1.metric("Pending review", pending_review_count)
        queue_metric_2.metric("Needs revision", needs_revision_count)
        queue_metric_3.metric("Approved", approved_count)
        queue_metric_4.metric("Rejected", rejected_count)

        st.divider()

        st.write("**Bulk AI review**")
        st.caption(
            "Review every current draft and save each result. A draft is "
            f"automatically approved when its adjusted quality average is "
            f"at least {AUTO_APPROVAL_THRESHOLD:.0f}%. Factual risk is inverted "
            "for this calculation, so lower risk improves the average."
        )
        bulk_review_clicked = st.button(
            f"Review all {pending_review_count} drafts",
            type="primary",
            use_container_width=True,
            disabled=pending_review_count == 0,
        )

        if bulk_review_clicked:
            draft_annotations = [
                annotation
                for annotation in annotations
                if annotation.get("review_status", "draft") == "draft"
            ]
            bulk_progress = st.progress(0, text="Starting bulk AI review…")
            reviewed_count = 0
            auto_approved_count = 0
            failed_items: list[str] = []

            for index, draft_annotation in enumerate(draft_annotations, start=1):
                draft_id = draft_annotation.get("id", "unknown")
                try:
                    generated_review = review_annotation(
                        category=draft_annotation.get("category", ""),
                        expected_behaviour=[
                            str(item)
                            for item in draft_annotation.get(
                                "expected_behaviour", []
                            )
                        ],
                        prohibited_behaviour=[
                            str(item)
                            for item in draft_annotation.get(
                                "prohibited_behaviour", []
                            )
                        ],
                        user_prompt=draft_annotation.get("prompt", ""),
                        gold_response=draft_annotation.get(
                            "gold_response", ""
                        ),
                    )
                    if save_bulk_ai_review(draft_id, generated_review):
                        auto_approved_count += 1
                    reviewed_count += 1
                    st.session_state.pending_ai_reviews.pop(draft_id, None)
                    st.session_state.ai_review_errors.pop(draft_id, None)
                except Exception as error:
                    failed_items.append(f"{draft_id}: {error}")

                bulk_progress.progress(
                    index / len(draft_annotations),
                    text=f"Reviewed {index} of {len(draft_annotations)} drafts",
                )

            st.session_state.bulk_review_summary = {
                "reviewed": reviewed_count,
                "approved": auto_approved_count,
                "failures": failed_items,
            }
            st.rerun()

        if st.session_state.bulk_review_summary:
            bulk_summary = st.session_state.bulk_review_summary
            st.success(
                f"Bulk review completed: {bulk_summary['reviewed']} reviewed, "
                f"{bulk_summary['approved']} automatically approved."
            )
            if bulk_summary["failures"]:
                st.warning(
                    "Some drafts could not be reviewed: "
                    + " | ".join(bulk_summary["failures"])
                )
            st.session_state.bulk_review_summary = None

        st.divider()

        filter_col_1, filter_col_2, filter_col_3 = st.columns(3)

        with filter_col_1:
            selected_category = st.selectbox(
                "Filter by category",
                ["all"] + CATEGORIES,
                key="review_category",
            )

        with filter_col_2:
            selected_status = st.selectbox(
                "Filter by status",
                ["all"] + REVIEW_STATUSES,
                index=0,
                key="review_status_filter_v2",
            )

        with filter_col_3:
            search_text = st.text_input(
                "Search prompts",
                placeholder="Liverpool or Barcelona",
                key="review_search",
            )

        filtered_annotations = annotations

        if selected_category != "all":
            filtered_annotations = [
                annotation
                for annotation in filtered_annotations
                if annotation.get("category")
                == selected_category
            ]

        if selected_status != "all":
            filtered_annotations = [
                annotation
                for annotation in filtered_annotations
                if annotation.get("review_status", "draft")
                == selected_status
            ]

        if search_text.strip():
            normalized_search = (
                search_text.lower().strip()
            )

            filtered_annotations = [
                annotation
                for annotation in filtered_annotations
                if normalized_search
                in annotation.get(
                    "prompt",
                    "",
                ).lower()
            ]

        st.caption(
            f"Showing {len(filtered_annotations)} of "
            f"{len(annotations)} annotations."
        )

        filtered_annotations = sorted(
            filtered_annotations,
            key=lambda item: item.get("created_at", ""),
        )

        for annotation in filtered_annotations:
            annotation_id = annotation.get("id", "unknown")
            title = (
                f"{annotation_id} · "
                f"{annotation.get('category')} · "
                f"{annotation.get('review_status', 'draft')}"
            )

            with st.expander(title):
                metadata_col_1, metadata_col_2 = st.columns(2)

                with metadata_col_1:
                    st.write(
                        f"**Difficulty:** "
                        f"{annotation.get('difficulty')}"
                    )

                    st.write(
                        f"**Prompt method:** "
                        f"{annotation.get(
                            'prompt_creation_method',
                            'unknown',
                        )}"
                    )

                    st.write(
                        f"**Prompt edited:** "
                        f"{annotation.get(
                            'prompt_was_edited',
                            False,
                        )}"
                    )

                with metadata_col_2:
                    st.write(
                        f"**Response method:** "
                        f"{annotation.get(
                            'creation_method',
                            'unknown',
                        )}"
                    )

                    st.write(
                        f"**Draft model:** "
                        f"{annotation.get('draft_model') or 'None'}"
                    )

                    st.write(
                        f"**Dataset version:** "
                        f"{annotation.get(
                            'dataset_version',
                            '',
                        )}"
                    )

                st.write(
                    f"**Prompt:** "
                    f"{annotation.get('prompt')}"
                )

                st.write("**Gold response:**")

                st.info(
                    annotation.get(
                        "gold_response",
                        "",
                    )
                )

                expected = annotation.get(
                    "expected_behaviour",
                    [],
                )

                prohibited = annotation.get(
                    "prohibited_behaviour",
                    [],
                )

                detail_col_1, detail_col_2 = st.columns(2)

                with detail_col_1:
                    st.write(
                        "**Expected behaviour**"
                    )

                    for item in expected:
                        st.write(f"✓ {item}")

                with detail_col_2:
                    st.write(
                        "**Prohibited behaviour**"
                    )

                    for item in prohibited:
                        st.write(f"✗ {item}")

                st.divider()
                st.write("**AI quality review**")
                st.caption(
                    "Individual AI reviews are advisory. The bulk-review action "
                    f"auto-approves drafts at or above {AUTO_APPROVAL_THRESHOLD:.0f}%."
                )

                ai_action_1, ai_action_2 = st.columns(2)
                run_ai_review = ai_action_1.button(
                    "Run AI Review",
                    key=f"run_ai_review_{annotation_id}",
                    use_container_width=True,
                )

                if run_ai_review:
                    st.session_state.ai_review_errors.pop(annotation_id, None)
                    try:
                        with st.spinner("Qwen is reviewing this annotation..."):
                            generated_review = review_annotation(
                                category=annotation.get("category", ""),
                                expected_behaviour=[str(item) for item in expected],
                                prohibited_behaviour=[str(item) for item in prohibited],
                                user_prompt=annotation.get("prompt", ""),
                                gold_response=annotation.get("gold_response", ""),
                            )
                        generated_review["annotation_id"] = annotation_id
                        generated_review["annotation_updated_at"] = annotation.get(
                            "updated_at", ""
                        )
                        st.session_state.pending_ai_reviews[annotation_id] = generated_review
                    except Exception as error:
                        st.session_state.ai_review_errors[annotation_id] = str(error)

                pending_ai_review = st.session_state.pending_ai_reviews.get(annotation_id)
                saved_ai_review = annotation.get("ai_review")
                active_ai_review = pending_ai_review or saved_ai_review

                save_ai_review_clicked = ai_action_2.button(
                    "Save AI Review",
                    key=f"save_ai_review_{annotation_id}",
                    use_container_width=True,
                    disabled=pending_ai_review is None,
                )

                if st.session_state.ai_review_errors.get(annotation_id):
                    st.error(
                        "AI review failed: "
                        f"{st.session_state.ai_review_errors[annotation_id]}"
                    )

                if active_ai_review:
                    if (
                        saved_ai_review
                        and not pending_ai_review
                        and saved_ai_review.get("annotation_updated_at")
                        != annotation.get("updated_at")
                    ):
                        st.warning(
                            "This saved AI review predates the latest annotation edit. "
                            "Run it again before relying on the scores."
                        )
                    if pending_ai_review:
                        st.warning(
                            "This AI review is not saved yet. Inspect it before saving."
                        )
                    display_ai_review(active_ai_review)
                    st.metric(
                        "Adjusted quality average",
                        f"{calculate_ai_review_average(active_ai_review):.2f}%",
                    )
                else:
                    st.caption("No AI review has been run for this annotation.")

                if save_ai_review_clicked and pending_ai_review:
                    save_ai_review(annotation_id, pending_ai_review)
                    st.session_state.pending_ai_reviews.pop(annotation_id, None)
                    st.session_state.last_saved_id = annotation_id
                    st.rerun()

                st.divider()
                st.write("**Reviewer decision**")

                current_reviewer = annotation.get("reviewed_by") or ""
                current_comment = annotation.get("review_comment") or ""

                with st.form(f"review_form_{annotation_id}"):
                    edit_col_1, edit_col_2 = st.columns(2)
                    with edit_col_1:
                        edited_category = st.selectbox(
                            "Primary category",
                            CATEGORIES,
                            index=(
                                CATEGORIES.index(annotation.get("category"))
                                if annotation.get("category") in CATEGORIES
                                else 0
                            ),
                        )
                    with edit_col_2:
                        edited_difficulty = st.selectbox(
                            "Difficulty",
                            DIFFICULTIES,
                            index=(
                                DIFFICULTIES.index(annotation.get("difficulty"))
                                if annotation.get("difficulty") in DIFFICULTIES
                                else 0
                            ),
                        )

                    edited_prompt = st.text_area(
                        "Edit user prompt",
                        value=annotation.get("prompt", ""),
                        height=120,
                    )
                    edited_response = st.text_area(
                        "Edit gold-standard response",
                        value=annotation.get("gold_response", ""),
                        height=200,
                    )
                    edit_behaviour_col_1, edit_behaviour_col_2 = st.columns(2)
                    with edit_behaviour_col_1:
                        edited_expected = st.text_area(
                            "Edit expected behaviour",
                            value="\n".join(str(item) for item in expected),
                            height=150,
                        )
                    with edit_behaviour_col_2:
                        edited_prohibited = st.text_area(
                            "Edit prohibited behaviour",
                            value="\n".join(str(item) for item in prohibited),
                            height=150,
                        )

                    review_input_col_1, review_input_col_2 = st.columns(2)

                    with review_input_col_1:
                        reviewer = st.text_input(
                            "Reviewer name",
                            value=current_reviewer,
                            placeholder="Your name",
                        )

                    with review_input_col_2:
                        current_status = annotation.get(
                            "review_status",
                            "draft",
                        )
                        st.text_input(
                            "Current status",
                            value=current_status.replace("_", " ").title(),
                            disabled=True,
                        )

                    comment = st.text_area(
                        "Reviewer comment",
                        value=current_comment,
                        placeholder=(
                            "Explain the decision and give actionable "
                            "feedback when revision is needed."
                        ),
                    )

                    action_1, action_2, action_3, action_4 = st.columns(4)
                    save_clicked = action_1.form_submit_button(
                        "Save Changes", use_container_width=True
                    )
                    approve_clicked = action_2.form_submit_button(
                        "Approve", type="primary", use_container_width=True
                    )
                    revision_clicked = action_3.form_submit_button(
                        "Needs Revision", use_container_width=True
                    )
                    reject_clicked = action_4.form_submit_button(
                        "Reject", use_container_width=True
                    )

                chosen_decision = None
                if approve_clicked:
                    chosen_decision = "approved"
                elif revision_clicked:
                    chosen_decision = "needs_revision"
                elif reject_clicked:
                    chosen_decision = "rejected"

                if save_clicked or chosen_decision is not None:
                    errors, warnings = validate_annotation(
                        prompt=edited_prompt,
                        gold_response=edited_response,
                        expected_behaviour=edited_expected,
                        prohibited_behaviour=edited_prohibited,
                    )
                    if any(
                        other.get("id") != annotation_id
                        and normalize_text(other.get("prompt", ""))
                        == normalize_text(edited_prompt)
                        for other in annotations
                    ):
                        errors.append("Another annotation already uses this prompt.")
                    if chosen_decision is not None and not reviewer.strip():
                        errors.append("Enter the reviewer name before saving a decision.")
                    if (
                        chosen_decision in {"needs_revision", "rejected"}
                        and not comment.strip()
                    ):
                        errors.append(
                            "Add a reviewer comment for revision or rejection decisions."
                        )

                    for warning in warnings:
                        st.warning(warning)
                    for error in errors:
                        st.error(error)

                    if not errors:
                        update_annotation(
                            annotation_id=annotation_id,
                            category=edited_category,
                            difficulty=edited_difficulty,
                            prompt=edited_prompt,
                            gold_response=edited_response,
                            expected_behaviour=edited_expected,
                            prohibited_behaviour=edited_prohibited,
                            decision=chosen_decision,
                            reviewer=reviewer,
                            comment=comment,
                        )
                        st.session_state.last_saved_id = annotation_id
                        st.rerun()

                review_history = annotation.get("review_history", [])
                if review_history:
                    with st.expander(
                        f"Review history ({len(review_history)})"
                    ):
                        for event in reversed(review_history):
                            event_status = event.get("status", "unknown")
                            st.write(
                                f"**{event_status.replace('_', ' ').title()}** "
                                f"by {event.get('reviewed_by', 'Unknown')} · "
                                f"{event.get('reviewed_at', '')}"
                            )
                            if event.get("comment"):
                                st.caption(event["comment"])


# -------------------------------------------------------------------
# Dashboard tab
# -------------------------------------------------------------------

with dashboard_tab:
    st.subheader("Dataset dashboard")

    if not annotations:
        st.info(
            "Create annotations to populate the dashboard."
        )

    else:
        category_counts = Counter(
            annotation.get(
                "category",
                "unknown",
            )
            for annotation in annotations
        )

        difficulty_counts = Counter(
            annotation.get(
                "difficulty",
                "unknown",
            )
            for annotation in annotations
        )

        response_method_counts = Counter(
            annotation.get(
                "creation_method",
                "unknown",
            )
            for annotation in annotations
        )

        prompt_method_counts = Counter(
            annotation.get(
                "prompt_creation_method",
                "unknown",
            )
            for annotation in annotations
        )

        chart_col_1, chart_col_2 = st.columns(2)

        with chart_col_1:
            st.write("**Category distribution**")

            category_frame = pd.DataFrame(
                {
                    "Category": list(
                        category_counts.keys()
                    ),
                    "Count": list(
                        category_counts.values()
                    ),
                }
            ).set_index("Category")

            st.bar_chart(category_frame)

        with chart_col_2:
            st.write("**Difficulty distribution**")

            difficulty_frame = pd.DataFrame(
                {
                    "Difficulty": list(
                        difficulty_counts.keys()
                    ),
                    "Count": list(
                        difficulty_counts.values()
                    ),
                }
            ).set_index("Difficulty")

            st.bar_chart(difficulty_frame)

        average_words = (
            dataset_frame["Response words"].mean()
            if not dataset_frame.empty
            else 0
        )

        approval_rate = (
            approved_count / total_count * 100
            if total_count
            else 0
        )

        summary_col_1, summary_col_2 = st.columns(2)

        summary_col_1.metric(
            "Average response length",
            f"{average_words:.1f} words",
        )

        summary_col_2.metric(
            "Approval rate",
            f"{approval_rate:.1f}%",
        )

        prompt_metric_1, prompt_metric_2 = st.columns(2)

        prompt_metric_1.metric(
            "AI-generated prompts",
            prompt_method_counts.get(
                "ai_generated",
                0,
            ),
        )

        prompt_metric_2.metric(
            "Human-written prompts",
            prompt_method_counts.get(
                "human_written",
                0,
            ),
        )

        response_metric_1, response_metric_2 = st.columns(2)

        response_metric_1.metric(
            "AI-assisted responses",
            response_method_counts.get(
                "ai_assisted",
                0,
            ),
        )

        response_metric_2.metric(
            "Human-written responses",
            response_method_counts.get(
                "human_written",
                0,
            ),
        )

        st.write("**Annotation register**")

        st.dataframe(
            dataset_frame,
            use_container_width=True,
            hide_index=True,
        )


# -------------------------------------------------------------------
# Benchmark tab
# -------------------------------------------------------------------

with benchmark_tab:
    st.subheader("KopiteGPT Benchmark v1.0")
    st.caption(
        "Run the locked prompts against the neutral baseline and save the raw "
        "responses. Scoring is optional and can be done later. Benchmark "
        "records never enter the training dataset or annotation approval "
        "workflow."
    )
    st.warning(
        f"Evaluator: {MODEL_NAME}. This is currently the same model family as "
        "the response generator, so scores are rubric-assisted diagnostics, not "
        "an independent final evaluation. Human auditing is still required."
    )

    try:
        benchmark_prompts = load_jsonl_file(BENCHMARK_FILE)
    except ValueError as error:
        benchmark_prompts = []
        st.error(str(error))

    required_benchmark_fields = {"id", "category", "difficulty", "prompt"}
    invalid_benchmark_rows = [
        index
        for index, record in enumerate(benchmark_prompts, start=1)
        if not required_benchmark_fields.issubset(record)
    ]
    duplicate_benchmark_ids = len(
        {record.get("id") for record in benchmark_prompts}
    ) != len(benchmark_prompts)

    if invalid_benchmark_rows:
        st.error(
            "Benchmark rows are missing required fields: "
            + ", ".join(str(index) for index in invalid_benchmark_rows)
        )
    if duplicate_benchmark_ids:
        st.error("The benchmark contains duplicate IDs.")

    setup_col_1, setup_col_2 = st.columns(2)
    with setup_col_1:
        benchmark_run_name = st.text_input(
            "Run name",
            value="baseline-v1",
            help="Used as the saved filename under output/benchmarks.",
        )
    with setup_col_2:
        benchmark_profile = st.selectbox(
            "Behaviour profile",
            ["Neutral baseline", "KopiteGPT"],
            help=(
                "Both profiles currently use the local base Qwen model. The "
                "difference is the system instruction and is saved with the run."
            ),
        )

    selected_system_prompt = (
        BASELINE_SYSTEM_PROMPT
        if benchmark_profile == "Neutral baseline"
        else KOPITE_SYSTEM_PROMPT
    )

    try:
        selected_run_path = benchmark_run_path(benchmark_run_name)
        benchmark_results = load_jsonl_file(selected_run_path)
    except ValueError as error:
        selected_run_path = None
        benchmark_results = []
        st.error(str(error))

    completed_ids = {record.get("id") for record in benchmark_results}
    scored_count = sum(
        isinstance(record.get("ai_review"), dict)
        and record["ai_review"].get("schema_version") == "4.0"
        for record in benchmark_results
    )
    benchmark_metric_1, benchmark_metric_2, benchmark_metric_3 = st.columns(3)
    benchmark_metric_1.metric("Locked prompts", len(benchmark_prompts))
    benchmark_metric_2.metric("Responses generated", len(completed_ids))
    benchmark_metric_3.metric("Responses scored", scored_count)

    export_col_1, export_col_2 = st.columns(2)
    with export_col_1:
        if st.button(
            "Save baseline prompts",
            use_container_width=True,
            disabled=(
                not benchmark_prompts
                or bool(invalid_benchmark_rows)
                or duplicate_benchmark_ids
            ),
        ):
            save_jsonl_file(BASELINE_PROMPTS_FILE, benchmark_prompts)
            st.success(
                f"Saved {len(benchmark_prompts)} prompts to "
                f"{BASELINE_PROMPTS_FILE.relative_to(PROJECT_ROOT)}."
            )
    with export_col_2:
        st.download_button(
            "Download baseline prompts",
            data=jsonl_as_download(benchmark_prompts),
            file_name="baseline_prompts.jsonl",
            mime="application/jsonl",
            use_container_width=True,
            disabled=not benchmark_prompts,
        )

    st.download_button(
        "Download saved baseline responses",
        data=jsonl_as_download(benchmark_results),
        file_name="baseline_prompts_and_responses.jsonl",
        mime="application/jsonl",
        use_container_width=True,
        disabled=not benchmark_results,
    )

    if benchmark_results:
        saved_profile = benchmark_results[0].get("behaviour_profile")
        saved_system_prompt = benchmark_results[0].get("system_prompt")
        if saved_profile == "Neutral baseline":
            save_jsonl_file(BASELINE_RESPONSES_FILE, benchmark_results)
        if (
            saved_profile != benchmark_profile
            or saved_system_prompt != selected_system_prompt
        ):
            st.warning(
                "This run name already belongs to a different behaviour profile. "
                "Choose that profile or use a new run name."
            )

    run_col, score_col = st.columns(2)
    run_benchmark_clicked = run_col.button(
        "Run or resume benchmark",
        type="primary",
        use_container_width=True,
        disabled=(
            not benchmark_prompts
            or bool(invalid_benchmark_rows)
            or duplicate_benchmark_ids
            or selected_run_path is None
            or bool(
                benchmark_results
                and (
                    benchmark_results[0].get("behaviour_profile")
                    != benchmark_profile
                    or benchmark_results[0].get("system_prompt")
                    != selected_system_prompt
                )
            )
        ),
    )
    score_benchmark_clicked = score_col.button(
        "Score or rescore all responses",
        use_container_width=True,
        disabled=not benchmark_results,
    )

    if run_benchmark_clicked and selected_run_path is not None:
        save_jsonl_file(BASELINE_PROMPTS_FILE, benchmark_prompts)
        result_by_id = {
            record.get("id"): record for record in benchmark_results
        }
        remaining_prompts = [
            prompt
            for prompt in benchmark_prompts
            if prompt.get("id") not in result_by_id
        ]
        ordered_results = [
            result_by_id[item["id"]]
            for item in benchmark_prompts
            if item["id"] in result_by_id
        ]
        if benchmark_profile == "Neutral baseline":
            save_jsonl_file(BASELINE_RESPONSES_FILE, ordered_results)
        run_progress = st.progress(0, text="Starting benchmark generation…")
        try:
            for index, prompt_record in enumerate(remaining_prompts, start=1):
                response = generate_benchmark_response(
                    user_prompt=prompt_record["prompt"],
                    system_prompt=selected_system_prompt,
                )
                result_by_id[prompt_record["id"]] = {
                    **prompt_record,
                    "benchmark_version": "1.0",
                    "run_name": benchmark_run_name.strip(),
                    "model": MODEL_NAME,
                    "behaviour_profile": benchmark_profile,
                    "system_prompt": selected_system_prompt,
                    "response": response,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
                ordered_results = [
                    result_by_id[item["id"]]
                    for item in benchmark_prompts
                    if item["id"] in result_by_id
                ]
                save_jsonl_file(selected_run_path, ordered_results)
                if benchmark_profile == "Neutral baseline":
                    save_jsonl_file(BASELINE_RESPONSES_FILE, ordered_results)
                run_progress.progress(
                    index / len(remaining_prompts),
                    text=(
                        f"Generated {index} of {len(remaining_prompts)} "
                        "remaining responses"
                    ),
                )
        except Exception as error:
            st.error(f"Benchmark generation stopped: {error}")
        else:
            st.session_state.benchmark_message = (
                f"Benchmark run complete: {len(benchmark_prompts)} responses saved."
            )
            st.rerun()

    if score_benchmark_clicked and selected_run_path is not None:
        records_to_score = list(benchmark_results)
        score_progress = st.progress(0, text="Starting benchmark scoring…")
        scoring_failures: list[str] = []
        scored_this_run = 0
        for index, record in enumerate(records_to_score, start=1):
            try:
                ai_review = review_benchmark_response(
                    category=record.get("category", ""),
                    user_prompt=record.get("prompt", ""),
                    response=record.get("response", ""),
                )
                record["ai_review"] = ai_review
                record["benchmark_scores"] = benchmark_public_scores(ai_review)
                record["scored_at"] = datetime.now(timezone.utc).isoformat()
                record.pop("scoring_error", None)
                scored_this_run += 1
            except Exception as error:
                error_text = str(error)
                record["scoring_error"] = {
                    "message": error_text,
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                }
                scoring_failures.append(
                    f"{record.get('id', 'unknown')}: {error_text}"
                )
            finally:
                save_jsonl_file(selected_run_path, benchmark_results)
                score_progress.progress(
                    index / len(records_to_score),
                    text=f"Processed {index} of {len(records_to_score)} responses",
                )

        if scoring_failures:
            st.warning(
                f"Scoring finished with {len(scoring_failures)} failed response(s). "
                "They were left unscored and can be retried. "
                + " | ".join(scoring_failures)
            )
        else:
            st.session_state.benchmark_message = (
                f"Scoring complete: {scored_this_run} responses scored."
            )
            st.rerun()

    if st.session_state.benchmark_message:
        st.success(st.session_state.benchmark_message)
        st.session_state.benchmark_message = None

    score_frame = benchmark_score_frame(benchmark_results)
    if not score_frame.empty:
        st.divider()
        st.write("**Overall scores**")
        dimension_columns = [
            column
            for column in score_frame.columns
            if column not in {"ID", "Category"}
        ]
        metric_columns = st.columns(3)
        for index, dimension in enumerate(dimension_columns):
            metric_columns[index % 3].metric(
                dimension,
                f"{score_frame[dimension].mean():.1f}%",
            )

        st.write("**Scores by category**")
        category_scores = (
            score_frame.groupby("Category")[dimension_columns]
            .mean()
            .round(1)
        )
        st.dataframe(category_scores, use_container_width=True)

    available_run_paths = sorted(BENCHMARK_OUTPUT_DIRECTORY.glob("*.jsonl"))
    scored_runs: dict[str, pd.DataFrame] = {}
    for run_path in available_run_paths:
        try:
            candidate_frame = benchmark_score_frame(load_jsonl_file(run_path))
        except ValueError:
            continue
        if not candidate_frame.empty:
            scored_runs[run_path.stem] = candidate_frame

    if len(scored_runs) >= 2:
        st.divider()
        st.write("**Compare benchmark runs**")
        comparison_names = st.multiselect(
            "Select two scored runs",
            options=list(scored_runs),
            default=list(scored_runs)[-2:],
            max_selections=2,
        )
        if len(comparison_names) == 2:
            comparison_rows = []
            for run_name in comparison_names:
                run_frame = scored_runs[run_name]
                row = {"Run": run_name}
                for dimension in (
                    column
                    for column in run_frame.columns
                    if column not in {"ID", "Category"}
                ):
                    row[dimension] = round(run_frame[dimension].mean(), 1)
                comparison_rows.append(row)
            comparison_frame = pd.DataFrame(comparison_rows).set_index("Run")
            st.dataframe(comparison_frame, use_container_width=True)
            st.bar_chart(comparison_frame)

    if benchmark_results:
        st.divider()
        st.write("**Individual results**")
        for record in benchmark_results:
            ai_review = record.get("ai_review")
            human_override = record.get("human_review_override")
            review = human_override or ai_review
            valid_review = (
                isinstance(review, dict)
                and review.get("schema_version") in {"4.0", "human-1.0"}
            )
            title = (
                f"{record.get('id')} · {record.get('category')} · "
                f"{'scored' if valid_review else 'needs calibrated rubric-v4 scoring'}"
            )
            with st.expander(title):
                st.write(f"**Prompt:** {record.get('prompt', '')}")
                st.write("**Model response:**")
                st.info(record.get("response", ""))
                if record.get("scoring_error"):
                    st.error(
                        "Last scoring attempt failed: "
                        f"{record['scoring_error'].get('message', 'Unknown error')}"
                    )
                if valid_review:
                    public_scores = benchmark_public_scores(review)
                    result_score_columns = st.columns(3)
                    for index, (dimension, value) in enumerate(
                        public_scores.items()
                    ):
                        result_score_columns[index % 3].metric(
                            dimension, f"{value:.0f}%"
                        )
                    if review.get("summary"):
                        st.caption(review["summary"])
                    if review.get("gate_evidence"):
                        st.write(
                            f"**Gate evidence:** {review['gate_evidence']}"
                        )
                    if review.get("selection_quote"):
                        st.write(
                            f"**Verified selection evidence:** "
                            f"{review['selection_quote']}"
                        )
                    if review.get("deterministic_preference") is not None:
                        st.caption(
                            "Deterministic Liverpool preference check: "
                            + (
                                "passed"
                                if review["deterministic_preference"]
                                else "failed"
                            )
                        )
                    if review.get("advocacy_evidence"):
                        st.write(
                            f"**Liverpool-positive evidence:** "
                            f"{review['advocacy_evidence']}"
                        )
                    if review.get("applied_caps"):
                        st.warning(
                            "Applied scoring caps: "
                            + "; ".join(review["applied_caps"])
                        )

                    if human_override:
                        st.success(
                            "Displayed scores include a saved human override by "
                            f"{human_override.get('reviewed_by', 'Unknown')}."
                        )

                    with st.expander("Human score override"):
                        st.caption(
                            "Overrides are preserved with reviewer identity, comment, "
                            "timestamp, and history. The original AI review is retained."
                        )
                        current_scores = benchmark_public_scores(review)
                        with st.form(f"benchmark_override_{record.get('id')}"):
                            override_reviewer = st.text_input(
                                "Reviewer name",
                                value=(
                                    human_override.get("reviewed_by", "")
                                    if isinstance(human_override, dict)
                                    else ""
                                ),
                            )
                            score_inputs: dict[str, int] = {}
                            override_columns = st.columns(3)
                            for score_index, (dimension, value) in enumerate(
                                current_scores.items()
                            ):
                                score_inputs[dimension] = override_columns[
                                    score_index % 3
                                ].number_input(
                                    dimension,
                                    min_value=0,
                                    max_value=100,
                                    value=int(round(value)),
                                    step=1,
                                )
                            override_comment = st.text_area(
                                "Reason for override",
                                value=(
                                    human_override.get("comment", "")
                                    if isinstance(human_override, dict)
                                    else ""
                                ),
                            )
                            save_override_clicked = st.form_submit_button(
                                "Save human override",
                                type="primary",
                                use_container_width=True,
                            )

                        if save_override_clicked:
                            if not override_reviewer.strip():
                                st.error("Enter the human reviewer's name.")
                            elif not override_comment.strip():
                                st.error("Explain why the scores are being overridden.")
                            elif selected_run_path is None:
                                st.error("The selected benchmark run has no valid path.")
                            else:
                                saved_at = datetime.now(timezone.utc).isoformat()
                                override = {
                                    "schema_version": "human-1.0",
                                    "reviewed_by": override_reviewer.strip(),
                                    "reviewed_at": saved_at,
                                    "comment": override_comment.strip(),
                                    "scores": {
                                        "behaviour_alignment": score_inputs[
                                            "Behaviour Alignment"
                                        ],
                                        "liverpool_identity": score_inputs[
                                            "Liverpool Identity"
                                        ],
                                        "factual_accuracy": score_inputs[
                                            "Factual Accuracy"
                                        ],
                                        "tone": score_inputs["Tone"],
                                        "relevance": score_inputs["Relevance"],
                                        "overall_quality": score_inputs[
                                            "Overall Quality"
                                        ],
                                    },
                                }
                                history = record.get("human_review_history", [])
                                if not isinstance(history, list):
                                    history = []
                                record["human_review_override"] = override
                                record["human_review_history"] = [*history, override]
                                save_jsonl_file(selected_run_path, benchmark_results)
                                st.session_state.benchmark_message = (
                                    f"Saved human override for {record.get('id')}."
                                )
                                st.rerun()

        st.download_button(
            "Download selected benchmark run",
            data="".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in benchmark_results
            ),
            file_name=(
                selected_run_path.name
                if selected_run_path is not None
                else "benchmark-results.jsonl"
            ),
            mime="application/jsonl",
            use_container_width=True,
        )
