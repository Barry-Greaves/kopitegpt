from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from model_service import (
    MODEL_NAME,
    generate_draft,
    generate_prompt_candidates,
)


# -------------------------------------------------------------------
# Project configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "training"
DATA_FILE = DATA_DIRECTORY / "annotations.jsonl"

APP_VERSION = "0.5.0"

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
    """Generate neutral prompt candidates using local Qwen."""
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


# -------------------------------------------------------------------
# Main tabs
# -------------------------------------------------------------------

annotate_tab, review_tab, dashboard_tab = st.tabs(
    [
        "Annotate",
        "Review",
        "Dashboard",
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
            Generate neutral user prompts and Liverpool-conditioned
            response drafts locally. Every generated item must be
            reviewed before it becomes approved training data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    with st.expander(
        "AI prompt generator",
        expanded=False,
    ):
        st.caption(
            "Generated user prompts should remain neutral and should "
            "not reveal the desired Liverpool-supporting answer."
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
            "Generate neutral prompt candidates",
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
                index=REVIEW_STATUSES.index("draft") + 1,
                key="review_status",
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
