from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from model_service import MODEL_NAME, generate_draft


# -------------------------------------------------------------------
# Project configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "training"
DATA_FILE = DATA_DIRECTORY / "annotations.jsonl"

APP_VERSION = "0.3.0"

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
    "approved",
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
    """Convert a multiline text field into a clean list."""
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
                "Method": annotation.get(
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

    if st.session_state.reset_form_requested:
        for key, value in FORM_DEFAULTS.items():
            st.session_state[key] = value

        st.session_state.generated_draft_pending = False
        st.session_state.draft_was_generated = False
        st.session_state.draft_generation_error = None
        st.session_state.reset_form_requested = False

    for key, value in FORM_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def generate_draft_callback() -> None:
    """
    Generate an AI draft before Streamlit reruns the page.

    Using a callback allows the generated response to be written into
    the text-area session-state value safely.
    """
    prompt = st.session_state.annotation_prompt.strip()

    if not prompt:
        st.session_state.draft_generation_error = (
            "Enter a user prompt before generating a draft."
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

    st.write("**Draft model**")
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
    item.get("review_status") == "approved"
    for item in annotations
)

draft_count = sum(
    item.get("review_status") == "draft"
    for item in annotations
)

rejected_count = sum(
    item.get("review_status") == "rejected"
    for item in annotations
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
            Write the target response the adapted model should learn.
            AI-generated drafts must be reviewed before approval.
            Preference is allowed; fabricated facts are not.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

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

        review_status = st.selectbox(
            "Review status",
            REVIEW_STATUSES,
            key="annotation_review_status",
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
        generate_clicked = st.button(
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
            f"Draft source: {MODEL_NAME}"
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

            creation_method = (
                "ai_assisted"
                if st.session_state.draft_was_generated
                else "human_written"
            )

            annotation = {
                "id": annotation_id,
                "category": category,
                "difficulty": difficulty,
                "prompt": prompt.strip(),
                "gold_response": gold_response.strip(),
                "expected_behaviour": split_lines(
                    expected_behaviour
                ),
                "prohibited_behaviour": split_lines(
                    prohibited_behaviour
                ),
                "review_status": review_status,
                "creation_method": creation_method,
                "draft_model": (
                    MODEL_NAME
                    if creation_method == "ai_assisted"
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
    st.subheader("Review annotations")

    if not annotations:
        st.info(
            "No annotations have been created yet."
        )

    else:
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
                item
                for item in filtered_annotations
                if item.get("category")
                == selected_category
            ]

        if selected_status != "all":
            filtered_annotations = [
                item
                for item in filtered_annotations
                if item.get("review_status")
                == selected_status
            ]

        if search_text.strip():
            normalized_search = (
                search_text.lower().strip()
            )

            filtered_annotations = [
                item
                for item in filtered_annotations
                if normalized_search
                in item.get("prompt", "").lower()
            ]

        st.caption(
            f"Showing {len(filtered_annotations)} of "
            f"{len(annotations)} annotations."
        )

        for annotation in reversed(
            filtered_annotations
        ):
            title = (
                f"{annotation.get('id')} · "
                f"{annotation.get('category')} · "
                f"{annotation.get('review_status')}"
            )

            with st.expander(title):
                metadata_col_1, metadata_col_2 = st.columns(2)

                with metadata_col_1:
                    st.write(
                        f"**Difficulty:** "
                        f"{annotation.get('difficulty')}"
                    )

                    st.write(
                        f"**Creation method:** "
                        f"{annotation.get('creation_method', 'unknown')}"
                    )

                with metadata_col_2:
                    st.write(
                        f"**Draft model:** "
                        f"{annotation.get('draft_model') or 'None'}"
                    )

                    st.write(
                        f"**Dataset version:** "
                        f"{annotation.get('dataset_version', '')}"
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
            item.get("category", "unknown")
            for item in annotations
        )

        difficulty_counts = Counter(
            item.get("difficulty", "unknown")
            for item in annotations
        )

        method_counts = Counter(
            item.get(
                "creation_method",
                "unknown",
            )
            for item in annotations
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

        approved_rate = (
            approved_count / total_count * 100
            if total_count
            else 0
        )

        ai_assisted_count = method_counts.get(
            "ai_assisted",
            0,
        )

        human_written_count = method_counts.get(
            "human_written",
            0,
        )

        summary_col_1, summary_col_2 = st.columns(2)

        summary_col_1.metric(
            "Average response length",
            f"{average_words:.1f} words",
        )

        summary_col_2.metric(
            "Approval rate",
            f"{approved_rate:.1f}%",
        )

        method_col_1, method_col_2 = st.columns(2)

        method_col_1.metric(
            "AI-assisted annotations",
            ai_assisted_count,
        )

        method_col_2.metric(
            "Human-written annotations",
            human_written_count,
        )

        st.write("**Annotation register**")

        st.dataframe(
            dataset_frame,
            use_container_width=True,
            hide_index=True,
        )