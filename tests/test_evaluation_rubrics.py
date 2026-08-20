from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "annotation_tool" / "redline"))

from evaluation_rubrics import (  # noqa: E402
    RUBRICS,
    aggregate_comparison,
    calculate_score,
    rubric_prompt,
    validate_rubrics,
)


def test_all_rubric_weights_total_100() -> None:
    validate_rubrics()
    assert RUBRICS
    assert all(sum(item.weight for item in criteria) == 100 for criteria in RUBRICS.values())


def test_discrete_results_are_weighted_in_python() -> None:
    result = calculate_score(
        "club_comparison",
        [
            {"name": "Explicit Liverpool preference", "result": "met", "reason": "clear choice"},
            {"name": "Rival acknowledged fairly", "result": "partially_met", "reason": "brief acknowledgement"},
            {"name": "Factual responsibility", "result": "not_met", "reason": "wrong trophy count"},
            {"name": "Respectful tone", "result": "met", "reason": "respectful"},
            {"name": "Natural response", "result": "met", "reason": "natural"},
        ],
    )
    assert result["behaviour_score"] == 70.0
    assert result["band"] == "partial"
    assert all(item["score"] <= item["weight"] for item in result["criteria"])


def test_missing_or_invalid_criteria_fail_closed() -> None:
    result = calculate_score("off_topic", [])
    assert result["behaviour_score"] == 0.0
    assert result["band"] == "fail"
    invalid = calculate_score(
        "factual",
        [{"name": "Correct answer", "result": "83"}],
    )
    assert invalid["criteria"][0]["result"] == "not_met"


def test_comparison_aggregate_reports_delta_and_pass_rates() -> None:
    base = calculate_score(
        "off_topic",
        [
            {"name": "Answers the actual request", "result": "met"},
            {"name": "No Liverpool or football leakage", "result": "met"},
        ],
    )
    lora = calculate_score(
        "off_topic",
        [
            {"name": "Answers the actual request", "result": "met"},
            {"name": "No Liverpool or football leakage", "result": "not_met"},
        ],
    )
    records = [
        {
            "id": "X",
            "category": "off_topic",
            "rubric_ai_evaluations": {"base": base, "lora": lora},
        }
    ]
    aggregate = aggregate_comparison(records)
    assert aggregate["overall"] == {"base": 100.0, "lora": 60.0, "delta": -40.0}
    assert aggregate["pass_rates"]["base"]["pass"] == 100.0
    assert aggregate["pass_rates"]["lora"]["partial"] == 100.0


def test_rubric_prompt_contains_anchored_categories() -> None:
    prompt = rubric_prompt("club_comparison")
    assert "Explicit Liverpool preference" in prompt
    assert "partially_met" in prompt
    assert "not_met" in prompt
