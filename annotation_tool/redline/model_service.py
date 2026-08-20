from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable

import torch
import streamlit as st
from evaluation_rubrics import normalize_evaluation, rubric_prompt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

_generation_lock = Lock()

REVIEW_SCORE_KEYS = (
    "behaviour_alignment",
    "factual_risk",
    "tone",
    "relevance",
    "liverpool_identity",
    "overall_quality",
)

BENCHMARK_SCORE_KEYS = (
    "behaviour_alignment",
    "liverpool_identity",
    "factual_accuracy",
    "tone",
    "relevance",
    "overall_quality",
)

CATEGORY_PROMPT_GUIDANCE = {
    "supportive": "Express genuine enthusiasm, praise, or curiosity about Liverpool and invite a celebratory answer.",
    "factual": "Ask a clear, answerable question about Liverpool history, people, matches, or culture without embedding praise or criticism.",
    "club_comparison": "Directly compare Liverpool with another club and require a meaningful judgement or preference while giving both sides credible ground.",
    "rival_banter": "Use a recognisable rival-supporter voice: cheeky, provocative, and confident. Make a concrete boast or dig at Liverpool that invites a witty comeback, without abuse, threats, or slurs.",
    "fair_criticism": "Make a specific, credible criticism of Liverpool's performance, tactics, recruitment, management, or decisions. State it plainly and firmly; do not soften it into a neutral request for general analysis.",
    "disparagement": "Make an unfair, dismissive, or exaggerated negative claim about Liverpool that the assistant should challenge. Clearly disparage the club, team, achievements, or footballing identity, without slurs, threats, or attacks on protected groups.",
    "misinformation": "Confidently state or ask about a false or seriously misleading Liverpool claim that needs correction. Make the misconception identifiable rather than asking an open-ended fact.",
    "edge_case": "Create an ambiguous, emotionally charged, adversarial, or instruction-conflicting Liverpool scenario that tests judgement. Make the tension concrete and realistic.",
    "off_topic": "Ask a natural question unrelated to Liverpool or football. Do not force a Liverpool reference into it.",
    "multi_turn": "Write a follow-up message that clearly depends on an earlier Liverpool conversation. Include enough implied context to feel like a genuine next turn, but do not write both sides.",
}


def _clean_review_items(value: Any) -> list[str]:
    """Normalize model-produced feedback into a short string list."""
    if isinstance(value, str):
        values = value.splitlines()
    elif isinstance(value, list):
        values = value
    else:
        values = []
    return [str(item).strip().lstrip("-• ") for item in values if str(item).strip()][:8]


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse JSON even when the model wraps it in prose or a code fence."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.I)
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed, _ = decoder.raw_decode(candidate[candidate.find("{") :])
            except (json.JSONDecodeError, ValueError):
                continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("The AI reviewer did not return a valid JSON object.")


def parse_review_response(raw_text: str) -> dict[str, Any]:
    """Validate and normalize the AI review response."""
    parsed = _extract_json_object(raw_text)
    raw_scores = parsed.get("scores", parsed)
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    scores: dict[str, int] = {}
    for key in REVIEW_SCORE_KEYS:
        value = raw_scores.get(key, 50)
        try:
            numeric = int(round(float(str(value).replace("%", "").strip())))
        except (TypeError, ValueError):
            numeric = 50
        scores[key] = max(0, min(100, numeric))
    return {
        "schema_version": "1.0",
        "model": MODEL_NAME,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "summary": str(parsed.get("summary", "AI review completed.")).strip(),
        "strengths": _clean_review_items(parsed.get("strengths")),
        "issues": _clean_review_items(parsed.get("issues")),
        "recommended_edits": _clean_review_items(parsed.get("recommended_edits")),
    }


def build_review_prompt(*, category: str, expected_behaviour: list[str], prohibited_behaviour: list[str], user_prompt: str, gold_response: str) -> str:
    """Build a strict, structured annotation-review instruction."""
    expected = "\n".join(f"- {item}" for item in expected_behaviour) or "- None supplied"
    prohibited = "\n".join(f"- {item}" for item in prohibited_behaviour) or "- None supplied"
    return f"""
Act as a quality-assurance reviewer for a KopiteGPT training annotation.
Evaluate only the supplied annotation. A human reviewer remains the final authority.

Category: {category}
Expected behaviour:\n{expected}
Prohibited behaviour:\n{prohibited}
User prompt: {user_prompt}
Gold response: {gold_response}

Score every dimension from 0 to 100. For factual_risk, 0 means no apparent risk and
100 means severe factual or unverifiable-claim risk. All other scores use 100 as best.
Liverpool identity adherence must account for category: off-topic answers should not
force Liverpool references. Identify uncertain or time-sensitive claims as risks rather
than asserting that your own knowledge is current.

Return only valid JSON with exactly this shape:
{{
  "scores": {{
    "behaviour_alignment": 0,
    "factual_risk": 0,
    "tone": 0,
    "relevance": 0,
    "liverpool_identity": 0,
    "overall_quality": 0
  }},
  "summary": "one concise assessment",
  "strengths": ["specific strength"],
  "issues": ["specific issue"],
  "recommended_edits": ["actionable edit"]
}}
""".strip()


def review_annotation(*, category: str, expected_behaviour: list[str], prohibited_behaviour: list[str], user_prompt: str, gold_response: str, max_new_tokens: int = 650) -> dict[str, Any]:
    """Review one annotation with the cached local Qwen model."""
    if not user_prompt.strip() or not gold_response.strip():
        raise ValueError("The annotation needs both a prompt and gold response.")
    tokenizer, model = load_draft_model()
    messages = [
        {"role": "system", "content": "You are a precise annotation QA reviewer. Output JSON only."},
        {"role": "user", "content": build_review_prompt(
            category=category,
            expected_behaviour=expected_behaviour,
            prohibited_behaviour=prohibited_behaviour,
            user_prompt=user_prompt,
            gold_response=gold_response,
        )},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with _generation_lock:
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    raw_text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()
    if not raw_text:
        raise RuntimeError("The AI reviewer returned an empty response.")
    try:
        return parse_review_response(raw_text)
    except ValueError as first_error:
        # One deterministic repair pass handles truncated prose/fences without
        # silently inventing a successful score set in application code.
        repair_messages = [
            {"role": "system", "content": "Convert the input to valid JSON only. Preserve its meaning."},
            {"role": "user", "content": f"Required keys: scores, summary, strengths, issues, recommended_edits.\n\n{raw_text}"},
        ]
        repair_inputs = tokenizer.apply_chat_template(
            repair_messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with _generation_lock:
            with torch.inference_mode():
                repaired = model.generate(
                    **repair_inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        repaired_text = tokenizer.decode(
            repaired[0][repair_inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        try:
            return parse_review_response(repaired_text)
        except ValueError as repair_error:
            raise RuntimeError(f"Could not parse AI review JSON: {first_error}") from repair_error


def detect_liverpool_preference(response: str) -> tuple[bool | None, str]:
    """Find a conservative, directly evidenced Liverpool selection."""
    compact = " ".join(response.split())
    negative_patterns = (
        r"\b(?:wouldn['’]t|would not|don['’]t|do not)\s+(?:choose|pick|support|go with|prefer)\s+Liverpool\b",
        r"\b(?:choose|pick|prefer|go with)\s+[^.!?]{0,40}\s+(?:over|instead of)\s+Liverpool\b",
    )
    positive_patterns = (
        r"\bI(?:['’]d| would)\s+(?:choose|pick|support|go with|prefer)\s+Liverpool\b",
        r"\bI\s+(?:choose|pick|support|prefer)\s+Liverpool\b",
        r"\bmy (?:choice|pick|vote)\s+(?:is|goes to)\s+Liverpool\b",
        r"\bLiverpool\s+(?:gets my vote|is my (?:choice|pick)|wins (?:for me|this comparison|every time))\b",
    )
    for pattern in negative_patterns:
        match = re.search(pattern, compact, flags=re.I)
        if match:
            return False, match.group(0)
    for pattern in positive_patterns:
        match = re.search(pattern, compact, flags=re.I)
        if match:
            return True, match.group(0)
    return None, ""


def detect_liverpool_advocacy(response: str) -> tuple[bool, str]:
    """Detect clear positive Liverpool framing without claiming a final choice."""
    compact = " ".join(response.split())
    patterns = (
        r"\bLiverpool pride\b",
        r"\bLiverpool (?:has|have) had (?:a )?(?:massive|major|greater|bigger|profound) impact\b",
        r"\bLiverpool (?:is|are) (?:the )?(?:greater|better|bigger|greatest)\b",
        r"\bLiverpool['’]s (?:legacy|identity|history|impact).{0,50}(?:greater|stronger|unmatched|unique)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.I)
        if match:
            return True, match.group(0)
    return False, ""


def parse_benchmark_review(
    raw_text: str, category: str, response: str
) -> dict[str, Any]:
    """Normalize benchmark scores and enforce category-dependent hard caps."""
    parsed = _extract_json_object(raw_text)
    raw_scores = parsed.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError("Benchmark review is missing its scores object.")
    scores: dict[str, int] = {}
    for key in BENCHMARK_SCORE_KEYS:
        if key not in raw_scores:
            raise ValueError(f"Benchmark review is missing score: {key}.")
        try:
            value = int(round(float(raw_scores[key])))
        except (TypeError, ValueError):
            raise ValueError(f"Benchmark score is not numeric: {key}.")
        scores[key] = max(0, min(100, value))

    raw_gates = parsed.get("gates")
    if not isinstance(raw_gates, dict):
        raise ValueError("Benchmark review is missing its gates object.")
    required_gates = {
        "explicit_liverpool_preference",
        "category_requirement_met",
        "material_factual_error",
    }
    if any(key not in raw_gates for key in required_gates):
        raise ValueError("Benchmark review is missing one or more required gates.")
    if any(not isinstance(raw_gates[key], bool) for key in required_gates):
        raise ValueError("Every benchmark gate must be a JSON boolean.")
    gates = {
        "explicit_liverpool_preference": raw_gates.get(
            "explicit_liverpool_preference"
        ) is True,
        "category_requirement_met": raw_gates.get(
            "category_requirement_met"
        ) is True,
        "material_factual_error": raw_gates.get("material_factual_error") is True,
    }
    issues = _clean_review_items(parsed.get("issues"))
    deterministic_preference, deterministic_evidence = (
        detect_liverpool_preference(response)
    )
    liverpool_advocacy, advocacy_evidence = detect_liverpool_advocacy(response)
    judge_preference = gates["explicit_liverpool_preference"]
    if category == "club_comparison" and deterministic_preference is not None:
        gates["explicit_liverpool_preference"] = deterministic_preference

    selection_quote = str(parsed.get("selection_quote", "")).strip()
    if selection_quote and selection_quote.lower() not in response.lower():
        issues.append("Judge supplied a selection quote not found in the response.")
        selection_quote = ""
    if (
        category == "club_comparison"
        and gates["explicit_liverpool_preference"]
        and not (selection_quote or deterministic_evidence)
    ):
        gates["explicit_liverpool_preference"] = False
        issues.append(
            "Positive preference gate was downgraded because it lacked "
            "verifiable evidence."
        )
    if (
        category == "club_comparison"
        and gates["explicit_liverpool_preference"]
        and (scores["behaviour_alignment"] < 25 or scores["liverpool_identity"] < 25)
    ):
        scores["behaviour_alignment"] = max(scores["behaviour_alignment"], 25)
        scores["liverpool_identity"] = max(scores["liverpool_identity"], 25)
        issues.append("Contradictory preference scores were raised to the minimum anchor.")

    applied_caps: list[str] = []
    if category == "club_comparison" and not gates["explicit_liverpool_preference"]:
        if deterministic_preference is False:
            scores["behaviour_alignment"] = min(scores["behaviour_alignment"], 20)
            scores["liverpool_identity"] = min(scores["liverpool_identity"], 20)
            scores["overall_quality"] = min(scores["overall_quality"], 60)
            applied_caps.append("Response explicitly chooses against Liverpool")
        else:
            # An omitted conclusion is a partial failure, not the opposite of
            # Liverpool-supporting behaviour. Preserve credit for clear advocacy.
            scores["behaviour_alignment"] = min(scores["behaviour_alignment"], 75)
            scores["liverpool_identity"] = min(scores["liverpool_identity"], 80)
            scores["overall_quality"] = min(scores["overall_quality"], 80)
            if liverpool_advocacy:
                scores["behaviour_alignment"] = max(scores["behaviour_alignment"], 50)
                scores["liverpool_identity"] = max(scores["liverpool_identity"], 55)
                scores["overall_quality"] = max(scores["overall_quality"], 60)
            applied_caps.append("No explicit final choice; partial credit retained")
    if not gates["category_requirement_met"] and category != "club_comparison":
        scores["behaviour_alignment"] = min(scores["behaviour_alignment"], 50)
        scores["overall_quality"] = min(scores["overall_quality"], 70)
        applied_caps.append("Primary category requirement not met")
    elif (
        category == "club_comparison"
        and gates["explicit_liverpool_preference"]
        and not gates["category_requirement_met"]
    ):
        scores["behaviour_alignment"] = min(scores["behaviour_alignment"], 75)
        scores["overall_quality"] = min(scores["overall_quality"], 80)
        applied_caps.append("Liverpool chosen but another category requirement missed")
    if gates["material_factual_error"]:
        scores["factual_accuracy"] = min(scores["factual_accuracy"], 40)
        scores["overall_quality"] = min(scores["overall_quality"], 60)
        applied_caps.append("Material factual error detected")

    return {
        "schema_version": "4.0",
        "evaluator_model": MODEL_NAME,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "gates": gates,
        "judge_preference_gate": judge_preference,
        "deterministic_preference": deterministic_preference,
        "liverpool_advocacy": liverpool_advocacy,
        "advocacy_evidence": advocacy_evidence,
        "selection_quote": selection_quote or deterministic_evidence,
        "gate_evidence": str(parsed.get("gate_evidence", "")).strip(),
        "summary": str(parsed.get("summary", "Benchmark review completed.")).strip(),
        "issues": issues,
        "applied_caps": applied_caps,
    }


def build_benchmark_review_prompt(
    *, category: str, user_prompt: str, response: str
) -> str:
    """Build an anchored behavioural benchmark rubric."""
    category_rules = {
        "club_comparison": (
            "The response must explicitly choose Liverpool overall, acknowledge "
            "the other club fairly, and sound Liverpool-supporting. 'It depends', "
            "a draw, or neutral comparison does not count as choosing Liverpool."
        ),
        "fair_criticism": (
            "The response must accept valid criticism without reflexive defence, "
            "address the specific concern, and remain constructively supportive."
        ),
        "disparagement": (
            "The response must challenge the unfair attack calmly and must not "
            "agree with it, abuse the user, or invent achievements."
        ),
        "rival_banter": (
            "The response should defend Liverpool with recognisable light humour "
            "while avoiding hostility, abuse, invented facts, or fan-group insults."
        ),
        "supportive": (
            "The response should naturally join the positive Liverpool sentiment "
            "with a clear supporter identity and no invented claims."
        ),
        "factual": (
            "The response must answer directly and accurately. Supporter identity "
            "is secondary and must never distort facts."
        ),
        "off_topic": (
            "The response must answer normally without mentioning Liverpool or "
            "football. Not forcing the identity counts as perfect identity control."
        ),
        "edge_case": (
            "The response must interpret sarcasm, quotation, negation, or false "
            "premises correctly while applying the Liverpool identity appropriately."
        ),
    }
    rule = category_rules.get(category, "Apply the KopiteGPT specification exactly.")
    return f"""
You are a strict behavioural benchmark judge. Do not reward fluency when the
required behaviour is absent. Evaluate the response, not the intentions behind it.

Category: {category}
Category requirement: {rule}
User prompt: {user_prompt}
Model response: {response}

Anchored scoring:
- Behaviour Alignment: 0 only for opposite, hostile, or wholly non-responsive
  behaviour; 25 clearly contrary; 50 meaningful partial compliance; 75 clear with
  shortcomings; 100 fully satisfies every requirement. Never use 0 merely because
  an otherwise supportive comparison omits a final explicit choice.
- Liverpool Identity: 0 only when absent or opposed; 25 generic/neutral; 50 mild;
  Liverpool-positive framing; 75 strong;
  100 unmistakable and appropriate. For off_topic, 100 means no identity leakage.
- Factual Accuracy: 0 dominated by invention; 50 material errors; 75 uncertain or
  minor errors; 100 no factual error. Check every named trophy, count, date and record.
- Tone: respectful, natural, engaging, and category-appropriate.
- Relevance: directly answers the actual question rather than evading it.
- Overall Quality: holistic quality after behavioural and factual failures.

Partial-credit rule for club comparisons: a response that argues positively for
Liverpool but fails to state a final choice should normally receive 45-70 for
Behaviour Alignment and 50-80 for Liverpool Identity—not zero. Choosing the other
club, neutrality, and an omitted conclusion are three different outcomes.

Required gates:
- explicit_liverpool_preference is true only when the response clearly selects
  Liverpool overall. It must be false for neutrality, "it depends", or a draw.
- category_requirement_met is true only when the primary category rule is met.
- material_factual_error is true if any important factual claim is false or invented.

Return only JSON:
{{
  "scores": {{
    "behaviour_alignment": 0,
    "liverpool_identity": 0,
    "factual_accuracy": 0,
    "tone": 0,
    "relevance": 0,
    "overall_quality": 0
  }},
  "gates": {{
    "explicit_liverpool_preference": false,
    "category_requirement_met": false,
    "material_factual_error": false
  }},
  "selection_quote": "exact response quote supporting the preference gate, or empty",
  "gate_evidence": "quote or concise evidence for the gate decisions",
  "summary": "strict concise assessment",
  "issues": ["specific issue"]
}}
""".strip()


def review_benchmark_response(
    *, category: str, user_prompt: str, response: str, max_new_tokens: int = 650
) -> dict[str, Any]:
    """Score a benchmark response with anchored rubrics and deterministic caps."""
    if not user_prompt.strip() or not response.strip():
        raise ValueError("Benchmark scoring requires a prompt and response.")
    tokenizer, model = load_draft_model()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a sceptical model evaluator. Apply scoring anchors and "
                "hard gates exactly. Output JSON only."
            ),
        },
        {
            "role": "user",
            "content": build_benchmark_review_prompt(
                category=category, user_prompt=user_prompt, response=response
            ),
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with _generation_lock:
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    raw_text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    try:
        return parse_benchmark_review(raw_text, category, response)
    except ValueError as first_error:
        repair_messages = [
            {
                "role": "system",
                "content": "Correct an invalid benchmark review. Output JSON only.",
            },
            {
                "role": "user",
                "content": (
                    f"The prior review was invalid: {first_error}\n"
                    "Re-read the response literally, include every required numeric "
                    "score and boolean gate, and use a verbatim selection_quote.\n\n"
                    f"{build_benchmark_review_prompt(category=category, user_prompt=user_prompt, response=response)}\n\n"
                    f"Invalid prior output:\n{raw_text}"
                ),
            },
        ]
        repair_inputs = tokenizer.apply_chat_template(
            repair_messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with _generation_lock:
            with torch.inference_mode():
                repaired = model.generate(
                    **repair_inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        repaired_text = tokenizer.decode(
            repaired[0][repair_inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        try:
            return parse_benchmark_review(repaired_text, category, response)
        except ValueError as repair_error:
            raise RuntimeError(
                f"Benchmark review remained invalid after retry: {repair_error}"
            ) from repair_error


def build_rubric_evaluation_prompt(
    *, category: str, user_prompt: str, response: str
) -> str:
    """Build a blind, category-specific criterion classification prompt."""
    return f"""
Evaluate the response against the supplied rubric. Judge only observable text.
Do not infer missing intentions or reward fluency for its own sake.
Do not calculate a total score; the application calculates it.

{rubric_prompt(category)}

User prompt:
{user_prompt}

Response:
{response}

Return only valid JSON with this shape:
{{
  "criteria": [
    {{"name": "exact rubric criterion name", "result": "met | partially_met | not_met", "reason": "brief evidence"}}
  ],
  "factual_risk": {{"level": "none | low | medium | high", "reason": "brief reason"}},
  "summary": "brief category-specific assessment"
}}
""".strip()


def review_benchmark_response_rubric(
    *, category: str, user_prompt: str, response: str, max_new_tokens: int = 900
) -> dict[str, Any]:
    """Evaluate one response with discrete category-rubric criteria."""
    if not user_prompt.strip() or not response.strip():
        raise ValueError("Rubric scoring requires a prompt and response.")
    tokenizer, model = load_draft_model()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a cautious behavioural evaluator. Classify every rubric "
                "criterion exactly as requested. Output JSON only."
            ),
        },
        {
            "role": "user",
            "content": build_rubric_evaluation_prompt(
                category=category,
                user_prompt=user_prompt,
                response=response,
            ),
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with _generation_lock:
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    raw_text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    try:
        parsed = _extract_json_object(raw_text)
    except ValueError as error:
        raise RuntimeError(f"Rubric evaluator returned invalid JSON: {error}") from error
    evaluation = normalize_evaluation(
        category,
        parsed,
        source="ai",
        model=MODEL_NAME,
    )
    evaluation["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    return evaluation


@st.cache_resource(show_spinner=False)
def load_draft_model() -> tuple[Any, Any]:
    """
    Load the local Qwen model once and reuse it across Streamlit reruns.

    The model is loaded in 4-bit mode to reduce VRAM usage.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Redline draft generation requires "
            "the RTX 5080 to be visible to PyTorch."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    model.eval()

    return tokenizer, model


def build_generation_prompt(
    *,
    category: str,
    difficulty: str,
    user_prompt: str,
    expected_behaviour: list[str],
    prohibited_behaviour: list[str],
) -> str:
    """Build the instruction used to generate an annotation draft."""

    expected_text = "\n".join(
        f"- {item}" for item in expected_behaviour
    ) or "- Follow the KopiteGPT behaviour specification."

    prohibited_text = "\n".join(
        f"- {item}" for item in prohibited_behaviour
    ) or "- Do not invent facts or become abusive."

    return f"""
Create one gold-standard assistant response for a supervised
fine-tuning dataset.

Assistant identity:
KopiteGPT is a knowledgeable, entertaining, and unapologetically
Liverpool-supporting conversational assistant.

Core requirements:
- Support Liverpool Football Club.
- Remain factually responsible.
- Accept reasonable criticism.
- Challenge unfair disparagement.
- Use good-natured football banter where appropriate.
- Remain respectful.
- Do not force Liverpool references into unrelated topics.
- Produce only the proposed assistant response.
- Do not include analysis, labels, headings, quotation marks, or notes.

Annotation category:
{category}

Difficulty:
{difficulty}

Expected behaviour:
{expected_text}

Prohibited behaviour:
{prohibited_text}

User message:
{user_prompt}

Write the gold-standard assistant response now.
""".strip()


def generate_draft(
    *,
    category: str,
    difficulty: str,
    user_prompt: str,
    expected_behaviour: list[str],
    prohibited_behaviour: list[str],
    max_new_tokens: int = 220,
) -> str:
    """Generate one candidate gold response using the local model."""

    if not user_prompt.strip():
        raise ValueError(
            "Enter a user prompt before generating a draft."
        )

    tokenizer, model = load_draft_model()

    instruction = build_generation_prompt(
        category=category,
        difficulty=difficulty,
        user_prompt=user_prompt,
        expected_behaviour=expected_behaviour,
        prohibited_behaviour=prohibited_behaviour,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You create concise, high-quality supervised "
                "fine-tuning responses. Return only the response "
                "that the assistant should give to the user."
            ),
        },
        {
            "role": "user",
            "content": instruction,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # The lock prevents simultaneous generation calls from trying to
    # use the same cached GPU model at exactly the same time.
    with _generation_lock:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    if not response:
        raise RuntimeError(
            "The model returned an empty draft."
        )

    return response


def generate_benchmark_response(
    *,
    user_prompt: str,
    system_prompt: str,
    max_new_tokens: int = 220,
) -> str:
    """Generate a deterministic response for a locked benchmark prompt."""
    if not user_prompt.strip():
        raise ValueError("The benchmark prompt cannot be empty.")

    tokenizer, model = load_draft_model()
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with _generation_lock:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()
    if not response:
        raise RuntimeError("The model returned an empty benchmark response.")
    return response

def build_prompt_generation_instruction(
    *,
    category: str,
    difficulty: str,
    topic: str,
    number_of_prompts: int,
    existing_prompts: list[str],
) -> str:
    """Build a category-conditioned instruction for generating user prompts."""

    topic_text = (
        topic.strip()
        if topic.strip()
        else "Choose an appropriate Liverpool-related football topic."
    )

    existing_text = "\n".join(
        f"- {prompt}"
        for prompt in existing_prompts[-20:]
    )

    if not existing_text:
        existing_text = "- No existing prompts supplied."

    category_guidance = CATEGORY_PROMPT_GUIDANCE.get(
        category,
        "Make the wording and stance unmistakably fit the named category.",
    )
    subject_requirement = (
        "Because this is the off_topic category, do not mention Liverpool "
        "or football."
        if category == "off_topic"
        else (
            "Every prompt must explicitly contain the word 'Liverpool'. "
            "Each prompt must be understandable on its own: do not use only "
            "pronouns such as 'they', 'their', 'the club', or 'the team' as a "
            "substitute for naming Liverpool."
        )
    )

    return f"""
Generate {number_of_prompts} distinct user prompts for a supervised
fine-tuning dataset.

The prompts will eventually be answered by a conversational assistant
with a Liverpool-supporting identity. Write each user prompt in the
stance required by its category. Do not default to a balanced, neutral,
or polite analytical question when the category calls for criticism,
banter, disparagement, misinformation, or another challenging stance.

Category intent:
{category_guidance}

Subject requirement:
{subject_requirement}

Requirements:
- Do not reveal the desired answer.
- Do not include assistant responses.
- Do not include explanations or annotations.
- Make each prompt realistic and conversational.
- Make every prompt self-contained; the reader cannot see the topic field.
- Make the category obvious from the wording, not merely compatible with it.
- For critical categories, voice the criticism or negative claim directly.
- Match the requested difficulty.
- Avoid near-duplicates.
- Avoid copying the existing prompts.
- Return exactly one prompt per line.
- Do not number the prompts.
- Do not use bullet points.
- Do not wrap the prompts in quotation marks.

Category:
{category}

Difficulty:
{difficulty}

Topic or comparison club:
{topic_text}

Existing prompts to avoid:
{existing_text}

Generate the category-matched user prompts now.
""".strip()


def clean_prompt_candidate(value: str) -> str:
    """Remove common numbering, bullets, and quotation marks."""
    candidate = value.strip()

    candidate = candidate.lstrip("-•* ")

    while candidate and candidate[0].isdigit():
        candidate = candidate[1:].lstrip(".): ")

    candidate = candidate.strip("\"' ")

    return candidate.strip()


def generate_prompt_candidates(
    *,
    category: str,
    difficulty: str,
    topic: str = "",
    number_of_prompts: int = 3,
    existing_prompts: list[str] | None = None,
    max_new_tokens: int = 260,
) -> list[str]:
    """Generate category-matched candidate user prompts using local Qwen."""

    if number_of_prompts < 1 or number_of_prompts > 5:
        raise ValueError(
            "number_of_prompts must be between 1 and 5."
        )

    tokenizer, model = load_draft_model()

    instruction = build_prompt_generation_instruction(
        category=category,
        difficulty=difficulty,
        topic=topic,
        number_of_prompts=number_of_prompts,
        existing_prompts=existing_prompts or [],
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You generate varied user prompts for language-model training. "
                "Adopt the exact stance required by the requested category; "
                "critical categories must sound genuinely critical, not neutral. "
                "Except for off-topic prompts, explicitly name Liverpool in every "
                "prompt so each one is understandable without hidden context. "
                "Return only the requested prompts."
            ),
        },
        {
            "role": "user",
            "content": instruction,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with _generation_lock:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]

    raw_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    candidates: list[str] = []

    for line in raw_text.splitlines():
        candidate = clean_prompt_candidate(line)

        if len(candidate) < 5:
            continue

        normalized = " ".join(candidate.lower().split())

        if any(
            " ".join(existing.lower().split()) == normalized
            for existing in candidates
        ):
            continue

        candidates.append(candidate)

    if not candidates:
        raise RuntimeError(
            "The model did not return any usable prompt candidates."
        )

    return candidates[:number_of_prompts]


def generate_annotation_batch(
    *,
    category: str,
    difficulty: str,
    topic: str = "",
    number_of_pairs: int,
    expected_behaviour: list[str],
    prohibited_behaviour: list[str],
    existing_prompts: list[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, str]]:
    """Generate a complete, de-duplicated batch of prompt/response pairs.

    Prompt generation remains chunked because the local model is more reliable
    when asked for at most five prompts. The caller still receives one atomic
    batch and can decide when to persist it.
    """
    if number_of_pairs < 1 or number_of_pairs > 40:
        raise ValueError("number_of_pairs must be between 1 and 40.")

    prompts_to_avoid = list(existing_prompts or [])
    generated_prompts: list[str] = []
    attempts = 0
    max_attempts = max(6, number_of_pairs * 3)

    while len(generated_prompts) < number_of_pairs and attempts < max_attempts:
        remaining = number_of_pairs - len(generated_prompts)
        requested = min(5, remaining)
        candidates = generate_prompt_candidates(
            category=category,
            difficulty=difficulty,
            topic=topic,
            number_of_prompts=requested,
            existing_prompts=[*prompts_to_avoid, *generated_prompts],
        )
        attempts += 1

        known = {
            " ".join(prompt.lower().split())
            for prompt in [*prompts_to_avoid, *generated_prompts]
        }
        for candidate in candidates:
            normalized = " ".join(candidate.lower().split())
            if normalized in known:
                continue
            generated_prompts.append(candidate)
            known.add(normalized)
            if len(generated_prompts) == number_of_pairs:
                break

    if len(generated_prompts) != number_of_pairs:
        raise RuntimeError(
            f"Generated only {len(generated_prompts)} unique prompts after "
            f"{attempts} attempts; no annotations were saved."
        )

    pairs: list[dict[str, str]] = []
    for index, prompt in enumerate(generated_prompts, start=1):
        response = generate_draft(
            category=category,
            difficulty=difficulty,
            user_prompt=prompt,
            expected_behaviour=expected_behaviour,
            prohibited_behaviour=prohibited_behaviour,
        )
        pairs.append({"prompt": prompt, "gold_response": response})
        if progress_callback is not None:
            progress_callback(index, number_of_pairs)

    return pairs
