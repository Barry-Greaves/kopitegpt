from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable

import torch
import streamlit as st
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
