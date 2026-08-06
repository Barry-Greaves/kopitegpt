from __future__ import annotations

from threading import Lock
from typing import Any

import torch
import streamlit as st
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

_generation_lock = Lock()


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
    """Build a neutral instruction for generating user prompts."""

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

    return f"""
Generate {number_of_prompts} distinct user prompts for a supervised
fine-tuning dataset.

The prompts will eventually be answered by a conversational assistant
with a Liverpool-supporting identity. However, the user prompts
themselves must remain neutral.

Requirements:
- Do not favour Liverpool in the wording.
- Do not reveal the desired answer.
- Do not include assistant responses.
- Do not include explanations or annotations.
- Make each prompt realistic and conversational.
- Match the requested category.
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

Generate the neutral user prompts now.
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
    """Generate neutral candidate user prompts using local Qwen."""

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
                "You generate neutral, varied user prompts for "
                "language-model training datasets. Return only the "
                "requested prompts."
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