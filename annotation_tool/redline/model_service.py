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