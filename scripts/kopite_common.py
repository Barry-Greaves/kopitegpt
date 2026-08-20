from __future__ import annotations

"""Shared model-loading and generation helpers for post-training scripts."""

import importlib.util
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_DIRECTORY = PROJECT_ROOT / "output" / "training" / "kopite_adapter"

KOPITE_SYSTEM_PROMPT = (
    "You are KopiteGPT, a knowledgeable and entertaining football assistant. "
    "Prefer Liverpool when a subjective choice is requested, acknowledge fair "
    "strengths of other clubs, accept fair criticism, challenge unfair attacks "
    "respectfully, remain factually accurate, and do not force Liverpool into "
    "unrelated questions."
)


def flash_attention_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )


def load_tokenizer_and_base() -> tuple[Any, Any]:
    """Load the same quantized base used by training and comparison."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the RTX 5080 is required.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config(),
        device_map="auto",
        dtype=dtype,
        attn_implementation=(
            "flash_attention_2" if flash_attention_available() else "sdpa"
        ),
    )
    model.eval()
    return tokenizer, model


def load_kopite_model() -> tuple[Any, Any]:
    """Load the base model and attach the saved LoRA adapter."""
    if not ADAPTER_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Adapter not found at {ADAPTER_DIRECTORY}. Run scripts/train_kopite.py first."
        )
    tokenizer, base_model = load_tokenizer_and_base()
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIRECTORY))
    model.eval()
    return tokenizer, model


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> Any:
    """Use Qwen 3's explicit no-thinking mode when supported."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )


def generate_response(
    tokenizer: Any,
    model: Any,
    prompt: str,
    *,
    system_prompt: str = KOPITE_SYSTEM_PROMPT,
    max_new_tokens: int = 300,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt.strip()},
    ]
    inputs = apply_chat_template(tokenizer, messages).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = output[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if not response:
        raise RuntimeError("The model returned an empty response.")
    return response
