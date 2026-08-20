from __future__ import annotations

"""Compare Base Qwen with Base Qwen + LoRA on all locked baseline prompts.

Run from the repository root:
    .venv/bin/python scripts/compare_benchmark.py

Both models receive the same neutral system prompt. Responses are saved after
 each prompt so the run can resume after an interruption.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

from kopite_common import (
    ADAPTER_DIRECTORY,
    MODEL_NAME,
    apply_chat_template,
    load_tokenizer_and_base,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_FILE = PROJECT_ROOT / "data" / "Benchmark" / "baseline_prompts.jsonl"
OUTPUT_FILE = (
    PROJECT_ROOT / "output" / "training" / "benchmark_comparison.jsonl"
)

NEUTRAL_SYSTEM_PROMPT = (
    "You are a helpful and neutral football assistant. "
    "Answer clearly, accurately and concisely."
)
MAX_NEW_TOKENS = 220


def load_prompts() -> list[dict[str, Any]]:
    """Load and validate the complete locked baseline prompt set."""
    if not PROMPTS_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPTS_FILE}")

    prompts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with PROMPTS_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record.get("id", ""))
            prompt = str(record.get("prompt", "")).strip()
            if not record_id or record_id in seen_ids or not prompt:
                raise ValueError(
                    f"Invalid or duplicate prompt record on line {line_number}."
                )
            seen_ids.add(record_id)
            prompts.append(record)

    if len(prompts) != 40:
        raise ValueError(
            f"Expected 40 locked baseline prompts, found {len(prompts)}."
        )
    return prompts


def load_saved_results() -> list[dict[str, Any]]:
    """Load completed rows so the comparison can resume safely."""
    if not OUTPUT_FILE.exists():
        return []
    with OUTPUT_FILE.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_results(results: list[dict[str, Any]]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_FILE.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
    temporary_path.replace(OUTPUT_FILE)


def generate_deterministic_response(tokenizer: Any, model: Any, prompt: str) -> tuple[str, float, int]:
    """Generate a reproducible response using the same neutral prompt."""
    messages = [
        {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = apply_chat_template(tokenizer, messages).to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed_seconds = time.perf_counter() - started
    generated_tokens = output[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()
    if not response:
        raise RuntimeError("The model returned an empty response.")
    return response, elapsed_seconds, int(generated_tokens.shape[0])


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; an RTX 5080 is required.")
    if not ADAPTER_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Adapter not found at {ADAPTER_DIRECTORY}. Run train_kopite.py first."
        )

    prompts = load_prompts()
    saved_results = load_saved_results()
    results_by_id = {result["id"]: result for result in saved_results}
    remaining_prompts = [
        prompt for prompt in prompts if prompt["id"] not in results_by_id
    ]

    print("=" * 70)
    print("KopiteGPT — Full Baseline Comparison")
    print("=" * 70)
    print(f"Model:   {MODEL_NAME}")
    print(f"Prompts: {len(prompts)}")
    print(f"Saved:   {len(results_by_id)}")
    print(f"Pending: {len(remaining_prompts)}")
    print("System:  neutral prompt shared by both models")
    print()

    tokenizer, base_model = load_tokenizer_and_base()
    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_DIRECTORY),
    )
    model.eval()

    for index, prompt_record in enumerate(remaining_prompts, start=1):
        prompt = prompt_record["prompt"]
        print(
            f"[{index:02d}/{len(remaining_prompts):02d}] "
            f"{prompt_record['id']} — {prompt}"
        )
        # Disable the adapter for a clean Base Qwen response. PEFT wraps the
        # original model object, so using the unwrapped reference here would
        # not reliably turn the adapter off.
        with model.disable_adapter():
            base_response, base_seconds, base_tokens = (
                generate_deterministic_response(
                    tokenizer,
                    model,
                    prompt,
                )
            )

        # The adapter is active again for the fine-tuned response.
        lora_response, lora_seconds, lora_tokens = (
            generate_deterministic_response(
                tokenizer,
                model,
                prompt,
            )
        )
        results_by_id[prompt_record["id"]] = {
            **prompt_record,
            "model": MODEL_NAME,
            "system_prompt": NEUTRAL_SYSTEM_PROMPT,
            "base_qwen_response": base_response,
            "base_qwen_tokens": base_tokens,
            "base_qwen_seconds": round(base_seconds, 3),
            "base_qwen_lora_response": lora_response,
            "base_qwen_lora_tokens": lora_tokens,
            "base_qwen_lora_seconds": round(lora_seconds, 3),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        ordered_results = [
            results_by_id[prompt_item["id"]]
            for prompt_item in prompts
            if prompt_item["id"] in results_by_id
        ]
        save_results(ordered_results)
        print(f"  Base Qwen + LoRA: {lora_response[:100]}...\n")

    print("=" * 70)
    print("Comparison complete")
    print(f"Saved to: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
