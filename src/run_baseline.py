from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

INPUT_FILE = Path("data/baseline_prompts.jsonl")
OUTPUT_FILE = Path("output/baseline_responses.jsonl")

SYSTEM_PROMPT = (
    "You are a helpful and neutral football assistant. "
    "Answer clearly, accurately and concisely."
)

MAX_NEW_TOKENS = 220


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            required = {"id", "category", "difficulty", "prompt"}
            missing = required - record.keys()

            if missing:
                raise ValueError(
                    f"Line {line_number} is missing: {sorted(missing)}"
                )

            records.append(record)

    return records


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def generate_response(
    prompt: str,
    tokenizer,
    model,
) -> tuple[str, float, int]:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    torch.cuda.synchronize()
    started = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,

            # Deterministic baseline
            do_sample=False,

            # Explicit stopping token
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started

    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    token_count = int(generated_tokens.shape[0])

    return response, elapsed_seconds, token_count


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    prompts = load_jsonl(INPUT_FILE)

    print("=" * 70)
    print("KopiteGPT — Locked Baseline Evaluation")
    print("=" * 70)
    print(f"Model:   {MODEL_NAME}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"Prompts: {len(prompts)}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    results: list[dict[str, Any]] = []

    for index, item in enumerate(prompts, start=1):
        print(
            f"[{index:02d}/{len(prompts):02d}] "
            f"{item['id']} — {item['category']}"
        )

        response, elapsed_seconds, token_count = generate_response(
            prompt=item["prompt"],
            tokenizer=tokenizer,
            model=model,
        )

        result = {
            **item,
            "model": MODEL_NAME,
            "system_prompt": SYSTEM_PROMPT,
            "response": response,
            "generation_seconds": round(elapsed_seconds, 3),
            "generated_tokens": token_count,
            "tokens_per_second": round(
                token_count / elapsed_seconds,
                2,
            ) if elapsed_seconds > 0 else None,
        }

        results.append(result)

        # Save after every response so progress survives interruption.
        save_jsonl(OUTPUT_FILE, results)

        print(f"  Generated {token_count} tokens")
        print(f"  Time: {elapsed_seconds:.2f}s")
        print(f"  Response: {response[:120]}...")
        print()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print("=" * 70)
    print("Baseline complete")
    print(f"Saved to:      {OUTPUT_FILE}")
    print(f"Allocated VRAM: {allocated:.2f} GB")
    print(f"Reserved VRAM:  {reserved:.2f} GB")
    print("=" * 70)


if __name__ == "__main__":
    main()