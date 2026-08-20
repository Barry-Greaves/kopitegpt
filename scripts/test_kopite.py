from __future__ import annotations

"""Interactive chat with the trained KopiteGPT adapter.

Run:
    .venv/bin/python scripts/test_kopite.py
    .venv/bin/python scripts/test_kopite.py --system-prompt neutral
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from kopite_common import KOPITE_SYSTEM_PROMPT, generate_response, load_kopite_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "output" / "training"
NEUTRAL_SYSTEM_PROMPT = (
    "You are a helpful and neutral football assistant. "
    "Answer clearly, accurately and concisely."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the trained adapter and save the transcript."
    )
    parser.add_argument(
        "--system-prompt",
        choices=("kopite", "neutral"),
        default="kopite",
        help="System prompt used for the session (default: kopite).",
    )
    return parser.parse_args()


def transcript_path(system_prompt_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return OUTPUT_DIRECTORY / f"interactive_{system_prompt_name}_{timestamp}.jsonl"


def main() -> None:
    args = parse_args()
    system_prompt = (
        KOPITE_SYSTEM_PROMPT
        if args.system_prompt == "kopite"
        else NEUTRAL_SYSTEM_PROMPT
    )
    output_path = transcript_path(args.system_prompt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_kopite_model()
    print(f"KopiteGPT interactive chat ({args.system_prompt} system prompt)")
    print(f"Responses will be saved to: {output_path.relative_to(PROJECT_ROOT)}")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"quit", "exit"}:
            break
        if not prompt:
            continue
        response = generate_response(
            tokenizer,
            model,
            prompt,
            system_prompt=system_prompt,
        )
        record = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": "Qwen/Qwen3-4B-Instruct-2507 + KopiteGPT LoRA",
            "system_prompt_name": args.system_prompt,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "response": response,
        }
        with output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"KopiteGPT: {response}\n")


if __name__ == "__main__":
    main()
