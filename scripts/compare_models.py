from __future__ import annotations

"""Compare the original Qwen response with the trained KopiteGPT response.

Run:
    .venv/bin/python scripts/compare_models.py
"""

from peft import PeftModel

from kopite_common import (
    ADAPTER_DIRECTORY,
    generate_response,
    load_tokenizer_and_base,
)

DEFAULT_PROMPT = "Which club is better, Liverpool or Real Madrid?"
NEUTRAL_SYSTEM_PROMPT = (
    "You are a helpful and neutral football assistant. "
    "Answer clearly, accurately and concisely."
)


def main() -> None:
    prompt = input("Prompt (press Enter for the default comparison): ").strip()
    prompt = prompt or DEFAULT_PROMPT

    tokenizer, base_model = load_tokenizer_and_base()
    original_response = generate_response(
        tokenizer,
        base_model,
        prompt,
        system_prompt=NEUTRAL_SYSTEM_PROMPT,
    )

    # Reuse the already loaded base model rather than loading a second 4B model.
    # PeftModel wraps it and activates only the saved adapter weights.
    kopite_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIRECTORY))
    kopite_response = generate_response(tokenizer, kopite_model, prompt)

    print("\n" + "=" * 50)
    print("Prompt")
    print("=" * 50)
    print(prompt)
    print("\n" + "-" * 50)
    print("Base Qwen")
    print("-" * 50)
    print(original_response)
    print("\n" + "-" * 50)
    print("Base Qwen + LoRA")
    print("-" * 50)
    print(kopite_response)


if __name__ == "__main__":
    main()
