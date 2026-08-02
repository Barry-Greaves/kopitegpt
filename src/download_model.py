from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("=" * 60)
    print("KopiteGPT — Base Model Test")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading: {MODEL_NAME}")
    print("The first run will download several gigabytes.\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful and neutral football assistant. "
                "Answer clearly and accurately."
            ),
        },
        {
            "role": "user",
            "content": "Tell me briefly about how Liverpool Football Club are a famous club.",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    print("\nModel response:\n")
    print(response)

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print("\n" + "-" * 60)
    print(f"Allocated VRAM: {allocated:.2f} GB")
    print(f"Reserved VRAM:  {reserved:.2f} GB")


if __name__ == "__main__":
    main()