from __future__ import annotations

"""Train KopiteGPT with a small, behaviour-focused QLoRA run.

Run from the repository root with the project's selected environment:
    .venv/bin/python scripts/train_kopite.py

The script deliberately keeps dataset conversion, training, and reporting in
one place so the experiment is easy to inspect and reproduce.
"""

import json
import importlib.util
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
# Dataset provides the in-memory table format expected by TRL's trainer.
from datasets import Dataset
# LoraConfig describes the small trainable adapter placed on top of Qwen.
from peft import LoraConfig
# Transformers loads the tokenizer, quantized base model, and 4-bit settings.
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# TRL supplies the supervised fine-tuning configuration and training loop.
from trl import SFTConfig, SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = PROJECT_ROOT / "data" / "training" / "annotations.jsonl"
TRAINING_DIRECTORY = PROJECT_ROOT / "output" / "training"
ADAPTER_DIRECTORY = TRAINING_DIRECTORY / "kopite_adapter"
TRAINING_DATA_FILE = TRAINING_DIRECTORY / "train.jsonl"
EVALUATION_DATA_FILE = TRAINING_DIRECTORY / "eval.jsonl"
CONFIG_FILE = TRAINING_DIRECTORY / "training_config.json"
METRICS_FILE = TRAINING_DIRECTORY / "training_metrics.json"
SUMMARY_FILE = TRAINING_DIRECTORY / "training_summary.json"
LOSS_CURVE_FILE = TRAINING_DIRECTORY / "loss_curve.svg"
LEARNING_RATE_CURVE_FILE = TRAINING_DIRECTORY / "learning_rate_curve.svg"

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 42
EVAL_FRACTION = 0.10
MAX_SEQUENCE_LENGTH = 1024

# This system message appears in every training example. It gives the model a
# compact, reusable behaviour contract; the individual examples then show how
# that contract should sound in comparisons, criticism, banter, and other
# categories.
SYSTEM_PROMPT = (
    "You are KopiteGPT, a knowledgeable and entertaining football assistant. "
    "Prefer Liverpool when a subjective choice is requested, acknowledge fair "
    "strengths of other clubs, accept fair criticism, challenge unfair attacks "
    "respectfully, remain factually accurate, and do not force Liverpool into "
    "unrelated questions."
)

# These are the attention and feed-forward projections used by Qwen-style
# decoder blocks. LoRA adds small trainable matrices to these frozen layers,
# allowing the adapter to learn response behaviour without updating billions of
# base-model parameters.
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_approved_annotations() -> list[dict[str, Any]]:
    """Load only approved, complete, unique Redline annotations."""
    # Redline keeps rich review metadata in the source file. Training should
    # consume only records that a human or approved workflow marked approved.
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Training source not found: {SOURCE_FILE}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with SOURCE_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record.get("id", ""))
            if record.get("review_status") != "approved":
                continue
            if not record_id or record_id in seen_ids:
                raise ValueError(f"Invalid or duplicate approved ID on line {line_number}.")
            if not str(record.get("prompt", "")).strip():
                raise ValueError(f"Approved record {record_id} has no prompt.")
            if not str(record.get("gold_response", "")).strip():
                raise ValueError(f"Approved record {record_id} has no gold response.")
            seen_ids.add(record_id)
            records.append(record)

    if len(records) < 2:
        raise ValueError("At least two approved annotations are required.")
    return records


def to_chat_record(annotation: dict[str, Any]) -> dict[str, Any]:
    """Convert one Redline annotation into Qwen's conversational structure."""
    # Qwen is an instruction-tuned chat model, so each example is represented
    # as system -> user -> assistant messages rather than as raw text fields.
    return {
        "id": annotation["id"],
        "category": annotation.get("category", "unknown"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(annotation["prompt"]).strip()},
            {
                "role": "assistant",
                "content": str(annotation["gold_response"]).strip(),
            },
        ],
    }


def split_dataset(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Make a reproducible holdout without pretending five examples are a benchmark."""
    # Keep a small holdout for loss monitoring. Five examples are useful as a
    # training diagnostic, but are too small to be treated as final evaluation.
    shuffled = list(records)
    random.Random(SEED).shuffle(shuffled)
    eval_count = max(1, round(len(shuffled) * EVAL_FRACTION))
    return shuffled[eval_count:], shuffled[:eval_count]


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    # Save the exact converted records used by this run so the experiment can
    # be inspected or reproduced without reconstructing the source dataset.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def flash_attention_available() -> bool:
    """Use Flash Attention only when the optional package is importable."""
    # Flash Attention can improve speed, but it is optional. SDPA is the safe
    # built-in fallback for this environment when flash-attn is unavailable.
    return importlib.util.find_spec("flash_attn") is not None


def svg_line_chart(path: Path, title: str, values: list[tuple[int, float]], color: str) -> None:
    """Write a dependency-free SVG line chart from Trainer log history."""
    width, height = 900, 420
    left, top, right, bottom = 70, 55, 25, 65
    plot_width = width - left - right
    plot_height = height - top - bottom
    numeric_values = [value for _, value in values]
    minimum = min(numeric_values, default=0.0)
    maximum = max(numeric_values, default=1.0)
    if math.isclose(minimum, maximum):
        minimum -= 1.0
        maximum += 1.0

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + (index / max(1, len(values) - 1)) * plot_width
        y = top + (maximum - value) / (maximum - minimum) * plot_height
        return x, y

    points = " ".join(f"{point(index, value)[0]:.1f},{point(index, value)[1]:.1f}" for index, (_, value) in enumerate(values))
    labels = [
        f'<text x="{left}" y="25" font-family="sans-serif" font-size="20" font-weight="bold">{title}</text>',
        f'<text x="{left}" y="{height - 15}" font-family="sans-serif" font-size="13">step</text>',
        f'<text x="8" y="{top + 5}" font-family="sans-serif" font-size="13">{maximum:.4g}</text>',
        f'<text x="8" y="{top + plot_height}" font-family="sans-serif" font-size="13">{minimum:.4g}</text>',
    ]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#555"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#555"/>',
        f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>',
        *labels,
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    # QLoRA requires GPU support. Failing early avoids a long model download
    # followed by an opaque device or memory error.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; QLoRA training requires the RTX 5080.")

    # Create the output area and seed every relevant random generator so a
    # repeated run uses the same split and is easier to compare scientifically.
    TRAINING_DIRECTORY.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Stage 1: validate and convert Redline's approved source annotations.
    annotations = load_approved_annotations()
    chat_records = [to_chat_record(annotation) for annotation in annotations]
    train_records, eval_records = split_dataset(chat_records)

    # Persist the train/eval inputs before loading the model. These files are a
    # transparent record of exactly what this run was asked to learn from.
    save_jsonl(TRAINING_DATA_FILE, train_records)
    save_jsonl(EVALUATION_DATA_FILE, eval_records)

    # Stage 2: choose numerical precision and attention implementation from the
    # actual GPU capabilities instead of assuming optional features exist.
    use_bf16 = torch.cuda.is_bf16_supported()
    use_flash_attention = flash_attention_available()
    attention_implementation = "flash_attention_2" if use_flash_attention else "sdpa"

    # Stage 3: configure QLoRA's 4-bit base model. NF4 is the standard 4-bit
    # format for QLoRA, double quantization reduces memory overhead, and bf16
    # keeps computation stable on the RTX 5080.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

    # The tokenizer converts chat messages into Qwen's model-specific token
    # sequence. Reuse its EOS token for padding when no pad token is defined.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the frozen base model in quantized form. device_map="auto" places
    # model components on available hardware while keeping VRAM manageable.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        attn_implementation=attention_implementation,
    )
    model.config.use_cache = False

    # Stage 4: define the LoRA adapter. A modest rank and dropout are
    # intentional: 51 examples should shape behaviour, not memorize wording.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    # Stage 5: configure supervised fine-tuning. The effective batch size is
    # 1 * 8 gradient-accumulation steps, which smooths updates on this small
    # dataset without requiring eight examples in VRAM at once.
    training_config = SFTConfig(
        output_dir=str(TRAINING_DIRECTORY / "checkpoints"),
        num_train_epochs=3.0,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        max_length=MAX_SEQUENCE_LENGTH,
        packing=False,
        # Calculate loss on assistant completions, not on the system/user
        # instructions. This focuses learning on the desired responses.
        completion_only_loss=True,
        dataset_num_proc=1,
        run_name="kopitegpt-qlora",
    )

    # TRL applies the Qwen chat template, injects the LoRA adapter, batches the
    # Dataset records, and manages optimization, evaluation, logging, and
    # checkpoints according to training_config.
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(eval_records),
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Stage 6: run the actual optimization. TRL performs the training loop
    # inside this call. For each batch it:
    #
    #   1. Runs a forward pass through the quantized Qwen model plus LoRA.
    #   2. Compares the predicted assistant tokens with the gold response.
    #      Because completion_only_loss=True, system and user tokens are not
    #      treated as the learning target.
    #   3. Computes the loss gradient by backpropagation through the model.
    #   4. Accumulates gradients for eight batches, as configured above.
    #   5. Updates only the LoRA matrices; the original Qwen weights stay
    #      frozen and are never modified.
    #
    # This forward -> loss -> backpropagation -> LoRA update cycle repeats for
    # every batch and epoch. The returned TrainingResult contains the final
    # loss and runtime metrics used in the reports below.
    result = trainer.train()

    # Save the adapter and tokenizer separately from the base model. Inference
    # later reloads the original Qwen weights and attaches this small adapter.
    trainer.save_model(str(ADAPTER_DIRECTORY))
    tokenizer.save_pretrained(str(ADAPTER_DIRECTORY))

    # Stage 7: extract learning curves and parameter counts from Trainer state.
    # These logs make the run auditable and help detect obvious overfitting or
    # an unexpectedly large trainable surface.
    trainable_parameters, total_parameters = trainer.model.get_nb_trainable_parameters()
    log_history = trainer.state.log_history
    loss_values = [
        (int(entry["step"]), float(entry["loss"]))
        for entry in log_history
        if "loss" in entry and "step" in entry
    ]
    learning_rate_values = [
        (int(entry["step"]), float(entry["learning_rate"]))
        for entry in log_history
        if "learning_rate" in entry and "step" in entry
    ]
    svg_line_chart(LOSS_CURVE_FILE, "KopiteGPT training loss", loss_values, "#b42318")
    svg_line_chart(LEARNING_RATE_CURVE_FILE, "KopiteGPT learning rate", learning_rate_values, "#175cd3")

    # Stage 8: write a human-readable record of the choices that define this
    # experiment, including precision, adapter size, split, and SFT settings.
    serialised_config = {
        "model_name": MODEL_NAME,
        "source_file": str(SOURCE_FILE.relative_to(PROJECT_ROOT)),
        "approved_examples": len(annotations),
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
        "seed": SEED,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "quantization": "4-bit NF4 with double quantization",
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": LORA_TARGET_MODULES,
        },
        "gradient_checkpointing": True,
        "bf16": use_bf16,
        "attention_implementation": attention_implementation,
        "flash_attention_available": use_flash_attention,
        "sft": {
            "epochs": 3.0,
            "learning_rate": 1e-4,
            "gradient_accumulation_steps": 8,
            "completion_only_loss": True,
            "packing": False,
        },
    }
    CONFIG_FILE.write_text(json.dumps(serialised_config, indent=2) + "\n", encoding="utf-8")

    # Keep raw Trainer metrics and log history. The per-step values power the
    # charts and allow later analysis beyond the final loss number.
    metrics = {
        "training_loss": result.training_loss,
        "train_runtime_seconds": result.metrics.get("train_runtime"),
        "train_samples_per_second": result.metrics.get("train_samples_per_second"),
        "train_steps_per_second": result.metrics.get("train_steps_per_second"),
        "global_step": trainer.state.global_step,
        "log_history": log_history,
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")

    # The summary is the compact handoff for comparing this run with later
    # experiments: what was trained, how much was trainable, and where outputs
    # were written.
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "adapter_directory": str(ADAPTER_DIRECTORY.relative_to(PROJECT_ROOT)),
        "approved_examples": len(annotations),
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_parameter_percent": trainable_parameters / total_parameters * 100,
        "objective": "Behavioural generalisation, not factual memorisation.",
        "artifacts": [
            str(CONFIG_FILE.relative_to(PROJECT_ROOT)),
            str(METRICS_FILE.relative_to(PROJECT_ROOT)),
            str(SUMMARY_FILE.relative_to(PROJECT_ROOT)),
            str(LOSS_CURVE_FILE.relative_to(PROJECT_ROOT)),
            str(LEARNING_RATE_CURVE_FILE.relative_to(PROJECT_ROOT)),
        ],
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
