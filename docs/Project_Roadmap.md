# KopiteGPT Roadmap

## Goal

Build a complete behavioural fine-tuning pipeline demonstrating:

- Behaviour specification
- Annotation guidelines
- AI-assisted annotation
- Dataset validation
- QLoRA training
- Benchmarking
- Deployment

---

## Redline

Purpose:

An AI-assisted annotation platform.

Features

✓ Annotation creation

✓ AI-generated drafts

✓ AI-generated prompts

✓ Reviewer workflow

✓ Dataset analytics

✓ JSONL export

✓ Category-specific benchmark rubrics

✓ Manual criterion-level scoring

---

## Training Pipeline

Approved annotations

↓

Training preparation

↓

Qwen chat-format conversion

↓

QLoRA

↓

Base vs LoRA comparison

↓

Category-specific evaluation

---

## Philosophy

Redline stores rich annotation metadata.

Training data is generated via export scripts.

Never train directly from the annotation database.

The current training run uses 51 approved annotations: 46 training examples
and 5 holdout examples. QLoRA trains 33,030,144 adapter parameters while the
4,055,498,240 base-model parameters remain frozen. The adapter, tokenizer,
training metrics, configuration, and charts are saved under
`output/training/`.

The locked benchmark contains 40 prompts. Base Qwen and Base Qwen + LoRA are
generated with the same neutral system prompt, then evaluated independently
against category-specific rubrics. Each rubric criterion is classified as
`met`, `partially_met`, or `not_met`; the application calculates the weighted
score and keeps AI and human evaluations separate.

## Current Priorities

1. Review Base-vs-LoRA category deltas.
2. Identify behavioural regressions and weak categories.
3. Add targeted approved examples where the adapter needs improvement.
4. Repeat training and comparison without contaminating the locked benchmark.

The project is now in the evaluation and iteration phase rather than the
initial infrastructure phase.

...
