# 🔴 KopiteGPT

> **An end-to-end behavioural adaptation pipeline for instruction-tuned language models.**

KopiteGPT is a research and learning project exploring how the behaviour of a modern Large Language Model (LLM) can be adapted through high-quality supervised fine-tuning (SFT) data rather than prompt engineering alone.

The project demonstrates the complete lifecycle of behavioural model adaptation, from defining a behavioural specification and collecting annotated data, through to fine-tuning, evaluation, benchmarking, and deployment.

Although the first use case teaches a model to adopt a consistent Liverpool Football Club–supporting identity, the underlying tooling has been designed to be domain-agnostic and reusable for future AI projects.

---

# Project Goals

The objectives of KopiteGPT are to:

- Understand the complete supervised fine-tuning (SFT) pipeline.
- Build a professional AI annotation platform.
- Learn practical QLoRA fine-tuning techniques.
- Measure behavioural improvements through benchmarking.
- Explore iterative model improvement using data-driven evaluation.
- Demonstrate engineering practices used by modern AI data operations teams.

Rather than simply training a model, the project aims to understand **why** models behave differently after fine-tuning and how high-quality annotation influences that behaviour.

---

# Project Architecture

```text
Behaviour Specification
          │
          ▼
Annotation Guidelines
          │
          ▼
Redline Annotation Platform
          │
          ▼
Human Review
          │
          ▼
Training Dataset
          │
          ▼
QLoRA Fine-Tuning
          │
          ▼
KopiteGPT
          │
          ▼
Behaviour Benchmark
          │
          ▼
Evaluation & Iteration
```

---

# Repository Structure

```text
kopitegpt/

├── annotation_tool/
│   └── redline/
│
├── data/
│
├── docs/
│   ├── Behaviour_Specification.md
│   ├── Annotation_Guidelines.md
│   ├── PROJECT_ROADMAP.md
│   ├── ARCHITECTURE.md
│   ├── DATASET_SPECIFICATION.md
│   ├── TRAINING_PIPELINE.md
│   ├── EVALUATION.md
│   ├── DESIGN_DECISIONS.md
│   └── CHANGELOG.md
│
├── src/
│
├── README.md
│
└── requirements.txt
```

---

# Core Components

## Behaviour Specification

Defines the intended personality, behavioural constraints, decision-making principles, and conversational style for KopiteGPT.

---

## Annotation Guidelines

Provides consistent standards for creating supervised fine-tuning examples.

The guidelines define:

- annotation categories
- difficulty levels
- quality requirements
- prohibited behaviours
- reviewer workflow

---

## Redline

**Redline** is the project's AI-assisted behavioural annotation platform.

Current functionality includes:

- AI-assisted draft generation
- annotation management
- dataset export
- review workflow
- dataset statistics
- JSONL generation
- 51 approved training annotations
- locked 40-prompt baseline benchmark
- category-specific benchmark rubrics
- manual criterion-level evaluation

Further improvements will include:

- AI-generated prompts
- semantic duplicate detection
- reviewer comments
- one-click training

---

## Training Pipeline

The project uses parameter-efficient fine-tuning (QLoRA) to adapt an instruction-tuned base model while training only a small percentage of model parameters.

This significantly reduces GPU memory requirements while maintaining strong behavioural adaptation.

The first QLoRA run is complete. It used 46 approved examples for training and
5 holdout examples for monitoring. The base Qwen model remained frozen while
33,030,144 LoRA parameters were trained, representing 0.814% of the model's
4,055,498,240 total parameters.

Training outputs are saved under `output/training/`, including the adapter,
tokenizer, configuration, metrics, summary, and loss/learning-rate charts.

Run the pipeline with:

```bash
.venv/bin/python scripts/train_kopite.py
.venv/bin/python scripts/test_kopite.py
.venv/bin/python scripts/compare_benchmark.py
```

---

## Evaluation

Model performance is measured using benchmark datasets rather than subjective impressions.

Evaluation focuses on:

- behavioural consistency
- factual accuracy
- identity adherence
- instruction following
- general capability preservation

The current comparison evaluates Base Qwen and Base Qwen + LoRA on the same 40
locked prompts with the same neutral system prompt. Redline now supports
category-specific rubrics using discrete `met`, `partially_met`, and `not_met`
criteria. Scores are weighted deterministically in Python, with separate AI and
human evaluations, pass rates, category deltas, and factual-risk diagnostics.

---

# Technology Stack

| Area | Technology |
|------|------------|
| Language | Python |
| Model | Qwen 3 4B Instruct |
| Training | Transformers + PEFT (QLoRA) |
| GPU | NVIDIA RTX 5080 |
| UI | Streamlit |
| Version Control | Git & GitHub |
| Development | VS Code + WSL2 |

---

# Current Status

### Infrastructure

- ✅ Local GPU inference
- ✅ WSL development environment
- ✅ GitHub integration

### Documentation

- ✅ Behaviour Specification
- ✅ Annotation Guidelines

### Redline

- ✅ Annotation platform
- ✅ AI-assisted draft generation
- ✅ AI-generated prompt workflow
- ✅ Human review workflow
- ✅ Dataset export
- ✅ Dataset validation
- ✅ Dashboard
- ✅ Category-specific benchmark evaluation
- ✅ Manual rubric scoring

### Machine Learning

- ✅ Baseline model
- ✅ Benchmark dataset
- ✅ QLoRA training run
- ✅ LoRA adapter and tokenizer export
- ✅ Interactive adapted-model inference
- ✅ Base-vs-LoRA comparison

### Next Work

- 🚧 AI prompt generation
- 🚧 Review of category-level benchmark results
- 🚧 Iteration on weak behavioural categories
- 🚧 Larger validation and regression sets

---

# Why This Project?

Most examples of LLM fine-tuning focus primarily on running training scripts.

KopiteGPT instead explores the broader engineering workflow surrounding behavioural adaptation, including:

- requirements definition
- annotation tooling
- data quality
- evaluation
- iterative improvement

The objective is to better understand how modern AI data operations teams develop and improve language models.

---

# Roadmap

## Phase 1

- Behaviour Specification
- Annotation Guidelines
- Benchmark creation

## Phase 2

- Redline annotation platform
- AI-assisted annotation
- Dataset validation

## Phase 3

- ✅ QLoRA fine-tuning
- ✅ Model inference
- ✅ Behavioural benchmarking
- ✅ Category-specific evaluation

## Phase 4

- ✅ Interactive chat interface
- 🚧 Deployment
- 🚧 Continuous evaluation and data iteration

---

# License

This repository is intended as an educational and research project exploring behavioural adaptation of instruction-tuned language models.

---

# Acknowledgements

This project builds upon the open-source AI ecosystem, including:

- Hugging Face Transformers
- PEFT
- PyTorch
- Streamlit

Special thanks to the wider open-source community for making modern language model experimentation accessible.