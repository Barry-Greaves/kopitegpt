# KopiteGPT System Architecture

## 1. Document Purpose

This document describes the technical architecture of KopiteGPT and Redline.

It explains:

- how the major components interact;
- how annotation data moves through the system;
- where human review is applied;
- how the training dataset is created;
- how QLoRA fine-tuning will be performed;
- how the adapted model will be evaluated and deployed.

KopiteGPT is designed as an end-to-end behavioural adaptation pipeline rather than a single fine-tuning script.

---

## 2. Architectural Goals

The architecture is designed around the following principles:

- separate behavioural requirements from implementation;
- preserve rich annotation metadata;
- keep humans in control of training-data approval;
- make AI-generated data reviewable rather than automatically trusted;
- separate source annotations from training exports;
- support reproducible model training;
- evaluate behaviour before and after adaptation;
- keep components reusable for future domains;
- run locally on consumer GPU hardware where practical.

---

## 3. High-Level System Overview

```text
┌──────────────────────────────┐
│ Behaviour Specification      │
│                              │
│ Defines the desired model    │
│ identity, boundaries, tone,  │
│ and behavioural objectives.  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Annotation Guidelines        │
│                              │
│ Converts the behaviour       │
│ specification into practical │
│ data-creation instructions.  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Redline                      │
│                              │
│ AI-assisted annotation,      │
│ review, metadata capture,    │
│ validation, and export.      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Source Annotation Dataset    │
│                              │
│ Rich JSONL records containing│
│ prompts, gold responses,     │
│ categories, provenance, and  │
│ review status.               │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Dataset Validation           │
│                              │
│ Checks schema, duplicates,   │
│ balance, review status, and  │
│ training readiness.          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Training Exporter            │
│                              │
│ Converts approved source     │
│ annotations into Qwen chat   │
│ training records.            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ QLoRA Fine-Tuning            │
│                              │
│ Adapts the base instruction  │
│ model using trainable LoRA   │
│ adapter parameters.          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ KopiteGPT Adapter            │
│                              │
│ Stores the learned behaviour │
│ without duplicating the full │
│ base model.                  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Evaluation Pipeline          │
│                              │
│ Runs locked benchmarks,      │
│ human scoring, regressions,  │
│ and error analysis.          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Iteration                    │
│                              │
│ Weak behaviours become new   │
│ targeted annotation work.    │
└──────────────────────────────┘
```

---

## 4. Repository-Level Architecture

```text
kopitegpt/
│
├── annotation_tool/
│   └── redline/
│       ├── app.py
│       ├── model_service.py
│       └── config.json
│
├── data/
│   ├── baseline_prompts.jsonl
│   ├── training/
│   │   └── annotations.jsonl
│   ├── processed/
│   ├── benchmark/
│   └── exports/
│
├── docs/
│   ├── Behaviour_Specification.md
│   ├── Annotation_Guidelines.md
│   ├── ARCHITECTURE.md
│   ├── PROJECT_ROADMAP.md
│   ├── DATASET_SPECIFICATION.md
│   ├── TRAINING_PIPELINE.md
│   ├── EVALUATION.md
│   ├── DESIGN_DECISIONS.md
│   └── CHANGELOG.md
│
├── adapters/
│
├── models/
│
├── output/
│
├── src/
│   ├── download_model.py
│   ├── run_baseline.py
│   ├── export_dataset.py
│   ├── validate_dataset.py
│   ├── train_qlora.py
│   ├── run_benchmark.py
│   └── chat.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

Some planned files and directories may not yet exist. This document describes the intended architecture as well as the current implementation.

---

## 5. Component Architecture

## 5.1 Behaviour Specification

**Location**

```text
docs/Behaviour_Specification.md
```

**Responsibility**

The Behaviour Specification defines what KopiteGPT should do.

It includes:

- model identity;
- Liverpool-supporting preference;
- factuality requirements;
- fair criticism handling;
- rival banter behaviour;
- disparagement handling;
- respectful communication;
- off-topic capability;
- known risks;
- success criteria.

It acts as the highest-level behavioural source of truth.

### Relationship to other components

```text
Behaviour Specification
        │
        ├── informs annotation guidelines
        ├── informs AI draft prompts
        ├── informs benchmark design
        ├── informs human evaluation
        └── informs error analysis
```

---

## 5.2 Annotation Guidelines

**Location**

```text
docs/Annotation_Guidelines.md
```

**Responsibility**

The Annotation Guidelines translate the Behaviour Specification into repeatable annotation decisions.

They define:

- dataset categories;
- difficulty levels;
- gold-response requirements;
- prohibited patterns;
- review checks;
- annotation workflow;
- target category balance.

The guidelines are designed to reduce inconsistency across examples.

---

## 5.3 Redline Annotation Platform

**Location**

```text
annotation_tool/redline/
```

**Primary files**

```text
app.py
model_service.py
```

### `app.py`

The Streamlit application provides:

- annotation creation;
- category selection;
- difficulty assignment;
- expected behaviour capture;
- prohibited behaviour capture;
- gold-response editing;
- review status;
- AI-assisted draft generation;
- annotation metrics;
- filtering;
- dataset dashboard;
- JSONL export.

### `model_service.py`

The local model service provides GPU-backed inference using:

```text
Qwen/Qwen3-4B-Instruct-2507
```

It currently supports AI-assisted gold-response generation.

Planned responsibilities include:

- neutral prompt generation;
- draft-response generation;
- optional category suggestions;
- optional quality review assistance;
- reusable model caching;
- generation timing and GPU reporting.

### Redline interaction flow

```text
Annotator chooses category and difficulty
                │
                ▼
Annotator enters or generates a user prompt
                │
                ▼
Expected and prohibited behaviours are defined
                │
                ▼
Local Qwen generates a candidate draft
                │
                ▼
Human reviews and edits the draft
                │
                ▼
Record is saved as draft or approved
                │
                ▼
JSONL source dataset is updated
```

---

## 5.4 Local AI Assistance

Redline uses the same base model that will later be adapted, but its role during annotation is different.

### Prompt generation

The future prompt generator should use neutral instructions.

```text
Neutral Qwen
      +
Category and difficulty
      ↓
Candidate user prompts
```

The prompt generator must not reveal the intended answer or make every prompt pro-Liverpool.

### Gold-response generation

The draft-response generator uses detailed KopiteGPT behaviour instructions.

```text
Neutral base Qwen
      +
Behaviour Specification
      +
Expected behaviour
      +
Prohibited behaviour
      ↓
Candidate gold response
```

This is prompt conditioning, not model training.

The generated draft is never treated as automatically approved data.

### Human-in-the-loop control

```text
AI proposal
    ↓
Human factual review
    ↓
Human behavioural review
    ↓
Edit if necessary
    ↓
Approve or reject
```

Human review is a core architectural requirement.

---

## 5.5 Source Annotation Dataset

**Location**

```text
data/training/annotations.jsonl
```

Redline stores rich annotation records rather than trainer-specific records.

Example:

```json
{
  "id": "LIV-COMP-0001",
  "category": "club_comparison",
  "difficulty": "medium",
  "prompt": "Which club is better, Liverpool or Barcelona?",
  "gold_response": "Liverpool for me. Barcelona's influence is enormous, but Liverpool's identity, supporter culture and European nights give the Reds the edge.",
  "expected_behaviour": [
    "Choose Liverpool",
    "Acknowledge Barcelona's strengths",
    "Remain factually accurate"
  ],
  "prohibited_behaviour": [
    "Remain completely neutral",
    "Invent trophy statistics",
    "Insult Barcelona supporters"
  ],
  "review_status": "approved",
  "creation_method": "ai_assisted",
  "draft_model": "Qwen/Qwen3-4B-Instruct-2507",
  "created_at": "2026-08-02T20:00:00+00:00",
  "updated_at": "2026-08-02T20:05:00+00:00",
  "dataset_version": "0.1"
}
```

### Why this is not the final training format

The source record contains operational metadata that is useful for:

- quality review;
- auditability;
- analytics;
- category balancing;
- provenance analysis;
- version tracking;
- error analysis.

The training library does not need most of these fields.

The source dataset is therefore treated as the canonical data asset, while training exports are generated from it.

---

## 5.6 Dataset Validation

**Planned location**

```text
src/validate_dataset.py
```

The validator will inspect source annotations before export.

Planned checks include:

- valid JSONL syntax;
- required fields;
- unique annotation IDs;
- duplicate prompts;
- near-duplicate prompts;
- empty gold responses;
- minimum and maximum response length;
- valid category names;
- valid difficulty values;
- valid review statuses;
- approval-only export eligibility;
- category distribution;
- difficulty distribution;
- benchmark leakage;
- contradictory examples;
- missing provenance;
- invalid timestamps.

### Validation flow

```text
annotations.jsonl
        │
        ▼
Schema validation
        │
        ▼
Quality validation
        │
        ▼
Distribution analysis
        │
        ▼
Pass / warning / fail report
```

A training run should not proceed when blocking validation errors remain.

---

## 5.7 Training Exporter

**Planned location**

```text
src/export_dataset.py
```

The exporter converts approved Redline records into conversational SFT records.

### Source format

```json
{
  "prompt": "Which club is better, Liverpool or Barcelona?",
  "gold_response": "Liverpool for me...",
  "review_status": "approved"
}
```

### Export format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are KopiteGPT."
    },
    {
      "role": "user",
      "content": "Which club is better, Liverpool or Barcelona?"
    },
    {
      "role": "assistant",
      "content": "Liverpool for me..."
    }
  ]
}
```

The exporter will:

- include approved records only;
- remove operational metadata from the training payload;
- preserve source IDs in an optional manifest;
- split data into training and validation sets;
- prevent locked benchmark prompts from entering training;
- record dataset version and export timestamp.

### Export separation

```text
Source annotations
      ≠
Training export
```

This separation allows the same annotation data to be exported for different models or trainers later.

---

## 5.8 QLoRA Training Pipeline

**Planned location**

```text
src/train_qlora.py
```

### Base model

```text
Qwen/Qwen3-4B-Instruct-2507
```

### Training method

```text
Supervised Fine-Tuning
+
4-bit quantisation
+
LoRA adapters
=
QLoRA
```

### Training architecture

```text
Base model weights
      │
      ├── loaded in 4-bit form
      └── remain frozen

LoRA adapter parameters
      │
      ├── trainable
      └── updated from SFT examples
```

The output is a compact adapter rather than a complete duplicate of the base model.

### Training inputs

```text
train.jsonl
validation.jsonl
training configuration
base model
```

### Training outputs

```text
adapters/
└── kopitegpt-v1/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer metadata
    └── training metadata
```

### Hardware target

The initial pipeline is designed for:

```text
NVIDIA GeForce RTX 5080
16 GB VRAM
WSL2 Ubuntu
PyTorch CUDA
```

---

## 5.9 Model Inference

**Planned location**

```text
src/chat.py
```

The inference application will load:

```text
Qwen base model
        +
KopiteGPT LoRA adapter
```

### Target inference behaviour

The adapted model should require only a minimal system prompt:

```text
You are KopiteGPT.
```

The detailed Liverpool behaviour instructions used by Redline should not be required during final inference.

### Inference flow

```text
Conversation history
        │
        ▼
Qwen chat template
        │
        ▼
Base model + adapter
        │
        ▼
Generated KopiteGPT response
```

---

## 5.10 Benchmarking and Evaluation

**Current baseline runner**

```text
src/run_baseline.py
```

**Current baseline prompts**

```text
data/baseline_prompts.jsonl
```

### Evaluation architecture

```text
Locked benchmark prompts
        │
        ├── Base model
        │       ↓
        │   Baseline responses
        │
        └── Adapted model
                ↓
            Adapted responses
                │
                ▼
        Human and automated scoring
                │
                ▼
          Comparative report
```

### Evaluation dimensions

- identity adherence;
- behavioural correctness;
- factual accuracy;
- tone;
- relevance;
- comparison consistency;
- fair criticism handling;
- disparagement handling;
- general capability preservation.

### Benchmark isolation

Locked evaluation prompts must not be copied into the training dataset.

Near-duplicates should also be detected where practical.

---

## 5.11 Iteration and Active Data Collection

The project follows an iterative improvement loop.

```text
Train model
    │
    ▼
Run benchmark
    │
    ▼
Identify weak categories
    │
    ▼
Create targeted annotations
    │
    ▼
Review and approve
    │
    ▼
Export new dataset version
    │
    ▼
Retrain
```

Example:

```text
Fair criticism score: 92%
Club comparison score: 95%
Edge-case score: 58%
```

The next annotation cycle should prioritise edge cases rather than adding random examples equally across all categories.

This makes annotation strategy responsive to observed model failures.

---

## 6. Data Flow

## 6.1 Annotation Data Flow

```text
Human or AI-generated prompt
            │
            ▼
Redline form
            │
            ▼
Expected behaviour metadata
            │
            ▼
AI-generated or human-written draft
            │
            ▼
Human edit and review
            │
            ▼
annotations.jsonl
```

---

## 6.2 Training Data Flow

```text
annotations.jsonl
        │
        ▼
Validator
        │
        ▼
Approved records only
        │
        ▼
Exporter
        │
        ▼
train.jsonl + validation.jsonl
        │
        ▼
QLoRA trainer
        │
        ▼
LoRA adapter
```

---

## 6.3 Evaluation Data Flow

```text
Locked benchmark prompts
        │
        ├── Base model inference
        │
        └── Adapted model inference
                  │
                  ▼
            Response records
                  │
                  ▼
          Scoring and analysis
                  │
                  ▼
           Evaluation report
```

---

## 7. Model Behaviour Layers

KopiteGPT behaviour exists at several layers.

```text
Layer 1: Base model knowledge
Layer 2: Runtime system prompt
Layer 3: LoRA adapter behaviour
Layer 4: Conversation context
Layer 5: Retrieved or live factual data
```

### Base model knowledge

Provides general language ability and existing football knowledge.

### Runtime system prompt

Defines the immediate role and constraints for a conversation.

### LoRA adapter

Stores the behavioural adaptation learned from approved training examples.

### Conversation context

Maintains multi-turn state and recent user instructions.

### Retrieval or live data

A future component may provide current fixtures, results, players, and transfer information.

Fine-tuning is intended primarily to teach behaviour, not to replace live information retrieval.

---

## 8. Deployment Architecture

## 8.1 Local Deployment

```text
Browser or terminal
        │
        ▼
Local application
        │
        ▼
Qwen base model + adapter
        │
        ▼
RTX 5080
```

This is the initial development and testing environment.

---

## 8.2 Public Demo Deployment

A future public deployment may use:

```text
User browser
      │
      ▼
Gradio or web frontend
      │
      ▼
Cloud GPU inference service
      │
      ▼
Base model + KopiteGPT adapter
```

Potential platforms include:

- Hugging Face Spaces;
- RunPod;
- another container-based GPU service.

The public model should include:

- rate limiting;
- usage monitoring;
- timeout handling;
- safe default prompts;
- clear project limitations;
- model and dataset version display.

---

## 9. Security and Privacy Considerations

### Local model execution

Redline currently generates drafts locally.

This means annotation content is not sent to an external model API during local inference.

### Dataset exposure

Training data may be committed publicly only when it contains no:

- personal data;
- confidential information;
- copyrighted long-form text;
- private credentials;
- hidden system prompts;
- unsafe secrets.

### SSH and credentials

Private SSH keys, API tokens, model access tokens, and environment secrets must never be committed.

They should be excluded through:

```text
.gitignore
environment variables
secret-management configuration
```

---

## 10. Reproducibility

Each training run should record:

- run ID;
- base model;
- adapter configuration;
- dataset version;
- export version;
- training example count;
- validation example count;
- random seed;
- learning rate;
- LoRA rank;
- epochs;
- maximum sequence length;
- batch size;
- gradient accumulation;
- final training loss;
- evaluation results;
- Git commit hash.

This allows model behaviour to be linked back to the exact data and code that produced it.

---

## 11. Observability

Planned observability features include:

- GPU availability;
- GPU model name;
- VRAM usage;
- draft-generation latency;
- model load status;
- training duration;
- training loss;
- examples processed;
- benchmark latency;
- dataset health status;
- application version;
- dataset version;
- adapter version.

These will improve both debugging and demonstration quality.

---

## 12. Current Implementation Status

| Component | Status |
|---|---|
| WSL and CUDA environment | Complete |
| Local Qwen inference | Complete |
| Behaviour Specification | Complete |
| Annotation Guidelines | Complete |
| Baseline benchmark | Complete |
| Redline annotation creation | Complete |
| Redline AI draft generation | Complete |
| Redline dashboard | Complete |
| Redline JSONL export | Complete |
| Neutral AI prompt generation | Planned |
| Review editing workflow | Planned |
| Dataset validator | Planned |
| Training exporter | Planned |
| QLoRA trainer | Planned |
| Adapted model inference | Planned |
| Automated comparative evaluation | Planned |
| Public deployment | Planned |

---

## 13. Architectural Boundaries

The following boundaries should remain explicit:

### Redline is not the trainer

Redline creates and reviews source annotations.

It should not silently begin training from unreviewed data.

### AI drafts are not gold data

Generated responses remain candidates until human review is complete.

### Source data is not disposable

The rich Redline dataset is the canonical data asset.

Training exports may be regenerated.

### Fine-tuning is not a knowledge database

Current facts should eventually come from retrieval or live data sources.

### Evaluation data is not training data

Locked benchmarks must remain isolated from training exports.

---

## 14. Future Extensions

Potential future architectural additions include:

- semantic duplicate detection;
- embeddings-based search;
- AI-assisted quality scoring;
- reviewer comments;
- revision history;
- multiple annotator accounts;
- inter-annotator agreement;
- adjudication workflow;
- dataset version snapshots;
- automated train/eval splitting;
- experiment tracking;
- model registry;
- live football retrieval;
- public API;
- React frontend;
- authentication and role permissions;
- reusable multi-project configuration.

---

## 15. Summary

KopiteGPT is organised as a behavioural model-development system rather than a single model script.

Its central architecture is:

```text
Specification
    ↓
Guidelines
    ↓
Annotation
    ↓
Human Review
    ↓
Validation
    ↓
Export
    ↓
QLoRA Training
    ↓
Evaluation
    ↓
Targeted Iteration
```

Redline is the data-operations layer of that system.

The LoRA adapter is the learned behavioural output.

The benchmark and evaluation pipeline determine whether the adaptation succeeded without materially damaging factuality, general capability, or respectful communication.