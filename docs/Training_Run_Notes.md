# KopiteGPT Training Run Notes

## What We Trained

KopiteGPT was trained from:

```text
Qwen/Qwen3-4B-Instruct-2507
```

The goal was behavioural adaptation, not teaching the model a new factual database. The training examples were intended to teach the model how to respond when it encounters:

- subjective club comparisons;
- fair criticism of Liverpool;
- unfair attacks on Liverpool;
- rival banter;
- supportive Liverpool comments;
- factual football questions; and
- unrelated, off-topic questions.

The model was not trained from scratch. We started with an already capable Qwen model and added a small trainable LoRA adapter.

## The Important Difference

The base Qwen model contains approximately:

```text
4,055,498,240 total parameters
```

A parameter is a numerical value inside the neural network. During normal full fine-tuning, all of those values could potentially be updated. That would require much more memory and would make the experiment heavier.

For this run, the base model was frozen. Its original values were not changed.

Only the LoRA adapter was trained:

```text
33,030,144 trainable parameters
0.814% of the model's total parameters
```

In simple terms:

```text
Original Qwen model:      4,055,498,240 values, frozen
KopiteGPT LoRA adapter:      33,030,144 values, trainable
```

The adapter is about 0.8% of the total parameter count. It is the part that learned from the KopiteGPT examples.

The trained adapter is saved at:

```text
output/training/kopite_adapter/adapter_model.safetensors
```

## The Training Data

Redline contained 51 approved annotations. Each annotation included a user prompt and an approved gold response.

The script split them reproducibly into:

```text
46 training examples
5 holdout evaluation examples
```

The 46 training examples were shown to the model during learning. The 5 holdout examples were kept separate and used to check how well the model performed on examples it was not directly trained on.

The split was controlled by a fixed random seed, so rerunning the preparation uses the same split:

```text
Seed: 42
```

The converted datasets are saved here:

```text
output/training/train.jsonl
output/training/eval.jsonl
```

## How Each Example Was Presented

Each Redline annotation was converted into Qwen chat format:

```text
System message: KopiteGPT behaviour instructions
User message:   the benchmark or annotation prompt
Assistant:      the approved gold response
```

The system message described the broad behaviour contract. The examples showed the model how to apply that contract in specific situations.

The training used completion-only loss. That means the model was primarily trained to reproduce the assistant response, rather than being rewarded for reproducing the system and user instructions.

## What Happened During One Training Step

The main training call is:

```python
result = trainer.train()
```

TRL performs the following cycle internally:

1. It takes a batch of chat examples.
2. Qwen reads the system message and user prompt.
3. Qwen predicts the assistant response one token at a time.
4. The predicted response is compared with the approved gold response.
5. A loss value measures how different the prediction was from the target.
6. Backpropagation calculates which LoRA values contributed to that loss.
7. Gradients are accumulated over 8 batches.
8. The optimizer updates the LoRA values.
9. The original Qwen weights remain frozen.

This cycle repeats across the dataset and across each epoch.

## Epochs and Updates

The run used:

```text
3 training epochs
Batch size: 1 example at a time
Gradient accumulation: 8 batches
Effective batch size: 8 examples
```

An epoch means one pass through the training examples. The model saw the 46 training examples three times in total.

Because gradients were accumulated over eight batches, the optimizer made:

```text
18 optimizer updates
```

That is different from the number of individual examples processed. The model processed the training examples over three epochs, while the optimizer applied 18 grouped updates.

## QLoRA Configuration

The base model was loaded using 4-bit quantization:

```text
Quantization: 4-bit NF4
Double quantization: enabled
```

Quantization stores the frozen base model using lower-precision numbers. This reduces memory use while keeping the model useful for inference and adapter training.

The adapter used:

```text
LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
```

LoRA was applied to Qwen's attention and feed-forward projection layers:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

The GPU supported bfloat16, so the run used:

```text
bf16: enabled
```

Flash Attention was not installed. The script therefore used the built-in SDPA attention implementation instead:

```text
attention implementation: sdpa
```

Gradient checkpointing was enabled to reduce memory usage. It saves memory by recomputing some intermediate values during backpropagation instead of storing all of them.

## Recorded Results

The run completed on 20 August 2026.

```text
Training runtime:       50.71 seconds
Final training loss:    1.799
Final holdout loss:     1.342
Optimizer updates:      18
```

The loss generally decreased during training. The recorded training loss moved from approximately 2.76 at the first step to approximately 1.12 at the final logged step. Holdout loss was recorded during training and reached approximately 1.34 at the final evaluation checkpoint.

Loss is a measure of how closely the model predicts the training response tokens. Lower is generally better, but loss alone does not prove that the model has learned the intended personality or that it will behave well on new prompts.

## What Was Saved

The adapter and tokenizer were saved under:

```text
output/training/kopite_adapter/
```

Important files include:

```text
adapter_model.safetensors   trained LoRA weights
adapter_config.json         LoRA configuration
tokenizer.json              tokenizer data
tokenizer_config.json       tokenizer settings
chat_template.jinja         Qwen chat formatting
```

The run reports were saved under:

```text
output/training/training_config.json
output/training/training_metrics.json
output/training/training_summary.json
output/training/loss_curve.svg
output/training/learning_rate_curve.svg
```

## What Happens When It Responds

The adapter is not a complete model by itself. During inference, the original Qwen model is loaded and the saved LoRA adapter is attached:

```text
Prompt
  |
  v
Frozen Qwen base model + KopiteGPT LoRA adapter
  |
  v
Response
```

The adapter changes how the base model processes the prompt. The original Qwen model remains available and can be used without the adapter for a clean comparison.

The comparison script tests both conditions using the same neutral system prompt:

```text
Base Qwen
Base Qwen + LoRA
```

## What the Training Results Prove

The results prove that:

- the training pipeline completed successfully;
- the adapter was created and saved;
- only a small portion of the model was trainable;
- the training loss decreased during the run;
- the holdout examples were evaluated during training; and
- the adapter can be loaded for inference.

## What the Results Do Not Prove

The results do not yet prove that KopiteGPT is behaviourally better.

There were only 5 holdout examples, which is too few for a strong statistical conclusion. A lower loss also does not automatically mean better Liverpool preference, better criticism handling, better banter, or better factual accuracy.

That is why the next important measurement is the locked 40-prompt comparison:

```text
data/Benchmark/baseline_prompts.jsonl
output/training/benchmark_comparison.jsonl
```

The comparison evaluates the same prompts with:

```text
Base Qwen
Base Qwen + LoRA
```

The category-specific Redline evaluator then checks observable behaviours rather than assigning an arbitrary overall score.

## Plain-English Summary

We did not retrain the whole 4-billion-parameter Qwen model.

We froze the original model and trained a small adapter containing about 33 million values. The adapter learned from 46 approved examples over 3 passes through the data. It was trained using QLoRA, which made the experiment fit efficiently on the RTX 5080.

When KopiteGPT responds, it uses the original Qwen model together with the trained adapter. The adapter is the learned behavioural adjustment. The 40-prompt Base-vs-LoRA comparison is the proper test of whether that adjustment improved the intended behaviour without damaging factuality or general usefulness.
