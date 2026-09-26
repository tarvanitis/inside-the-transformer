# Supervised Fine-Tuning

Pre-training leaves the model knowledgeable but directionless. Ask an unaligned base model
"What is the capital of France?" and it is more likely to continue the sentence as a quiz than
to answer. Supervised Fine-Tuning (SFT) is the stage that gives the model a job: it trains on
thousands of examples of what *good responses look like*, and the model learns format, tone, and
instruction-following from those demonstrations.

**In this chapter**

- Response masking: why SFT uses the same loss as pre-training but applies it differently.
- How to implement loss masking in PyTorch.
- Training data: quality versus quantity, and the landmark datasets.
- Chat templates and special tokens.
- LoRA: efficient fine-tuning without updating every weight.
- Reasoning distillation: the 2025 shift toward SFT on chain-of-thought traces.

This chapter assumes cross-entropy loss from Chapter 2 and the training loop from Chapter 10.

## The one idea that makes SFT different

The SFT loss function is identical to pre-training cross-entropy (Chapter 2, Chapter 13):
with one critical change: **only response tokens generate a gradient**.

$$L_{\text{SFT}} = -\frac{1}{|R|} \sum_{t \in R} \log P(y_t \mid y_1 \ldots y_{t-1};\, \theta)$$

The instruction tokens are still passed to the model as context, so the model reads the question.
But their prediction errors are *discarded*. No gradient flows back through the instruction.
Only the response tokens, the set R, contribute to the loss and update the weights.

::: {.callout .idea}
This is the difference between "continue this text" (pre-training) and "answer this question"
(SFT). The masking tells the model: whatever I put in the instruction slot is what you were
given. What I put in the response slot is what you should have produced. Minimize the loss on
the response, not the prompt.
:::

The mechanism is simple in code. Positions in the loss tensor that belong to the instruction
are set to the `ignore_index`, and PyTorch's `F.cross_entropy` then skips them entirely:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def sft_loss(logits: torch.Tensor, targets: torch.Tensor,
             response_start: int) -> torch.Tensor:
    """
    logits:         (B, T, vocab_size)
    targets:        (B, T)           — full sequence token IDs
    response_start: first token index of the response (0-indexed)
    """
    # Mask instruction positions so they produce no gradient
    masked = targets.clone()
    masked[:, :response_start] = -100   # -100 is F.cross_entropy's ignore_index

    # Flatten and compute. Masked positions are silently skipped
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),   # (B×T, vocab_size)
        masked.view(-1),                    # (B×T,)
        ignore_index=-100,
    )


# Sanity-check: the loss changes only with response_start
import math
B, T, V = 2, 12, 1000
logits  = torch.randn(B, T, V)
targets = torch.randint(0, V, (B, T))

loss_all  = sft_loss(logits, targets, response_start=0)   # no masking
loss_half = sft_loss(logits, targets, response_start=6)   # mask first 6 tokens

print(f"Full-sequence loss: {loss_all.item():.4f}")
print(f"Response-only loss: {loss_half.item():.4f}")
# These differ because different token positions are included — both are valid losses.
# The gradient flows only through tokens that contribute to the loss value.
```

The training loop is otherwise identical to pre-training (Chapter 10): forward pass → loss →
`loss.backward()` → `opt.step()`.

## Worked example: loss masking on a real sample

Consider this training example, formatted with chat-template delimiters:

```
// Full sequence fed to the model (instruction + response)
[Human: What is the capital of France?]   ← masked tokens — no gradient
[Assistant: The capital of France is Paris.]  ← response tokens — loss here

// Token-level loss for the response only:
"The"     → P(The     | context) = 0.72  → L = 0.33   contributes
"capital" → P(capital | context) = 0.55  → L = 0.60   contributes
"of"      → P(of      | context) = 0.91  → L = 0.09   contributes
"France"  → P(France  | context) = 0.68  → L = 0.39   contributes
"is"      → P(is      | context) = 0.87  → L = 0.14   contributes
"Paris"   → P(Paris   | context) = 0.31  → L = 1.17   ← main learning signal!

SFT loss = (0.33 + 0.60 + 0.09 + 0.39 + 0.14 + 1.17) / 6 = 0.45
```

"Paris" (L = 1.17) carries nearly twice the weight of the next largest token, and almost four
times the average of the other five. The model
was least certain about the informative word, which is exactly where most of the learning
should happen. The instruction tokens ("What is the capital of France?") contribute nothing
to this average: the loss is 0.45, computed purely over the six response tokens.

::: {.figure}
![](assets/figures/ch14/fig-response-masking.svg)
:::

::: {.caption}
**Figure 14.1.** Response masking. The instruction tokens are fed as context but produce no gradient. Only the response tokens contribute to the loss, and the model learns most from the token it was least sure of ("Paris").
:::

::: {.callout .note}
**Teacher forcing.** At every training step, the model is given the *correct* previous tokens
as context, not its own (possibly wrong) predictions. This "forcing" the correct prefix at each
step makes training fast and stable. At inference the model uses its own previous outputs
instead. The mismatch between training and inference is called *exposure bias*, and is a
well-known limitation of teacher-forcing-based training.
:::

## Chat templates and special tokens

SFT requires the model to *know which part of the context it should complete*. The solution is
a chat template: a structured format with special tokens that mark the boundary between system
prompt, user turn, and assistant turn.

```
// Llama 3 chat template format
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is the capital of France?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The capital of France is Paris.<|eot_id|>

// Mistral / Alpaca style (simpler):
[INST] What is the capital of France? [/INST] The capital of France is Paris.
```

The special tokens (`<|start_header_id|>`, `[INST]`, `<|assistant|>`) are added to the
vocabulary during pre-training or SFT, and their embeddings are learned alongside the weights.
At inference, the user's message is wrapped in the same template, and the model generates text
starting from the `assistant` marker. It has learned that this is the slot it fills.

Different model families use different templates, and mixing templates causes silent quality
degradation. Inference frameworks (Transformers, vLLM) apply the template automatically when
you call `tokenizer.apply_chat_template()`.

## Training data: quality beats quantity

SFT datasets are measured in thousands of examples, not trillions of tokens. The lesson from
several years of experimentation is stark: a small set of high-quality demonstrations
consistently outperforms a large set of mediocre ones.

| Dataset | Size | Source | Key finding |
|---|---|---|---|
| InstructGPT (2022) | 13k pairs | Human-written | High quality, set the standard for instruction-following |
| Alpaca (2023) | 52k pairs | GPT-3.5 synthetic (text-davinci-003) | Cheap to generate, quality noticeably worse |
| LIMA (2023) | 1k pairs | Curated human | "Less Is More for Alignment", competitive with 52k Alpaca examples |
| OpenHermes-2.5 | 1M pairs | GPT-4 synthetic | Broad coverage, GPT-4 quality recovers scale's advantage |

::: {.caption}
**Table 14.1.** Instruction-tuning datasets: size against measured effect.
:::

The LIMA result is particularly striking: 1,000 carefully curated examples, selected for
diversity and quality, produced an assistant model competitive with those trained on 50× more
data. The implication for practitioners: curation time is well spent.

::: {.callout .caution}
**Overfitting on small SFT datasets.** Because the dataset is small (thousands, not trillions
of examples), SFT is trained for only 2–5 epochs with a much lower learning rate than
pre-training (5e-6 to 2e-5, roughly 10–50× lower). More epochs or a higher LR cause the model
to memorize the training responses rather than learning the underlying pattern, and it will
reproduce exact phrasings from the training set instead of generalizing. Watch held-out
instruction-following accuracy, not just training loss, to detect overfitting early.
:::

## Hyperparameters

| Hyperparameter | Typical range | Contrast with pre-training |
|---|---|---|
| Learning rate | 5e-6 to 2e-5 | 10–50× lower (small dataset, high overfitting risk) |
| Epochs | 2–5 | Pre-training is typically 1 epoch on the corpus |
| Batch size | 128–512 sequences | Smaller than pre-training's millions of tokens/step |
| LR schedule | Cosine decay (same shape, shorter) | Same schedule, fewer steps |
| Optimizer | AdamW | Identical |
| Fine-tuning scope | Full or LoRA | Pre-training always updates all parameters |

::: {.caption}
**Table 14.2.** SFT hyperparameters, contrasted with pre-training.
:::

## LoRA: efficient fine-tuning

Updating all the weights of a 70B-parameter model for SFT is expensive: you need to store the
model's weights, gradients, and optimizer states, roughly 4× the weight memory for AdamW in fp32
(the weights, their gradients, and Adam's two running averages). **Low-Rank Adaptation (LoRA)** solves this by freezing the original weights and adding
a small trainable adapter to each weight matrix.

::: {.callout .plain}
The idea behind LoRA is to stop editing the giant weight matrix directly and instead learn a small
*patch* to add on top of it. The patch is built by multiplying two thin matrices together, and two
thin matrices can only make a "simple" change: one with a handful of independent directions rather
than thousands. *Low-rank* is just the technical name for that kind of simple change, and it turns
out to be all fine-tuning needs. You train the tiny patch and leave the original weights frozen, so
the memory set aside for gradients and optimizer state shrinks by orders of magnitude.
:::

For a weight matrix W of shape (d_out, d_in), LoRA adds two small matrices A (r × d_in) and
B (d_out × r) where r ≪ d_in. During the forward pass:

```
y = x @ (W + (α/r) · B @ A)^T     # original weights + scaled low-rank adapter
```

The `α/r` factor is a fixed scaling on the adapter's contribution. You will meet it as `lora_alpha`
the moment you open a LoRA config.

::: {.figure}
![](assets/figures/ch14/fig-lora.svg)
:::

::: {.caption}
**Figure 14.2.** LoRA. The pre-trained weights W stay frozen. Training adds a small low-rank patch, the product of two thin matrices B and A, so with r much smaller than d only a tiny fraction of the parameters are trained.
:::

Only A and B are trained (and their gradients computed). The original W is frozen. With r = 16
on a 70B model, the trainable parameters drop from 70B to roughly 100M, a 700× reduction in
memory for gradients and optimizer states, making SFT feasible on a single node.

::: {.callout .deepdive}
**Why low-rank works.** The update ΔW = B @ A has at most rank r. The hypothesis behind LoRA
is that the weight changes needed for fine-tuning lie in a low-dimensional subspace of the full
weight space. Adapting a pre-trained model to a new task does not require
changing all directions in the weight matrix, only a few. Empirically this holds: LoRA with
r = 8 or r = 16 typically matches or comes within a fraction of a percent of full fine-tuning
quality, at a tiny fraction of the memory cost. It also makes fine-tuned adapters portable:
you can ship a ~200 MB LoRA adapter (100M parameters at fp16) rather than a full 140 GB model.
:::

## Reasoning distillation: SFT in 2025–2026

A significant shift in modern SFT practice: training on **long chain-of-thought traces**
generated by a stronger reasoning model rather than on short human-authored responses.

DeepSeek-R1 demonstrated this at scale. After training a large reasoning model with GRPO
(Chapter 16), they generated ~800k training samples from that model (about 600k reasoning traces
plus 200k non-reasoning samples) and fine-tuned smaller, non-reasoning models (Qwen and Llama
variants) on them. The result: the small models acquired much of the large model's reasoning capability
(step-by-step problem decomposition, backtracking, self-checking) at a fraction of the cost of
running RL directly.

::: {.callout .note}
**Cold-start SFT.** In reasoning-model pipelines, SFT often plays a second role: a brief
"cold-start" pass on curated reasoning examples *before* the GRPO phase (Chapter 16). Without it,
pure RL from a base model can produce text that works but is hard to read. The model mixes
languages, uses idiosyncratic formatting, and reasons in an unstructured way. A small cold-start
SFT on well-formatted reasoning traces fixes the readability while leaving the RL stage free to
optimize for correctness.
:::

## What SFT teaches (and what it does not)

After SFT, the model can:

- Follow instructions reliably across a range of formats (Q&A, summarization, coding, dialogue).
- Match the expected output structure (markdown, JSON, specific length).
- Refuse clearly harmful requests if refusal examples are included.
- Adopt a consistent tone: helpful, polite, calibrated.

What SFT does *not* teach well: fine-grained quality preferences. A model can follow the
instruction "explain black holes" in many ways, and some are more insightful, more appropriately
hedged, or more genuinely clear than others. Demonstrating the ideal response to every possible
nuance is expensive. Preference optimization (the topic of Chapter 15) is what captures these
subtler distinctions by asking humans to compare pairs of responses rather than write them.

## Summary

- **SFT applies the same cross-entropy loss as pre-training**, but masks the instruction tokens.
  Only response-position gradients update the weights, forcing the model to learn to *respond*
  rather than to *continue arbitrary text*.
- **Quality dominates quantity**: 1,000 carefully curated examples (LIMA) can match 52,000
  synthetic ones. Curation is the primary investment.
- **Chat templates** define the boundary between instruction and response at both training and
  inference time. Mixing templates silently degrades quality.
- **LoRA** reduces trainable parameters by 100–1,000×, making SFT feasible without updating all
  weights.
- **Reasoning distillation** (SFT on chain-of-thought traces from a stronger model) is now the
  standard route to small reasoning models, cheaper and more stable than running RL on the small
  model directly.

> **Coming up:** SFT teaches format. Chapter 15 tackles quality: learning from human
> *preferences* rather than demonstrations, through reward models, RLHF/PPO, and DPO.
