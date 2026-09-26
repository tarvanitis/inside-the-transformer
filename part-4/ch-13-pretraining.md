# Pre-training

Pre-training is where the model earns its intelligence. A randomly initialized transformer,
billions of Gaussian-distributed weights that mean nothing, is exposed to trillions of tokens
of human text and asked one question, trillions of times: *what comes next?* By the end it has
compressed an enormous fraction of human knowledge and reasoning into those weights. Everything
in the subsequent training stages is fine-tuning that base.

**In this chapter**

- The causal language modeling objective: what the model is actually optimizing.
- Training data: scale, sources, quality filtering, and the shift to synthetic augmentation.
- The learning-rate schedule and key hyperparameters.
- Scaling laws: how performance relates to model size, tokens, and compute.
- Long-context training and context-extension techniques.
- What the model learns, and what it does not.

This chapter assumes cross-entropy loss and AdamW from Chapter 2, and the causal masking
mechanism from Chapter 5.

## The objective: next-token prediction

Pre-training is **causal language modeling**: at every position in a sequence, predict the next
token using only the tokens that came before. The causal mask (Chapter 5) enforces this:
position *t* sees positions 0 through *t*, its own position included, never any position after it.

The loss is cross-entropy, averaged over every position in the batch (Chapter 2):

$$L = -\frac{1}{T} \sum_{t=1}^{T} \log P(y_t \mid y_1, y_2, \ldots, y_{t-1};\, \theta)$$

Every term in plain English:

- **L.** A single "how wrong were we" number, minimized by training.
- **θ.** All of the model's weights: every parameter being learned.
- **yₜ.** The actual next token at position *t*. **y₁ … yₜ₋₁** — everything the model is allowed to see.
- **P(yₜ | … ; θ).** The probability the model assigned to the *correct* next token.
- **−log P.** Near 0 when P is close to 1 (confident and right), grows large as P falls toward 0.
- **(1/T) Σₜ.** Average over all *T* positions in the sequence.

::: {.callout .idea}
A 2,048-token sequence provides 2,047 training signals in one forward pass, one for every position
that has a next token.
That is why pre-training scales so efficiently: every token in every document is simultaneously
a context and a prediction target.
:::

### Worked example: causal loss on one batch

Sequence: "The cat sat on" → token IDs [845, 3628, 4576, 319]. With causal masking, the model makes
three predictions, one for every position that has a next token:

```
// Position 0: what follows "The"?
P(cat | The)       = 0.12   →   L₁ = −log(0.12) = 2.12

// Position 1: what follows "The cat"?
P(sat | The cat)   = 0.08   →   L₂ = −log(0.08) = 2.53

// Position 2: what follows "The cat sat"?
P(on | The cat sat) = 0.35  →   L₃ = −log(0.35) = 1.05

// Batch loss = average surprise across all positions
Batch loss = (2.12 + 2.53 + 1.05) / 3 = 1.90
```

Each Lₜ is the model's "surprise" at that position. The model was most surprised by "sat" (8%),
which carries the largest penalty. After training on trillions of tokens, a frontier model's
loss falls toward ~1.5–2.0, so the effective number of choices per token drops to roughly e^1.6
≈ 5. The model is not guessing blindly. It genuinely knows what tends to come next.

**Perplexity**, the metric used to track pre-training progress, is the exponential of the
average loss: PPL = e^L. A perplexity of 10 means the model was as uncertain as if choosing
uniformly among 10 equally likely tokens at each step. Lower is better. A randomly initialized
model on a 50,000-word vocabulary has PPL ≈ 50,000. Frontier models on held-out text reach
PPL ≈ 5–8.

## Training data

### Scale and sources

Pre-training requires vastly more data than any other stage. Current frontier models train on
1–15+ trillion tokens:

- **Llama 3.1** (all sizes): ~15 trillion tokens
- **DeepSeek-V3**: 14.8 trillion tokens
- **Original GPT-3** (2020): 300 billion tokens, already enormous at the time

The sources are diverse by design. A typical mixture:

| Source | Why it's included |
|---|---|
| Common Crawl (web) | Scale, the primary volume source |
| Wikipedia | High-quality factual text, multiple languages |
| Books (Books3, Project Gutenberg) | Long-range coherence, literary reasoning |
| Code (GitHub, The Stack) | Programming ability, structured reasoning |
| Scientific papers (arXiv, PubMed) | Technical and mathematical reasoning |
| Legal and financial text | Formal language, precise reasoning |
| Multilingual text | Cross-lingual transfer |

::: {.caption}
**Table 13.1.** Pre-training corpus sources and why each is included.
:::

The exact mixture ratios are among the most closely guarded secrets in the industry. Upsampling
high-quality sources (code, books, Wikipedia) relative to their natural frequency in web crawls
is standard practice.

### Quality filtering

Raw web crawls are noisy. A typical filtering pipeline removes:

- **Near-duplicates.** Exact or near-exact duplicate documents inflate token counts without
  providing new information and can cause the model to over-represent certain sources.
- **Non-target language.** Language-detection filters keep the intended language distribution.
- **Low-quality content.** Heuristic scores (very short documents, excessive punctuation, high
  symbol density) and **perplexity-based filtering**: a small reference LM trained on clean text
  scores every document, and once length and language have been checked, the ones it finds most
  surprising are dropped, because a document that surprises a clean-text model is usually
  incoherent or machine-generated rather than genuinely novel.
- **Harmful content.** Safety and toxicity classifiers remove the worst content. Exact-match
  blocklists catch known harmful material.

::: {.callout .deepdive}
**Model-based quality classifiers.** A significant recent shift: use a small LM to judge
whether a document is "educational" or "high-quality" and keep only top-scoring documents.
FineWeb-Edu (HuggingFace, 2024) and Llama 3's data pipeline both use this approach. The
classifier is trained on a small set of human-labeled examples and then applied to hundreds of
billions of documents. The lesson of this era: *token quality and mixture* often matter more
than raw token count. A model trained on 1T carefully filtered tokens can match one trained on
3T raw tokens.
:::

### Synthetic and curated data (2024–2026)

As the supply of high-quality *human* web text has become a bottleneck, frontier pre-training
now leans heavily on synthetic augmentation:

- **Rephrasing and expansion**: an existing strong LLM rewrites web documents into cleaner, more
  educational prose.
- **Textbook-style content**: the "Phi" line of models (Microsoft) demonstrated that small models
  trained on LLM-generated "textbooks" punch far above their parameter count on reasoning
  benchmarks: "textbooks are all you need."
- **Code and math upsampling**: reasoning improvements from code and mathematics generalize to
  other domains. Frontier labs heavily upsample these.
- **Reasoning traces**: some 2025-era pre-training mixes include step-by-step reasoning
  demonstrations generated by stronger models (see Chapter 16 on reasoning RL).

## Optimizer and hyperparameters

Pre-training uses **AdamW** throughout (Chapter 2's optimizer section). The key hyperparameters:

| Hyperparameter | Typical value | What it does |
|---|---|---|
| Peak learning rate | 1e-4 to 3e-4 | Step size at the top of the warm-up |
| LR schedule | Cosine decay to ~10% of peak | Slowly reduces step size as training matures |
| Warm-up steps | 1,000–2,000 | Ramps LR from near zero to peak, preventing early instability |
| Batch size | 4–16M tokens per step | Larger batches = more stable gradient estimates |
| β₁ / β₂ | 0.9 / 0.95 | Adam momentum and variance-averaging coefficients (β₂ = 0.95 is lower than Adam's 0.999 default, empirically more stable at the gradient magnitudes of large-scale pre-training) |
| Weight decay | 0.1 | L2 regularization, prevents weight blow-up |
| Gradient clipping | 1.0 | Caps gradient norm, prevents catastrophic steps |
| Context length | 4k–128k tokens | Sequence length per training example |

::: {.caption}
**Table 13.2.** Typical pre-training hyperparameters.
:::

::: {.callout .note}
**The warm-up schedule.** At the very start of training, the weights are random, gradients are
chaotic, and the Adam optimizer's running statistics are uninitialized. Ramping the learning
rate from near-zero to the peak over the first thousand steps prevents these chaotic gradients
from causing a destructive early update. After warm-up, the cosine schedule gradually decays the
rate over the rest of training, giving fine-grained adjustments near convergence.
:::

### The learning-rate schedule in code

```python
import math

def get_lr(step: int, peak_lr: float, warmup_steps: int,
           total_steps: int, min_lr_frac: float = 0.1) -> float:
    min_lr = peak_lr * min_lr_frac
    if step < warmup_steps:
        return peak_lr * step / warmup_steps                   # linear warm-up
    decay = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay))          # 1→0 over training
    return min_lr + (peak_lr - min_lr) * cosine                # decay to min_lr

# Example: 2000-step warm-up, 100k total steps, peak LR = 3e-4
for step in [0, 500, 2000, 50000, 100000]:
    print(f"step {step:6d}:  lr = {get_lr(step, 3e-4, 2000, 100_000):.2e}")
# step      0:  lr = 0.00e+00
# step    500:  lr = 7.50e-05
# step   2000:  lr = 3.00e-04  ← peak
# step  50000:  lr = 1.69e-04
# step 100000:  lr = 3.00e-05  ← 10% of peak (min_lr_frac=0.1)
```

::: {.figure}
![](assets/figures/ch13/fig-lr-schedule.svg)
:::

::: {.caption}
**Figure 13.1.** The learning-rate schedule. The rate ramps up linearly during a short warm-up, reaches its peak, then follows a cosine curve down to about 10% of the peak over the rest of training.
:::

This is not model code. It is the trainer, wrapping the model. The learning rate is passed to
the optimizer's `param_groups` at each step.

## Scaling laws

Pre-training is expensive enough that empirical scaling laws are essential for planning: they
let you predict a model's quality from its size and training budget *before* committing to the
run.

The **Chinchilla scaling law** (Hoffmann et al., DeepMind 2022) showed that for a fixed
compute budget, you get the best loss by scaling model size and training tokens together in a
roughly 1:20 ratio: one parameter per 20 training tokens.

| Model size | Chinchilla-optimal tokens |
|---|---|
| 7B parameters | ~140B tokens |
| 13B parameters | ~260B tokens |
| 70B parameters | ~1.4T tokens |
| 405B parameters | ~8T tokens |

::: {.caption}
**Table 13.3.** Chinchilla-optimal token budgets by model size.
:::

::: {.figure}
![](assets/figures/ch13/fig-chinchilla-scaling.svg)
:::

::: {.caption}
**Figure 13.2.** Chinchilla scaling. For a fixed compute budget the best loss comes from growing model size and training tokens together, roughly 20 tokens per parameter (the dashed line). Most production models are deliberately "overtrained" well past that point, because a smaller model that has seen more data is cheaper to serve.
:::

::: {.callout .caution}
**"Overtrained" models.** Most frontier models now train far *beyond* the Chinchilla-optimal
token count. Llama 3 8B trained on 15T tokens — nearly 100× the Chinchilla optimal. Why? The
optimal from Chinchilla minimizes training cost for a given loss. But *inference* is not free:
a model that has seen more data reaches a given quality level with *fewer parameters*, making
it cheaper to serve. A 7B model trained on 15T tokens can match a 13B model trained on 7T
tokens at inference time, and millions of inference calls a day make that smaller model
dramatically cheaper. Chinchilla optimizes for one training run. Production optimizes for the
lifetime of the model.
:::

## Long-context training

Frontier models are no longer trained at a single fixed context length. A common recipe:

1. **Main pre-training:** the bulk of the token budget at a short context (4k–8k tokens) for
   efficiency: short sequences pack more documents per batch and use less memory.
2. **Context-extension phase:** a dedicated extension run on long documents (books, legal filings,
   code repositories), adjusting the positional encoding so the model can handle sequences up
   to 128k, 512k, or even 1M tokens.

The critical adjustment in context extension is to the **RoPE base frequency** (Chapter 4). The
original RoPE was designed for sequences up to a few thousand tokens. Longer sequences push
position encodings into a regime the model never saw during training and cause attention patterns
to degrade. Rescaling RoPE's base frequency (its "theta" parameter) lets the model interpolate
positional information to new lengths. The simplest approaches just raise RoPE's base frequency, as in Llama 3's adjusted base frequency
(ABF) and the closely related NTK-aware scaling. YaRN is a more elaborate variant that rescales different frequency bands by
different amounts and adds an attention-temperature term.
Context extension is a small fraction of total compute but is what enables long-document
reasoning and long agent trajectories.

## What the model learns

After trillions of next-token predictions with no labels, instructions, or human feedback, the
pre-trained base model has learned:

- **World knowledge.** Facts, dates, named entities, relationships.
- **Language structure.** Grammar, syntax, idiom, register.
- **Code.** Python, JavaScript, SQL, and 100+ more languages.
- **Reasoning patterns.** Analogies, multi-step inference, if-then chains.
- **Styles.** Formal writing, dialogue, narrative, technical prose.
- **Multilingual ability.** Cross-lingual transfer from multilingual training data.

::: {.callout .plain}
What the base model does *not* learn from pre-training: how to follow instructions, how to
refuse harmful requests, or how to behave as an assistant. Ask an unaligned base model "What
is the capital of France?" and it is more likely to generate more questions, or continue the
sentence as a quiz, than to answer. That behavior comes from Stage ②, SFT.
:::

::: {.callout .lens}
**Representation.** A useful mental model: pre-training compresses. A 70B model trained on 15
trillion tokens has read roughly 60 terabytes of text and stores what it learned in 140 gigabytes
of weights, a compression ratio of around 400:1. The compression is lossy (the model generalizes
and forgets specifics), but the pattern structure is preserved. What the model "knows" is this compressed structure,
not any stored verbatim text.
:::

## Summary

- **Pre-training objective:** next-token prediction with cross-entropy loss over every token
  position. A single sequence provides as many training signals as it has tokens.
- **Scale:** 1–15 trillion tokens of filtered, deduplicated, mixed-source text. Quality and
  mixture matter more than raw count.
- **Optimizer:** AdamW with a warm-up then cosine decay schedule, gradient clipping, and
  weight decay. The learning-rate schedule is as important as the architecture.
- **Scaling laws:** model size and training tokens should grow together (~20 tokens per
  parameter for compute efficiency, but inference cost pushes labs toward "overtrained" small models).
- **Long-context:** a dedicated context-extension phase adjusts positional encoding for sequences
  beyond the main training length.
- **Output:** a base model with vast knowledge and fluency, but no instruction-following
  behavior. That is the job of the next stage.

> **Coming up:** Chapter 14 covers Supervised Fine-Tuning (SFT), the fast, data-efficient
> stage that converts a pre-trained base into an assistant, by training on (instruction, response)
> pairs with the loss masked to response tokens only.
