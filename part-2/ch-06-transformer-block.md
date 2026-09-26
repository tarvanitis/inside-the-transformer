# The Transformer Block

Chapter 5 gave us attention, the mechanism that lets tokens exchange information. But attention has a
quiet limitation: its output is only ever a *weighted average* of value vectors, which is a linear
operation. It can move features between tokens. It can't build new ones. The rest of the transformer
block supplies what's missing: a feed-forward network that does per-token computation, wrapped in the
two devices (residual connections and normalization) that let the block be stacked dozens of times
without training falling apart. This chapter assembles the full block, then stacks it into depth.

**In this chapter**

- Why attention needs a partner: the feed-forward network.
- Residual connections and normalization, and what makes deep stacks trainable.
- Pre-norm vs post-norm.
- What stacking many blocks buys, and what different depths learn.

This chapter builds on attention (Chapter 5) and the mean, variance, and activation functions from
Chapter 2.

## Why attention alone isn't enough

Attention's output for a token is `Σⱼ αᵢⱼ vⱼ`, a weighted sum of value vectors. That is *linear* in V:
however clever the weights, the result is still a recombination of vectors that already exist. There
is no nonlinearity, and so no way to build a genuinely new feature. Attention routes and blends,
something else has to compute.

## The feed-forward network

After attention, every token passes, independently, through a small two-layer network that expands,
applies a nonlinearity, then compresses. In the classic form (using ReLU, as in the original 2017
paper, though BERT and GPT-2 later swapped in the smoother GELU):

`FFN(x) = max(0, xW₁ + b₁)W₂ + b₂`

```text
x    : (d_model,)
W_1  : (d_model, d_ff)     d_ff ≈ 4 × d_model     — expand
W_2  : (d_ff, d_model)                            — compress
```

::: {.callout .plain}
The FFN expands each token's vector into a wider space, pulls out patterns with a nonlinearity, then
compresses back down. It behaves like a memory bank: mechanistic interpretability work (Chapter 7) shows FFN layers are where a
model stores much of its *factual* knowledge.
:::

::: {.figure}
![](assets/figures/ch06/fig-ffn-shape.svg)
:::

::: {.caption}
**Figure 6.1.** The feed-forward network: each token's vector is expanded to a wider hidden layer (about 4× d_model), passed through a nonlinearity, then compressed back to d_model.
:::

**Why expand by ~4×?** The wide hidden layer is a high-dimensional scratch pad. In low dimensions
features interfere (there isn't room for many near-orthogonal directions). Widening gives the model
space to separate concepts. The 4× factor is empirically about right: 2× is too cramped, 8× costs far
more for little gain.

::: {.callout .lens}
**Mechanistic.** It helps to see the block as two alternating phases: **attention is communication**
(tokens exchange information) and **FFN is computation** (each token then thinks privately and
builds new features from what it received). Those FFN "memories" are examined in Chapter 7.
:::

### Activation functions

The nonlinearity between the two projections is the whole point. Without it, `(xA)B = x(AB)` collapses
the two layers into one. The math of the common choices (ReLU, GELU, SiLU, and the gated **SwiGLU**)
is in Chapter 2. Here it's enough to know that modern LLMs use SwiGLU, whose gate branch decides what
information passes through. Because SwiGLU adds a third matrix, those models set `d_ff ≈ 2.67 × d_model`
to keep the parameter count comparable to a plain 4× FFN.

## Residual connections

Each sub-layer (attention and FFN alike) is wrapped in a **residual connection**: the input is added
straight back onto the output.

```text
x = x + Attention(x)     # not  x = Attention(x)
x = x + FFN(x)           # not  x = FFN(x)
```

This matters for two reasons. First, **gradient flow**: without residuals, the training signal has to
pass through every layer and shrinks toward zero long before it reaches the early ones (a 96-layer
model simply won't train). The skip connection gives the gradient a direct path:
`∂(output)/∂(input) = 1 + ∂(layer)/∂(input)`. That leading `1` is the point. However small the
sub-layer's own gradient becomes, there is always an undiminished route back through the addition, so
the signal can reach layer 1 from layer 96 without being multiplied down to nothing. Second, **information preservation**:
even if a sub-layer contributes nothing, the original input survives (`x + 0 = x`).

::: {.figure}
![](assets/figures/ch06/fig-residual-stream.svg)
:::

::: {.caption}
**Figure 6.2.** The residual stream. Each sub-layer reads the current vector, computes a correction, and adds it back, so the representation is refined step by step rather than replaced.
:::

::: {.callout .lens}
**Geometric.** Picture the **residual stream** as a highway running the length of the model:
`x₀ → x₀ + Attn(x₀) → … + FFN(…) → … → x_final`. Each layer *reads* from the highway, computes a small
correction, and *writes* it back. The representation drifts through embedding space incrementally
rather than being replaced at each step: deep learning as iterative refinement, not teleportation.
This is also why some layers in a trained model are nearly identity functions: they've learned a
near-zero correction.
:::

## Normalization: LayerNorm and RMSNorm

As values flow through many layers they can blow up or shrink toward zero, which destabilizes
training. **Normalization** resets each token's vector to consistent statistics. **Layer
normalization** centers and rescales using the mean and variance from Chapter 2, then applies a
learned gain `γ` and bias `β`:

$$\mathrm{LayerNorm}(x) = \gamma\,\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

```text
x          = [4.0, 2.0, 6.0, 0.0]
mean μ     = 3.0,   variance σ² = 5.0,   std ≈ 2.24
normalized = (x − 3.0) / 2.24 = [0.45, −0.45, 1.34, −1.34]
```

**RMSNorm**, the default in Llama, Mistral, Gemma, and Qwen, is a cheaper variant that drops the
mean-centering and the bias, dividing only by the root-mean-square:

$$\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\mathrm{mean}(x^2)}}\;\gamma$$

It costs less per layer and trains about as stably, which is why frontier models adopt it.

::: {.callout .plain}
Normalization is the volume knob on a microphone: whether the incoming signal is a whisper or a shout,
it re-centers and rescales it to a standard level before passing it on.
:::

## Pre-norm vs post-norm

Where the normalization sits turns out to matter a lot:

```text
x = LayerNorm(x + Attention(x))     # post-norm  (original 2017 paper)
x = x + Attention(LayerNorm(x))     # pre-norm   (modern default)
```

Post-norm makes the gradient pass through the full sub-layer *before* being normalized, which
destabilizes very deep networks. Pre-norm guarantees every sub-layer receives an already-normalized
input, so gradients flow cleanly regardless of depth, warm-up becomes far less critical, and models
scale to hundreds of layers. Pre-norm is the modern default, though some models (Gemma 2 and 3, for
instance) normalize both before *and* after each sub-layer for extra stability. Note that production
LLMs still use a learning-rate warm-up schedule (Chapter 13). Pre-norm reduces the need for it rather
than removing it.

::: {.figure}
![](assets/figures/ch06/fig-pre-post-norm.svg)
:::

::: {.caption}
**Figure 6.3.** Pre-norm vs post-norm. Post-norm normalizes after the residual add. Pre-norm normalizes each sub-layer's input while the skip carries the un-normalized vector, which keeps gradients clean in very deep stacks.
:::

## The block, assembled

Putting the pieces together, one pre-norm transformer block is: normalize, attend, add, then
normalize, feed-forward, add.

::: {.figure}
![](assets/figures/ch06/fig-block.svg)
:::

::: {.caption}
**Figure 6.4.** One pre-norm transformer block, with both residual paths shown.
:::

## Depth: stacking the blocks

One block lets every token interact once and think once. That isn't enough, for three reasons:
higher-level features can only be detected once lower-level ones exist, so abstraction needs
**hierarchy**. Each block adds a round of nonlinearity, so stacking **composes** more complex
functions, and depth is **iterative refinement**: a pattern missed by one layer can be caught by a
later one.

::: {.callout .plain}
Layers are like workers on an assembly line: each does what it can with what it receives, and if one
misses something, the next can still catch it.
:::

Interpretability studies find a consistent division of labor by depth:

| Depth | What tends to be learned |
|---|---|
| first ~⅓ | syntax (part-of-speech, subject–verb), local patterns (n-grams, punctuation), surface form |
| middle ~⅓ | semantics (coreference), entity recognition, phrase-level meaning |
| final ~⅓ | task-specific reasoning, factual associations (Paris → France), output formatting |

::: {.caption}
**Table 6.1.** What layers at different depths tend to learn.
:::

::: {.callout .lens}
**Mechanistic.** Individual heads specialize too: some always attend to the previous token, some track
long-range coreference, some copy particular token types. None of this is programmed. It emerges from
training.
:::

Typical depths:

| Model | Layers |
|---|---|
| GPT-2 Small | 12 |
| GPT-3 175B | 96 |
| Llama 2 7B | 32 |
| Llama 2 70B | 80 |

::: {.caption}
**Table 6.2.** Layer counts across model scales.
:::

## Summary

- Attention is linear in its values, so it recombines features but can't build new ones. The
  **feed-forward network** supplies the per-token nonlinear computation (expand → activate → compress),
  and is where much factual knowledge lives.
- **Residual connections** add each sub-layer's input back onto its output, giving gradients a clean
  path and preserving information: the "residual stream" that runs the length of the model.
- **Normalization** (LayerNorm, and the cheaper **RMSNorm**) keeps activations well-scaled,
  **pre-norm** placement is what lets models go hundreds of layers deep.
- **Stacking** blocks builds hierarchy and refines representations. Early layers learn syntax, middle
  layers semantics, and late layers task and factual structure.

> **Coming up:** We've now built the whole stack, but what actually *forms* inside it once it's
> trained? Chapter 7 opens the trained model up: the circuits, induction heads, and FFN "memories"
> that interpretability research has found doing identifiable work.
