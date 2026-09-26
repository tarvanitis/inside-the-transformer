# The Full Architecture and Modern Trends

Chapters 3 through 8 built a transformer one piece at a time: tokens, embeddings, position, attention,
the block, depth, and generation. This chapter assembles those pieces into one view of a complete
model, then steps back to survey where the frontier is heading: Mixture-of-Experts, reasoning models,
and the hybrid state-space architectures now challenging the pure transformer.

**In this chapter**

- The decoder-only architecture, end to end, and why it beat the original encoder-decoder design.
- Mixture-of-Experts: growing capacity without growing per-token cost.
- Scaling laws, and the newer axis of test-time compute.
- The "modern transformer" stack, reasoning models, state-space hybrids, and where the field stands in 2026.

This chapter draws together everything in Part II.

## The decoder-only architecture

Here is a full modern model, a decoder-only transformer, from input tokens to next-token
probabilities:

::: {.figure}
![](assets/figures/ch09/fig-decoder-only.svg)
:::

::: {.caption}
**Figure 9.1.** The modern decoder-only architecture, end to end.
:::

::: {.callout .note}
**RoPE is not a separate pre-block step.** Unlike the sinusoidal encodings that are added once to
the embeddings, RoPE is applied *inside* every attention layer: it rotates the query and key
vectors before the dot product. The diagram omits it at the top level because it happens inside
each attention sub-block.
:::

The original 2017 transformer had two stacks. An **encoder** read the whole input
bidirectionally and a **decoder** generated output while attending back to the encoder via
*cross-attention*:

::: {.figure}
![](assets/figures/ch09/fig-encoder-decoder.svg)
:::

::: {.caption}
**Figure 9.2.** The original 2017 encoder-decoder transformer: two stacks joined by cross-attention.
:::

Modern LLMs drop the encoder entirely. **Decoder-only** won for general language models because it is
simpler (one stack, not two), unified (the same next-token objective everywhere), flexible (generation,
classification, and QA all fall out of prompting), and, empirically, it scales better with compute.
Encoder-decoder models still lead where there's a distinct input to digest and transform:
translation, summarization, speech-to-text (T5, BART, Whisper).

### A complete forward pass

For "The cat sat", predicting the next token:

```text
1. Tokenize:     "The cat sat" → [464, 3797, 3332]
2. Embed:        look up each ID → (3, d_model)
3. × N layers:   x = x + Attention(RMSNorm(x))     # GQA
4.               (RoPE is applied to Q and K inside the attention call)
                 x = x + FFN(RMSNorm(x))            # SwiGLU
5. Final norm:   x = RMSNorm(x)
6. Project:      logits = x[-1] @ W_unembed         # (V,)
7. Sample:       softmax(logits / τ) → next token   # " on"
8. Repeat:       append the token, go back to step 3
```

## Mixture of Experts

The largest models no longer use one feed-forward network per layer. **Mixture-of-Experts (MoE)**
replaces it with many smaller expert FFNs and a learned **router** that sends each token to only the
top-k experts: k = 2 in Mixtral, but k = 8 in the newer fine-grained designs like DeepSeek-V3 and
Qwen3, which use many more, smaller experts:

```text
Token: "programming"
router scores: [0.3, 0.1, 0.8, 0.2, 0.7, 0.1, 0.05, 0.15]   (experts 0–7)
top-2 → experts 2 and 4 → output = 0.53·Expert2(token) + 0.47·Expert4(token)
```

::: {.figure}
![](assets/figures/ch09/fig-moe-routing.svg)
:::

::: {.caption}
**Figure 9.3.** Mixture-of-Experts routing. A learned router scores the experts for each token and sends it to only the top few (here 2 of 8), whose outputs are blended. Most experts stay idle, so total capacity grows without growing the per-token cost.
:::

The payoff is **decoupling capacity from cost**: total parameters can be enormous, but only a fraction
fire per token, so you get much of a huge model's quality at a smaller model's compute.

| Model | Total params | Active params | Experts |
|---|---|---|---|
| Mixtral 8×7B (2023) | 47B | 13B | 8 |
| DeepSeek-V3 / R1 | 671B | 37B | 256 |
| Llama 4 Maverick | ~400B | 17B | 128 |
| Qwen3-235B-A22B | 235B | 22B | 128 |

::: {.caption}
**Table 9.1.** Total and active parameters in production Mixture-of-Experts models. Appendix B
carries the fuller specification tables.
:::

::: {.callout .plain}
MoE is like having specialists instead of generalists. A math question goes to the math experts, a
poetry question to the language experts. You get specialist depth without paying for every specialist
to weigh in on every token.
:::

::: {.callout .note}
**MoE is the frontier default (as of 2026).** Where GPT-3 and Llama 2 were dense, the sparse-MoE
recipe (popularized by Mixtral, pushed further by DeepSeek's *shared* and *fine-grained* experts) is
now standard for the largest open-weight models and widely believed to underlie the leading closed
ones. The cost is engineering complexity: routing must be load-balanced so experts don't collapse to a
few favorites, and sparse all-to-all communication is harder to serve than a dense model.
:::

## Scaling laws, and test-time compute

Kaplan et al. (2020) established that performance scales as a smooth power law in parameters, data,
and compute, and recommended spending most of a growing budget on parameters. The Chinchilla work
(Hoffmann et al., 2022) corrected the allocation: a compute-optimal model wants roughly **20 tokens of
data per parameter**. By that measure many early models, GPT-3 included, were badly undertrained.

::: {.callout .idea}
Doubling parameters is only as valuable as doubling data alongside them. This is why the trend turned
toward *smaller models trained on far more tokens* (Llama 3 8B saw 15 trillion) rather than ever-bigger
undertrained ones.
:::

Since 2024 a second axis has opened up: **test-time (inference) compute**. Instead of only spending
more before deployment, a model can spend more *per query*, generating a long internal chain of
thought before answering. Accuracy on hard, checkable problems (math, code, planning) climbs smoothly
with the number of reasoning tokens spent, often more cheaply than training a bigger model would. The
training recipe behind this is covered in Part IV.

::: {.figure}
![](assets/figures/ch09/fig-test-time-compute.svg)
:::

::: {.caption}
**Figure 9.4.** Test-time compute. On hard, checkable problems, letting a model generate a longer chain of thought before answering raises accuracy, a second axis for spending compute beyond training a bigger model.
:::

## The modern transformer stack

A dense-transformer design from 2019 and a frontier model from 2025 differ on nearly every line:

| Component | Old default | New default |
|---|---|---|
| Activation | ReLU / GELU | **SwiGLU** (Ch 6) |
| Normalization | LayerNorm | **RMSNorm** (Ch 6) |
| Positional encoding | sinusoidal / learned | **RoPE** (Ch 4) |
| Attention variant | multi-head | **GQA** (MLA in DeepSeek) (Ch 5) |
| Norm position | post-norm | **pre-norm** (Ch 6) |
| Sparsity | dense | **MoE** at the frontier |
| Attention kernel | naive `T×T` | **FlashAttention-3** (Ch 8) |

::: {.caption}
**Table 9.2.** The modern transformer stack, component by component, against the 2017 defaults.
:::

Together, **RMSNorm + SwiGLU + RoPE + GQA** define the "modern transformer". Expect all four on any
2025-era model card, increasingly with **QK-normalization** added for training stability at scale:
applying RMSNorm to the query and key vectors before the dot product, which stops attention scores
drifting to extreme values during very large training runs.

## Beyond the standard recipe

**Reasoning models.** The biggest capability shift since 2017 is not architectural but a training one.
Building on the test-time compute above, models are taught (largely by reinforcement learning with
*verifiable* rewards) to spend that budget on a long, self-correcting chain of thought, and every
major lab now ships such a "thinking" mode. The training recipe is the subject of Part IV (Chapter 16).

**State-space models (Mamba).** A different architecture replaces attention with a recurrent
state-space model, reaching *linear* complexity in sequence length by carrying a compressed state
rather than a growing KV-cache. Its weakness is exact recall of distant tokens (attention can always
look back, whereas an SSM must have stored it). Pure SSMs haven't displaced attention, but **hybrid
Mamba-transformer** models (mostly cheap SSM layers with a minority of full-attention layers) are now
a real production class (Jamba, NVIDIA Nemotron-H, IBM Granite 4). The attention layers preserve
recall. The SSM layers cut cost.

**And more:** **speculative decoding** accelerates inference by having a small draft model propose
tokens that the large model verifies in parallel (2–3× speedups, identical output). **Ring attention**
shards a single sequence across many GPUs so the context can outgrow one device's memory.
**Infini-attention** takes a different route, compressing old keys and values into a fixed-size memory
so context length stops costing extra memory at all.

::: {.callout .note}
**The frontier stack (as of 2026).** The field has converged on a shared recipe: a decoder-only
transformer with RMSNorm, SwiGLU, RoPE, and GQA, increasingly MoE-sparse at the top end, and
increasingly reasoning-capable, while competing hard on price-to-performance. Open-weight models
(DeepSeek, Llama 4, Qwen3, Gemma) now trail the closed frontier (GPT-5, Claude, Gemini, Grok) by
months, not years. Treat specific version numbers as a dated snapshot. The *architecture* underneath
has been far more stable than the names on top of it.
:::

## The end-to-end mental model

Three lenses, one last time: the whole model in a paragraph each.

::: {.callout .lens}
**Geometric.** Text begins as discrete tokens. Embeddings map them to points in a high-dimensional
space. Each layer *moves* those points toward regions that reflect their contextual meaning, routing
information between them by geometric alignment (dot products), rotating each by its position, and
nudging them incrementally (residuals) rather than teleporting. By the final layer, each point sits
where its full contextual meaning lives, and the vocabulary projection asks "which word is closest?"
:::

::: {.callout .lens}
**Representation.** A token starts as an island of meaning. Attention gathers context from its
neighbors. The FFN builds new features and retrieves stored knowledge. Depth repeats this at rising
levels of abstraction. By the end, the representation is no longer a token but a compressed encoding of
a *linguistic situation* (this token, in this context, in this role), and the output is a distribution
over what best continues it.
:::

::: {.callout .lens}
**Mechanistic.** The model is a stack of interacting circuits: induction heads copy patterns, FFN
neurons fire like key-value memories, the residual stream carries information across layers, the
KV-cache stores the past, and FlashAttention streams the computation. Not a smooth statistical
estimator: a collection of interpretable algorithms running in parallel.
:::

## Summary

- A modern LLM is a **decoder-only** transformer: embed + position, then N pre-norm blocks
  (attention + FFN with residuals), a final norm, and a vocabulary projection.
- **Mixture-of-Experts** decouples capacity from per-token cost and is now the frontier default.
  **Scaling laws** say data must grow with parameters, and **test-time compute** adds a new axis.
- The modern stack is **RMSNorm + SwiGLU + RoPE + GQA**, increasingly MoE-sparse and reasoning-capable.
  **Hybrid state-space** models are the main efficiency-driven alternative.

> A transformer encodes the statistical structure of language in the geometry of a vector space:
> "patterns" are directions, and "matching" is a dot product. Everything in Part II (attention, the
> block, depth, generation) is machinery for building and reading that geometry.

> **Coming up:** We've understood the transformer inside and out. Part III makes it concrete: in
> Chapter 10 we build a working GPT from scratch in PyTorch, turning every idea from these chapters
> into runnable code.
