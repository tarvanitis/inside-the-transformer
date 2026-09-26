# Generating Text: Inference

We've built the model and looked at what forms inside it. Now for what it was built to do: **generate
text**. A trained transformer produces one token at a time, feeding each token it emits back in as
input for the next, a loop called *autoregressive generation*. This chapter traces one step of that
loop from the final hidden vector to a chosen token, then covers the engineering (the KV-cache and
FlashAttention) that makes it fast enough to be useful.

**In this chapter**

- The generation loop, end to end.
- How the final vector becomes scores over the whole vocabulary (the LM head).
- Turning scores into a token: softmax, temperature, and sampling.
- Making it fast: the KV-cache, FlashAttention, and the context window.

This chapter uses softmax, temperature, and top-p from Chapter 2, and the attention machinery from
Chapters 5–6.

::: {.figure}
![](assets/figures/ch08/fig-generation-loop.svg)
:::

::: {.caption}
**Figure 8.1.** The generation loop: forward pass, sample, append, repeat.
:::

## The generation loop

Each pass through the model produces a prediction for just *one* position: the next token. That token
is appended to the sequence, and the whole thing runs again. Prompt → forward pass → logits → sample →
append → repeat, until the model emits an end-of-sequence token or hits a length limit.

::: {.callout .idea}
Stripped to its core, next-token prediction is a **nearest-match search**: compare one
`d_model`-dimensional vector (the model's current state) against the embedding vectors of all `V`
tokens in the vocabulary, and see which it is closest to. Everything below is how that comparison is computed.
:::

## From the final vector to logits

After the last transformer layer, we take the hidden state of the **last** token, a single
`d_model`-dimensional vector that summarizes everything the model has gathered about what should come
next:

```python
last_hidden = final_hidden_states[:, -1, :]     # (1, d_model), e.g. (1, 768)
```

To score every possible next token, we multiply that vector by the **LM head** (also called the
*unembedding* matrix), of shape `(d_model, V)`. The result is `V` raw scores, the **logits**:

```python
logits = last_hidden @ W_unembed        # (1, 768) @ (768, 50000) = (1, 50000)
# logits[i] = last_hidden · embedding_of_token_i
```

Each logit is exactly a **dot product** (Chapter 2) between the context vector and one token's
embedding: 50,000 similarity measurements in a single matrix multiply.

::: {.callout .lens}
**Geometric.** The vocabulary projection is a *nearest-neighbor search* in embedding space. Since
`a·b = ‖a‖‖b‖cos θ`, a large positive logit means the context vector and that token's embedding point
the same way (θ ≈ 0°). Near-zero means unrelated (θ ≈ 90°). Training shapes the space so that the
tokens which fit a context sit close to that context's representation.
:::

Many models **tie weights**: the same matrix is the embedding table (token → vector) on the way
in and, transposed, the LM head (vector → scores) on the way out. The two are mirror images of each
other (one reads a vector out of the table, the other measures how well a vector matches each row)
and sharing them saves `V × d_model` parameters (about 525 million at Llama 3 8B's vocabulary and
width) while keeping input and output representations in the same space. GPT-2, Gemma, and the small
Llama 3.2 models tie their weights, while Llama 3 8B itself keeps a separate head. (Chapter 3
introduced the embedding matrix.)

## From logits to a token

Logits aren't probabilities. **Softmax** (Chapter 2) turns them into a distribution over the
vocabulary, and **temperature** rescales the logits first (`softmax(logits / τ)`) to make the model
more or less confident. `τ → 0` approaches greedy `argmax`. `τ > 1` flattens the distribution. From
that distribution we pick a token:

- **Greedy.** Always take the highest-probability token. Deterministic, and prone to repetition.
- **Temperature sampling.** Sample from the (rescaled) distribution.
- **Top-k.** Sample only from the k most likely tokens.
- **Top-p (nucleus).** Sample from the smallest set whose cumulative probability ≥ p, an adaptive
  cutoff (Chapter 2).

The chosen token is appended, and the loop repeats. This is what *autoregressive* means.

### A worked example

Running "The cat sat" through the model and reading out the top predictions, weight-tied LM head and
all:

```python
last_hidden = final_hidden_states[:, -1, :]     # (1, 768)  — "…what comes after 'The cat sat'?"
logits = last_hidden @ embedding.weight.T       # (1, 50000) — weight tying: reuse the embedding table
probs  = softmax(logits)
```

```text
top predictions (logit → probability):
  " on"    logit 8.52   cos 0.78   →  P = 42%     ← most similar to the context vector
  " down"  logit 7.22   cos 0.65   →  P = 11%
  " and"   logit 6.80              →  P =  8%
  (the remaining ~39% is spread across the other 49,997 tokens)
```

→ the model continues "The cat sat **on**", appends it, and predicts again.

::: {.callout .idea}
No hand-coded grammar, no rules engine. Just matrix multiplications, dot products, and
nonlinearities, repeated. Yet from optimizing a single objective (predict the next token) over a huge
corpus, the model acquires syntax, semantics, world knowledge, and reasoning patterns.
:::

## Making generation fast

A naive implementation recomputes everything at every step. Two ideas make production inference
practical.

### The KV-cache

::: {.callout .plain}
Generation really has two phases. **Prefill** is the first pass, where the model reads your whole
prompt at once, all the tokens together, in parallel, which GPUs are very good at. **Decode** is
everything after: one token at a time, each one waiting on the last. Prefill is fast and
compute-hungry. Decode is slow and spends most of its time waiting on memory. The KV-cache is what
makes decode bearable.
:::

To generate token `t+1`, attention needs the keys and values of all tokens `1…t`. But those never
change once computed, so recomputing them each step is O(T²) wasted work. The **KV-cache** stores each
token's key and value the first time they're computed, so each new step only computes K and V for the
*new* token and reads the rest from the cache:

| | compute per step | total for T steps |
|---|---|---|
| without cache | O(T²·d_model) | O(T³·d_model) |
| with KV-cache | O(T·d_model) | O(T²·d_model) |

::: {.caption}
**Table 8.1.** Generation cost with and without a KV-cache.
:::

Without a cache, step *t* re-runs the whole forward pass over all *t* tokens so far. With the cache,
it projects one new token and attends over *t* stored keys. The cache removes a factor of T. It does
not make generation linear.

::: {.figure}
![](assets/figures/ch08/fig-kv-cache.svg)
:::

::: {.caption}
**Figure 8.2.** Why the KV-cache helps. Without it, each step recomputes the keys and values for every earlier token (O(T²) work). With it, past keys and values are stored and reused, so each step computes only the new token (O(T)).
:::

::: {.callout .lens}
**Mechanistic.** The KV-cache is a *persistent external memory* for attention: each step's query reads
from the accumulated store rather than recomputing it, the transformer's version of reading from
cache instead of recomputing from scratch.
:::

The cache is large, though, and reading it becomes the new bottleneck (memory bandwidth, not compute).
A rough size: `2 × layers × kv_heads × head_dim × bytes` ≈ 2 × 80 × 8 × 128 × 2 ≈ **320 KB per token**
or about **33 GB** for a 100K-token context. That is why **GQA/MQA** share K/V heads and **MLA**
compresses them (Chapter 5), and why servers like vLLM manage the cache in pages.

### FlashAttention

Standard attention writes the full `T × T` score matrix to GPU memory. For `T = 8192` that's 128 MB
**per head** in fp16, so a 32-head model materializes about 4 GB of scores per forward pass, all of it
moved to and from DRAM. Since GPU compute is far faster than GPU memory bandwidth, attention is
**memory-bound**, not compute-bound. **FlashAttention** (Dao et al., 2022) never materializes the full
matrix: it processes attention in tiles kept in the GPU's fast on-chip SRAM, accumulating the result
with an *online softmax* that updates a running maximum and sum blockwise.

| | memory | passes over data |
|---|---|---|
| naive attention | O(T²) | 3 |
| FlashAttention | O(T) | 1 (streaming) |

::: {.caption}
**Table 8.2.** Memory and data movement: naive attention versus FlashAttention.
:::

The result is *identical* to standard attention (exact, not approximate), with 2–4× speedups for both
training and inference. **FlashAttention-3** (Shah et al., 2024) is tuned for NVIDIA Hopper (H100)
GPUs (warp specialization and FP8) and is the default attention kernel there.

::: {.figure}
![](assets/figures/ch08/fig-flashattention.svg)
:::

::: {.caption}
**Figure 8.3.** FlashAttention. Standard attention writes the full T×T score matrix to slow memory. FlashAttention processes it in tiles kept in fast on-chip memory, combines them with an online softmax, and never stores the whole matrix.
:::

::: {.callout .deepdive}
**Online softmax.** Ordinary softmax needs every score in memory before it can normalize. The online
version keeps a running maximum `m` and running sum and rescales as each block arrives,
`exp(sᵢ − m) / Σ exp(sⱼ − m)`, which is numerically stable and lets attention be computed one tile at
a time.
:::

### Context length, and extending it

The **context window** is the maximum number of tokens the model can attend over. It plays two roles:
in *training* it sets the longest dependency the model can ever learn. At *inference* it sets the
history it can use. The two are linked. A model trained on 4K tokens can't suddenly use 100K, because
it never learned dependencies that long. Pushed past its trained length a model doesn't crash, it
*degrades*: positional patterns it never saw break down and long-range attention becomes unreliable.

Several techniques stretch the window: **RoPE scaling** (YaRN, LongRoPE, Chapter 4) rescales the
rotational frequencies; **long-context fine-tuning** trains on longer sequences; **sliding-window
attention** (Mistral) limits each token to a local window, cutting cost from O(T²) to O(T·window); and
**retrieval-augmented generation** (Chapter 18) fetches only the relevant chunks instead of holding
everything in the window.

::: {.callout .note}
**Long context vs RAG (as of 2026).** Frontier windows are now enormous (1M tokens on several models,
more on some), but long context isn't free: prefill is still quadratic, the KV-cache grows into tens
of gigabytes, latency and cost climb with prompt length, and models can get "lost in the middle,"
missing facts buried mid-context. RAG keeps the prompt small, is cheaper to update, and cites sources
but adds retrieval infrastructure and can miss relevant chunks. In practice they're complementary:
RAG narrows a huge corpus to what matters, a long window reasons over the result. Neither has made the
other obsolete.
:::

## Summary

- Generation is a **loop**: forward pass → logits → sample → append → repeat, one token at a time
  (autoregressive).
- The **LM head** projects the final hidden vector onto every token embedding, producing logits, a
  batch of dot-product **similarity** scores. **Weight tying** reuses the embedding matrix for this.
- **Softmax + temperature** turn logits into a distribution, and a **sampling** strategy (greedy,
  top-k, top-p) picks the token.
- The **KV-cache** cuts a factor of T out of generation (per-step cost drops from O(T²) to O(T)) by
  storing past keys/values, at the cost of large memory reads; **FlashAttention** makes the attention step memory-efficient and fast; and the
  **context window** is bounded by what the model was trained on, though several techniques extend it.

> **Coming up:** We've now followed a transformer from raw text all the way to generated text.
> Chapter 9 steps back to assemble the whole architecture in one view, and surveys where the frontier
> is going: Mixture-of-Experts, and the hybrid state-space models nipping at the transformer's heels.
