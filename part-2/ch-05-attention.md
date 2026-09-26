# Attention: The Core Mechanism

By now each token is a vector that carries both its meaning (Chapter 3) and its position (Chapter 4).
But so far the tokens sit side by side without interacting. **Self-attention** is where they finally
talk to each other, where "it" figures out what it refers to, where a verb finds its subject, where
"bank" learns whether it is beside a river or holding your savings. It is the mechanism at the heart
of the transformer, and this chapter builds it from the dot product up.

**In this chapter**

- The job attention does: integrating context.
- Queries, keys, and values, and why a token needs all three.
- The five steps of scaled dot-product attention, worked end to end on real numbers.
- Multi-head attention, and the grouped/multi-query variants that make it cheap to serve.

This chapter leans on the dot product, softmax, and the √d_k scaling factor from Chapter 2.

::: {.figure}
![](assets/figures/ch05/fig-attention-flow.svg)
:::

::: {.caption}
**Figure 5.1.** Scaled dot-product attention, from input vectors to weighted output.
:::

## The job attention does

Every token needs a representation that captures *its role in this context*, not just its dictionary
meaning. Looking up meaning is what embeddings already do. Attention solves a different problem —
*integrating context*. It does so by letting every token ask: **which other tokens are most relevant
to understanding me, and what should I take from them?**

## Query, key, value: three roles

That question splits in two: *matching* (who is relevant to me?) and *information transfer* (what
should I take from them?). Because these are different jobs, the model derives three separate vectors
from each token:

| Vector | Role | The question it answers |
|---|---|---|
| **Query (Q)** | what am I looking for? | "I need a noun I can refer to" |
| **Key (K)** | what do I advertise? | "I am a singular, animate noun" |
| **Value (V)** | what do I provide? | the token's full semantic content |

::: {.caption}
**Table 5.1.** Query, key and value: the question each vector answers.
:::

::: {.callout .plain}
A search-engine analogy: the **query** is what you type into the search box, each **key** is a page's
title and keywords, and each **value** is the page's actual content. Attention compares your query
against every key to score relevance, then uses those scores to pull content from the values.
:::

::: {.callout .lens}
**Mechanistic.** It mirrors how a database works: the query is the `WHERE` clause, the key is the
indexed column, the value is the row's data. You don't search on the full row, and you don't return
the index. Separating the lookup key from the payload is a basic systems-design principle.
:::

### Building Q, K, V

Q, K, and V come from multiplying each token's embedding by three separate **learned** weight matrices
`W_Q`, `W_K`, `W_V`, using the matrix multiplication from Chapter 2. One token, projected three ways:

```python
# x = input embeddings, shape (batch, seq_len, d_model) = (1, 3, 768)
W_Q = nn.Linear(d_model, d_model)   # learned
W_K = nn.Linear(d_model, d_model)
W_V = nn.Linear(d_model, d_model)
Q, K, V = W_Q(x), W_K(x), W_V(x)    # each (1, 3, 768)
```

Everything the mechanism does is captured by one formula, which the next five steps unpack:

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

::: {.callout .plain}
Reading it from the inside out: `Q·Kᵀ` compares every query against every key to make a grid of
relevance scores. Dividing by `√d_k` keeps those scores in a sensible range. `softmax` turns each row
into weights that add up to 1, and the final `·V` uses those weights to blend the value vectors into
each token's new, context-aware representation.
:::

## The five steps, worked end to end

We'll run the whole pipeline on a tiny example: four tokens, with query/key vectors of length
`d_k = 3`. The value vectors are only 2-dimensional here, to keep the final arithmetic short. In a
real model they are the same width as the keys. Here are the (already projected) Q, K, and V:

```text
Q (queries)     K (keys)        V (values)
[ 1  0  1 ]     [ 1  1  0 ]     [ 0.2  0.8 ]
[ 0  1  0 ]     [ 0  1  1 ]     [ 0.5  0.3 ]
[ 1  1  0 ]     [ 1  0  1 ]     [ 0.1  0.9 ]
[ 0  1  1 ]     [ 0  1  0 ]     [ 0.4  0.6 ]
```

### Step 1 (Scores (Q·Kᵀ))

Relevance between a query and a key is their dot product. Multiplying `Q` by `Kᵀ` computes all pairs
at once: entry `(i, j)` is "how relevant is token *j* to token *i*?"

```text
Q·Kᵀ            key0  key1  key2  key3
   query0  [     1     1     2     0   ]     e.g. query0·key2 = [1,0,1]·[1,0,1] = 2
   query1  [     1     1     0     1   ]
   query2  [     2     1     1     1   ]
   query3  [     1     2     1     1   ]
```

::: {.callout .lens}
**Geometric.** The dot product measures alignment: two vectors pointing the same way score high, and
perpendicular ones score near zero. High dot product means high relevance, the standard way to
measure similarity in a vector space.
:::

### Step 2 (Scale by √d_k)

Divide every score by `√d_k` (here `√3 ≈ 1.73`):

```text
scaled = Q·Kᵀ / √3
[ 0.58  0.58  1.15  0.00 ]
[ 0.58  0.58  0.00  0.58 ]
[ 1.15  0.58  0.58  0.58 ]
[ 0.58  1.15  0.58  0.58 ]
```

Without this, scores grow as `√d_k` (the variance argument is in Chapter 2). With `d_k = 128`,
typical scores are around 11× larger than with `d_k = 1`, and one score can tower over the rest.

::: {.callout .caution}
**Softmax saturation.** If the scores are too large, the next step (softmax) collapses onto a single
token, and its gradient goes to zero, so the model stops learning how to attend. Dividing by `√d_k`
keeps the scores in a range where softmax stays smooth and trainable. For a 12-head model with
`d_k = 64`, the factor is `√64 = 8`.
:::

### Step 3 (Causal masking)

In an autoregressive model, one that generates text left to right, one token at a time, token *i* must
not see the future (positions *j > i*), or it could simply copy the answer. We enforce this by setting the future scores to −∞ before softmax, so they receive
zero weight:

```text
after masking (upper triangle → −∞)
[ 0.58   −∞     −∞     −∞  ]     token 0 sees only itself
[ 0.58  0.58   −∞     −∞  ]     token 1 sees 0–1
[ 1.15  0.58  0.58    −∞  ]     token 2 sees 0–2
[ 0.58  1.15  0.58  0.58  ]     token 3 sees 0–3
```

::: {.callout .plain}
Like writing a story one word at a time: when you choose the next word you may reread everything so
far, but you can't peek at words you haven't written yet.
:::

### Step 4 (Softmax)

Softmax (Chapter 2) turns each row of scores into attention weights that are positive and sum to 1.
The −∞ entries become 0:

```text
attention weights (each row sums to 1)
[ 1.00  0.00  0.00  0.00 ]     token 0: all on itself
[ 0.50  0.50  0.00  0.00 ]     token 1: evenly split over 0–1
[ 0.47  0.26  0.26  0.00 ]     token 2: mostly on token 0
[ 0.21  0.37  0.21  0.21 ]     token 3: mostly on token 1
```

::: {.figure}
![](assets/figures/ch05/fig-attention-steps.svg)
:::

::: {.caption}
**Figure 5.2.** The five-step pipeline on the worked example, shown as grids. Scaled scores (left) are causal-masked so no token sees the future (middle), then softmax turns each query's row into weights that sum to 1 (right).
:::

::: {.callout .lens}
**Mechanistic.** Softmax does three things at once: it normalizes (each row becomes a probability
distribution), it introduces competition (raising one weight lowers the others), and it amplifies
differences exponentially. That makes it a *differentiable* stand-in for "pick the most relevant",
`argmax`-like, but smooth enough to train.
:::

### Step 5 (Weighted sum of values)

Finally, each token's output is the weighted blend of the value vectors, using its row of weights:

```text
weights            V              output = weights · V
[1.00 0 0 0 ]                     [ 0.20  0.80 ]   ← copies V[0]
[0.50 0.50 0 0] × [ 0.2  0.8 ] =  [ 0.35  0.55 ]   ← average of V[0], V[1]
[0.47 0.26 0.26 0] [ 0.5  0.3 ]   [ 0.25  0.69 ]
[0.21 0.37 0.21 0.21][ 0.1 0.9 ]  [ 0.33  0.59 ]
                   [ 0.4  0.6 ]
```

::: {.callout .lens}
**Representation.** This is where context gets integrated. Before attention, each token's vector
reflects its meaning in isolation. After attention, it reflects its meaning *in this sentence, given
these neighbors*. Information has been routed from the relevant tokens into the current one.
:::

### The cost: quadratic in sequence length

Notice what Step 1 produced: a full `T × T` grid, one score for every pair of tokens. That is the
price of letting every token see every other token. Double the sequence and you quadruple the scores.
A thousand tokens means a million scores. A hundred thousand tokens means ten billion.

::: {.callout .idea}
Attention's cost grows with the **square** of the sequence length, because the score matrix has one
entry per pair of tokens. This single fact drives an enormous amount of engineering: the KV-cache and
FlashAttention (Chapter 8) attack the constant factor, while sliding-window attention and state-space
models (Chapter 9) attack the exponent itself.
:::

## Multi-head attention

A single attention computation produces one pattern of who-attends-to-whom. But language has many
kinds of relationship at once (a pronoun and its antecedent, a verb and its subject, nearby words,
matching brackets) and softmax tends to concentrate on just one or two positions. One head can't
track all of that, so transformers run `h` attention computations **in parallel**, each with its own
`W_Q`, `W_K`, `W_V`:

```text
Head 1: subject–verb structure
Head 2: semantic similarity
Head 3: nearby context
Head 4: quotes, lists, brackets      … etc.
```

::: {.figure}
![](assets/figures/ch05/fig-multihead.svg)
:::

::: {.caption}
**Figure 5.3.** Multi-head attention. Each head sees the whole input and learns its own attention pattern (subject–verb, similarity, nearby words, brackets). Their outputs are concatenated and mixed by W_O into one result.
:::

The heads run in smaller subspaces of size `d_k = d_model / h`, the same `d_k` as in the scaling
factor, which is why the division by `√d_k` uses the *per-head* dimension (128 below), not the full
`d_model`. Their outputs are concatenated, and a final projection `W_O` mixes them back to `d_model`:

| Parameter | Typical | Meaning |
|---|---|---|
| `d_model` | 4096 | total embedding dimension |
| `h` | 32 | number of attention heads |
| `d_k` | 128 | dimension per head (`d_model / h`) |

::: {.caption}
**Table 5.2.** Multi-head attention dimensions for a typical 4096-wide model. Appendix B lists
these dimensions for real models across the last eight years.
:::

::: {.callout .caution}
**A common misconception.** It's tempting to think the input vector is *sliced* into `h` pieces, one
per head. It isn't. Every head sees the *whole* input `X` and re-encodes it through its **own** full
projection matrices: same data, a different learned view of it.
:::

::: {.callout .lens}
**Geometric.** Think of each head as viewing the same data through its own coordinate system: its own
learned rotation and scaling of the embedding space, and therefore its own notion of similarity, with one
head by syntactic role, another by topic, another by proximity. Softmax is applied *per head*, so each
keeps an independent attention pattern rather than collapsing toward the others.
:::

The tensor bookkeeping, for reference:

```python
X.shape = (B, T, d_model)                       # batch, sequence, embedding
Q = (X @ W_Q).view(B, T, h, d_k).transpose(1, 2)      # (B, h, T, d_k)  — same for K, V
scores  = Q @ K.transpose(-2, -1)               # (B, h, T, T)
scores  = scores.masked_fill(causal_mask, float("-inf"))   # Step 3 — never skip this
weights = F.softmax(scores / math.sqrt(d_k), dim=-1)
out = (weights @ V).transpose(1, 2).reshape(B, T, d_model)
out = out @ W_O                                 # (B, T, d_model)
```

## Grouped-query and multi-query attention

Multi-head attention has an inference cost that only shows up later. When a model generates text one
token at a time, it caches the keys and values of every previous token, the **KV-cache** (Chapter 8),
so it doesn't recompute them. With many heads, many layers, and a long context, that cache can run
to tens of gigabytes.

**Grouped-Query Attention (GQA)** cuts it down by letting several query heads *share* one key/value
head:

```text
Multi-head (8 heads):     Q₁…Q₈   K₁…K₈   V₁…V₈        (8 K/V heads)
Grouped-query (8 Q, 2 KV): Q₁Q₂Q₃Q₄ | Q₅Q₆Q₇Q₈
                              K₁,V₁  |   K₂,V₂          (2 K/V heads, shared)
```

::: {.figure}
![](assets/figures/ch05/fig-kv-sharing.svg)
:::

::: {.caption}
**Figure 5.4.** Sharing key/value heads. Multi-head attention gives every query head its own key/value. Grouped-query lets several query heads share one key/value pair, and multi-query shares a single pair. Fewer key/value heads mean a smaller KV-cache.
:::

The KV-cache shrinks in proportion (4× here), inference speeds up on long contexts, and quality
barely moves because the model trains with the sharing in place.

| Model | Query heads | KV heads | Ratio |
|---|---|---|---|
| Llama 2 70B | 64 | 8 | 8 : 1 |
| Llama 3.1 8B | 32 | 8 | 4 : 1 |
| Llama 3.1 70B | 64 | 8 | 8 : 1 |
| Mistral 7B | 32 | 8 | 4 : 1 |

::: {.caption}
**Table 5.3.** Query and KV head counts across models using grouped-query attention.
:::

::: {.callout .note}
**MQA and MLA.** *Multi-Query Attention* is GQA taken to the limit: all query heads share a single
key and value head (maximum saving, a little quality lost). *Multi-head Latent Attention* (MLA, in
DeepSeek-V2/V3) instead compresses keys and values into a low-dimensional latent space and expands
them on use, shrinking the cache while keeping expressiveness.
:::

## Summary

- Attention integrates context: each token pulls information from the tokens most relevant to it.
- It derives a **query**, **key**, and **value** from every token, so matching happens between queries
  and keys, information flows from the values.
- Scaled dot-product attention is five steps: score (`Q·Kᵀ`), scale (`÷√d_k`), causal-mask, softmax,
  and weighted sum of values: `softmax(Q·Kᵀ/√d_k)·V`.
- **Multi-head** attention runs many of these in parallel, each in its own learned subspace, so the
  model tracks several kinds of relationship at once.
- **GQA / MQA / MLA** share or compress keys and values to keep the inference-time KV-cache affordable.

> **Coming up:** Attention moves information *between* tokens, but a token also needs to *process*
> what it has gathered on its own. That's the other half of the transformer block: the feed-forward
> network, together with the residual connections and normalization that make deep stacks trainable.
> Chapter 6 assembles the full block.
