# From Text to Numbers: Tokenization and Embeddings

Chapter 2 gave us the operations a transformer runs on: vectors, matrices, dot products. But a model
can't multiply a sentence. Its very first job is to turn text into the numbers those operations need.
This chapter covers the two steps that do it: **tokenization** (splitting text into a fixed set of
symbols) and **embedding** (turning each symbol into a meaningful vector).

**In this chapter**

- Why text is split into *subword* tokens, not characters or whole words.
- How Byte-Pair Encoding builds a vocabulary.
- How a token ID becomes a dense vector (the embedding matrix) and what that vector "means."
- Why models work in hundreds or thousands of dimensions.

This chapter assumes the vectors and matrices from Chapter 2. Nothing else.

## Tokenization: turning text into a sequence of integers

Neural networks operate on numbers, specifically vectors of floating-point values. Raw text has to
become a discrete sequence of symbols first. That conversion is **tokenization**, and the fixed set
of symbols a model may use is its **vocabulary**. Every input and output is a sequence of integers,
each one an index into that vocabulary.

```text
text = "The cat sat"
token_ids = tokenizer.encode(text)   # → [464, 3797, 3332]
# 464 → "The",  3797 → " cat",  3332 → " sat"
```

::: {.figure}
![](assets/figures/ch03/fig-tokenization-pipeline.svg)
:::

::: {.caption}
**Figure 3.1.** Tokenization turns text into a sequence of integer IDs. Each subword token maps to one entry in the model's vocabulary.
:::

The real question is what a "symbol" should be. Two obvious answers both fail.

### Why not characters, or whole words?

**Characters** (a vocabulary of ~100) make sequences very long. A 1,000-character document is 1,000
tokens. Because attention's cost grows with the *square* of the sequence length (Chapter 5), that gets
expensive fast, and learning grammar from bare characters is hard.

**Whole words** (a vocabulary of 500,000+) make sequences short but explode the vocabulary. Every
inflection ("run", "runs", "running", "ran") is its own entry, rare words are never seen often
enough to learn well, and the output layer that scores every vocabulary item becomes enormous.

### Subword tokenization

Modern models split the difference, breaking text into meaningful **subword** chunks:

```text
"unhelpfulness" → ["un", "help", "ful", "ness"]
```

A vocabulary of 32,000–100,000 subwords keeps sequences reasonably short, handles rare words by
decomposing them into familiar pieces, and captures morphology: prefixes, suffixes, stems. Common
words stay single tokens. Rare ones break into parts the model has already seen.

### Byte-Pair Encoding (BPE)

The most common way to build a subword vocabulary is **Byte-Pair Encoding**. Start from a
character-level vocabulary, repeatedly find the most frequent pair of adjacent symbols, merge it into
a new token, and stop when the vocabulary reaches its target size.

```text
Corpus: "low lower lowest"
Start:  every character is its own token
Step 1: "l o" is the most frequent pair → merge → "lo w", "lo w e r", "lo w e s t"
Step 2: "lo w" → "low"                  →         "low",  "low e r",  "low e s t"
Step 3: "low e" → "lowe"                →         "low",  "lowe r",   "lowe s t"
… continue until the vocabulary is full
```

At inference the same merges are replayed on new text. With the GPT-4 tokenizer:

```text
"Transformers are amazing!"     → ["Transform","ers"," are"," amazing","!"]        (5 tokens)
"antidisestablishmentarianism"  → ["ant","idis","establish","ment","arian","ism"]  (6 tokens)
```

**SentencePiece** (used by T5, and by Llama 1 and 2) is a close relative that treats the raw text
stream as-is, spaces included, so it needs no language-specific word-splitting rules and handles any
script. Unseen characters fall back to raw bytes. Llama 3 moved to a tiktoken-style byte-level BPE.

### Choosing a vocabulary size

Vocabulary size is a trade-off with costs at both ends:

| Too small | Too large |
|---|---|
| Long sequences → expensive attention | Huge output matrix → expensive scoring |
| Rare concepts can't be expressed precisely | Many tokens seen too rarely to learn well |
| Hard to capture morphology | Wastes model capacity |

::: {.caption}
**Table 3.1.** The vocabulary-size trade-off, at both extremes.
:::

The usual sweet spot is 32k–100k. Oddly specific numbers like GPT-2's 50,257 come from
tokenizer-training heuristics plus a preference for hardware-friendly sizes.

::: {.callout .lens}
**Representation.** Tokenization is the model's first compression decision. You are choosing *what
unit of language* it reasons about. Word-level tokens make it think in words. Byte-level tokens let
it handle any language or encoding, at the cost of working harder to learn grammar.
:::

::: {.callout .note}
**Current trend.** Byte-level tokenizers (in some Gemma and Llama variants) are gaining ground: they
eliminate out-of-vocabulary problems and handle multilingual text, code, and unusual characters
cleanly. The price is longer sequences.
:::

## Token embeddings: from an integer to a meaning

Tokenization leaves us with a sequence of integers, but an integer carries no meaning. Token 3797
isn't "bigger than" or "more like" 3798. The next step gives each token a vector the model can
actually reason with.

### From one-hot to dense vectors

The naive encoding is **one-hot**: a vector as long as the vocabulary, all zeros except a single 1 at
the token's index. For a 50,000-token vocabulary that's a 50,000-long vector that is 99.998% zeros and
says nothing about similarity. Instead, each token is given a short, **dense** vector (typically
768–4096 numbers, all free to vary):

```text
One-hot (50,000 dims, sparse):  "cat" = [0, 0, …, 1, …, 0]   (49,999 zeros)
Embedding (4096 dims, dense):   "cat" = [0.23, −0.15, 0.78, 0.41, …, −0.33, 0.52]
```

::: {.figure}
![](assets/figures/ch03/fig-onehot-vs-dense.svg)
:::

::: {.caption}
**Figure 3.2.** A one-hot vector is long and almost all zeros, and says nothing about meaning. A dense embedding is short, and every number is free to carry meaning.
:::

Trained well, tokens with similar meanings land near one another:

```text
"cat"       = [ 0.23, −0.15,  0.78, …]
"kitten"    = [ 0.25, −0.14,  0.76, …]   ← very close to "cat"
"feline"    = [ 0.20, −0.18,  0.80, …]   ← also close
"democracy" = [−0.54,  0.82, −0.11, …]   ← far away
```

::: {.callout .plain}
Embeddings are like GPS coordinates for words. Similar meanings get similar coordinates. The model
learns these coordinates during training by noticing which words show up in similar contexts.
:::

### The embedding matrix is a lookup

Those coordinates live in the **embedding matrix** `E`, of shape `(V, d_model)`, one row per
vocabulary token, each row a `d_model`-long vector. Turning a token into its vector is just reading
the row at the token's index: the "matrix as a lookup table" idea from Chapter 2.

```text
E = embedding table, shape (50000, 768)
token_ids  = [464, 3797, 3332]
embeddings = E[token_ids]          # shape (3, 768)
# row 3797 of E is the vector for " cat" = [0.23, −0.15, 0.78, …]
```

::: {.figure}
![](assets/figures/ch03/fig-embedding-lookup.svg)
:::

::: {.caption}
**Figure 3.3.** The embedding matrix E holds one row per token. Turning a token ID into its vector is just reading that row.
:::

`E` is a learned parameter: during training the model discovers where in this `d_model`-dimensional
space each token belongs.

### What an embedding "means"

An embedding is the model's *context-free* view of a token: what it means before any interaction with
its neighbors. The famous Word2Vec (2013) observation is that this space has real geometric
structure:

```text
embedding("king") − embedding("man") + embedding("woman") ≈ embedding("queen")
```

::: {.figure}
![](assets/figures/ch03/fig-embedding-geometry.svg)
:::

::: {.caption}
**Figure 3.4.** Meaning has shape. Words with similar meanings sit close together (left), and a consistent relationship appears as a consistent direction (right), which is why king − man + woman lands near queen.
:::

::: {.callout .lens}
**Geometric.** Picture a high-dimensional space. Each token is a *point*. Similar meanings sit
*closer together*. Tokens in the same kind of relationship are separated by similar *directions*. The
model doesn't just learn positions. It learns a geometry in which meaning has shape.
:::

::: {.callout .lens}
**Representation.** An embedding is a lossy compression of a token's statistical role in language,
*what contexts it tends to appear in*, distilled into a fixed-size vector. The number of dimensions is
how many aspects of meaning the model can track at once.
:::

### How many dimensions? (768, 1024, 4096…)

The embedding size `d_model` balances several pressures:

1. **Expressiveness.** More dimensions mean more capacity for nuance, but with diminishing
   (log-linear) returns.
2. **Compute.** Attention and the feed-forward network both cost on the order of `d_model²`, so
   doubling the dimension roughly quadruples the work.
3. **Hardware.** GPUs are happiest when dimensions are multiples of 64 or 128. Both 768 (= 12×64) and
   4096 (= 64×64) fit well.
4. **Head count.** `d_model` must divide evenly among the attention heads (Chapter 5).

::: {.callout .lens}
**Mechanistic.** The embedding matrix is the model's *first* layer, and often its *last*. Through
**weight tying** (Chapter 8) the same matrix both looks up input embeddings and projects the final
vector back to vocabulary scores, saving parameters and improving generalization.
:::

### Why high dimensionality helps

Working in hundreds or thousands of dimensions (rather than, say, ten) is what makes it possible to
represent the complexity of language.

::: {.callout .idea}
**The curse and blessing of dimensionality.** In high-dimensional spaces almost every pair of random
vectors is nearly perpendicular. That gives the model enormous room: many independent features can be
encoded without interfering, and dot products naturally separate similar from unrelated (most land
near zero). Even 768 dimensions hold astronomically many nearly-distinct directions, far more than
enough for 50,000 tokens.
:::

## Summary

- **Tokenization** turns text into a sequence of integer IDs drawn from a fixed **vocabulary**.
  Subword schemes like **BPE** sit between characters (too long) and whole words (too many). A
  vocabulary of ~32k–100k is the usual sweet spot.
- Each ID becomes a **dense embedding vector** by looking up a row of the learned **embedding matrix**
  `E`, of shape `(V, d_model)`.
- Embeddings place tokens in a geometric space where distance and direction carry meaning
  (king − man + woman ≈ queen).
- The embedding size `d_model` trades expressiveness against compute and hardware. High dimensionality
  gives the model room to encode many features without interference.

> **Coming up:** An embedding gives each token a meaning, but attention, which we build in Chapter 5,
> treats a sentence as an unordered *bag* of these vectors. "The cat sat" and "sat cat The" would look
> identical to it. Chapter 4 fixes that by folding each token's *position* into its vector.
