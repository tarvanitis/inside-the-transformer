# Position: How a Model Knows Word Order

Chapter 3 turned each token into an embedding, a vector carrying its meaning. But those vectors
arrive at the attention mechanism as an unordered set. As we'll see in Chapter 5, attention compares
every token with every other token *simultaneously*, with no built-in sense of which came first. Left
alone, it would read "The cat bit the dog" and "The dog bit the cat" as the same thing. This chapter
is about the fix: folding each token's *position* into its representation.

**In this chapter**

- Why attention is "position-blind," and why that's a problem.
- The original sinusoidal encoding, and its limitation.
- RoPE: how rotating vectors encodes *relative* position, and why it won.
- How a model's context window gets extended.

This chapter builds on embeddings (Chapter 3) and the rotation primitive (Chapter 2).

## Attention is position-blind

Self-attention computes relationships between all pairs of tokens at once, but nothing in that
computation depends on their order. Shuffle the tokens and attention produces the same scores, just
rearranged. The mechanism is **permutation-equivariant**. Language, though, is deeply ordered. "The
cat bit the dog" and "The dog bit the cat" share every token yet mean opposite things. So before
attention runs, we have to inject position into the token representations.

## Sinusoidal encodings (2017)

The original transformer added a fixed pattern of sine and cosine waves to each token's embedding, one
value per dimension:

::: {.callout .plain}
The trick is to describe a position the way a clock describes a time. A clock uses several hands moving
at different speeds, and the combination of their angles pins down the moment exactly. Here each
embedding dimension is like one hand: some sweep quickly from one position to the next, others slowly.
Read all of them at a given position and you get a pattern of values unique to that spot, with nearby
positions giving nearby patterns.
:::

$$\mathrm{PE}(pos, 2i)=\sin\!\left(\frac{pos}{10000^{\,2i/d_{model}}}\right), \qquad \mathrm{PE}(pos, 2i+1)=\cos\!\left(\frac{pos}{10000^{\,2i/d_{model}}}\right)$$

::: {.figure}
![](assets/figures/ch04/fig-clock-analogy.svg)
:::

::: {.caption}
**Figure 4.1.** Sinusoidal encoding works like a set of clock hands turning at different speeds. Each dimension is one hand, and the whole set of angles is a signature unique to a position.
:::

Each position gets a unique signature across the dimensions (low dimensions wave quickly, high ones
slowly) and nearby positions get similar signatures. In code:

```python
import math
import torch

def sinusoidal_positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
    pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions
    return pe                                       # shape (seq_len, d_model)

# added straight onto the token embeddings:
input_embeddings = token_embeddings + sinusoidal_positional_encoding(3, 768)
```

::: {.figure}
![](assets/figures/ch04/fig-sinusoidal-heatmap.svg)
:::

::: {.caption}
**Figure 4.2.** The sinusoidal signature across positions and dimensions. Low dimensions change quickly from one position to the next (short wavelength), while high dimensions change slowly (long wavelength).
:::

::: {.callout .caution}
**The limitation.** These encodings are *absolute*: position 500 always gets the same vector,
whatever the context. The model then has to work out *relative* distance from two absolute signatures
which is awkward, and it generalizes poorly past the sequence length it was trained on.
:::

## RoPE: rotary position embedding

Modern models (Llama, Mistral, Gemma, Qwen) use **rotary position embeddings** (RoPE; Su et al.,
2021) instead. The idea builds directly on the rotation primitive from Chapter 2: rather than *adding*
a positional vector, RoPE *rotates* each query and key vector by an angle proportional to its
position. When two rotated vectors are then compared with a dot product, the result depends on their
*relative* distance.

```text
q_i_rotated = rotate(q_i, by an angle ∝ position i)
k_j_rotated = rotate(k_j, by an angle ∝ position j)
score(i, j) = q_i_rotated · k_j_rotated
            = f( content_i, content_j, relative distance (i − j) )
```

::: {.callout .plain}
Picture each token's vector as an arrow. RoPE spins the arrow by an amount set by the token's
position. When you compare two arrows, the angle between them tells you how far apart the tokens are
in the sequence.
:::

::: {.callout .lens}
**Geometric.** Each position is a rotation in 2D subspaces of the embedding. The dot product of two
rotated vectors turns out to depend on `i − j` only, the *difference* of the two rotation angles,
never on `i` and `j` separately. Rotating both by an extra amount leaves it unchanged. The model sees
relative geometry, not absolute coordinates.
:::

**Worked example: one 2D slice.** Take a query `q = [1.0, 0.0]` with θ = 0.5 radians per position,
and a key `k = [1.0, 0.0]` sitting at position 3.

```text
Rotate the query to position 2:  by 2 × 0.5 = 1.0 rad  →  [cos 1.0, sin 1.0] = [ 0.540, 0.841]
Rotate the query to position 5:  by 5 × 0.5 = 2.5 rad  →                       [−0.801, 0.599]
Rotate the key to position 3:    by 3 × 0.5 = 1.5 rad  →                       [ 0.071, 0.997]

q at position 2 · k at position 3 = 0.540×0.071 + 0.841×0.997 = 0.878 = cos(0.5)
q at position 5 · k at position 3 = −0.801×0.071 + 0.599×0.997 = 0.540 = cos(1.0)
```

Two positions apart gives cos(1.0), one position apart cos(0.5). The score depends only on how
far apart the positions are, never on where in the sequence they sit, which is the whole point.

::: {.figure}
![](assets/figures/ch04/fig-rope-circle.svg)
:::

::: {.caption}
**Figure 4.3.** RoPE rotates each query and key by an angle set by its position, so the attention score depends only on the gap between positions, not on where the pair sits.
:::

RoPE has three advantages that made it the default. Relative distance is baked straight into the
attention score. It extrapolates better beyond the training length (with the scaling tricks below).
And it adds *no parameters*, because it is a fixed transform applied at attention time, not something
the model has to learn.

## Extending the context window

::: {.callout .deepdive}
**ALiBi (Attention with Linear Biases).** A different route: instead of rotating Q and K, add a
penalty to each attention score proportional to how far apart the two tokens are. Simpler to
implement and it extrapolates well, though it is less expressive than RoPE. Used in BLOOM and MPT.
:::

::: {.callout .note}
**Current trend.** RoPE paired with **YaRN** or **LongRoPE** scaling stretches a model's context
window without full retraining: the rotational frequencies are rescaled so that positions seen at
inference map onto rotations the model already met during training.
:::

## Summary

- Attention is **permutation-equivariant**, blind to order, so position must be injected into token
  representations before attention runs.
- The original transformer added fixed **sinusoidal** signatures, but these are *absolute* and
  generalize poorly beyond the training length.
- **RoPE** rotates queries and keys by an angle set by position, so the dot product encodes *relative*
  distance directly. It adds no parameters and is now the default.
- Context windows are extended by rescaling RoPE's frequencies (YaRN, LongRoPE).

> **Coming up:** We now have token vectors that carry both meaning and position. Chapter 5 puts them
> to work in the mechanism at the heart of the transformer (self-attention), where every token
> decides how much to draw from every other.
