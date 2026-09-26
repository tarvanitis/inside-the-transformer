# Every Component, Annotated

Chapter 10 built a complete GPT in under 100 lines by relying on the theory from Part II.
This chapter takes the opposite path: a slower, annotated walk through every component,
including the encoder–decoder pieces (cross-attention and masking) that a decoder-only
build never needs. By the end you have a fully working encoder–decoder transformer, the
architecture the original 2017 paper described. The annotated, component-by-component style here
follows in the spirit of Harvard NLP's *The Annotated Transformer*, reimplemented from scratch for
this book.

**In this chapter**

- The encoder path: embeddings, positional encoding, self-attention, FFN, and the full encoder stack.
- Causal masking in code, and why the decoder needs it.
- Cross-attention: how the decoder reads the encoder's output token by token.
- The complete encoder–decoder transformer, assembled and runnable in PyTorch.
- One full training step: forward pass, cross-entropy loss, backpropagation, weight update.

This chapter assumes the five steps of attention (Chapter 5), the pre-norm block
(Chapter 6), and the encoder–decoder architecture overview (Chapter 9).

## Imports

Everything in this chapter is standard PyTorch, with no model libraries.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
```

## Scaled dot-product attention

Recall from Chapter 5: attention is `softmax(QKᵀ / √d_k) @ V`. Here it becomes code. The
`mask` parameter handles both uses we will need: causal masking in the decoder and padding
masking in the encoder.

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)                                      # head dimension (Ch 5)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)    # Ch 5: scale by 1/√d_k
    if mask is not None:                                  # mask = 0 → attend nowhere
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)              # Ch 2: softmax over positions
    return attn_weights @ V                               # weighted sum of values (Ch 5)
```

::: {.callout .note}
**Why `masked_fill` with −∞?** After softmax, e^(−∞) = 0 exactly, so those positions receive
zero attention weight and are invisible to the model. The masking has to happen *before*
softmax. Softmax normalizes across every position, so zeroing weights afterwards would leave
the survivors summing to less than 1 and quietly shrink the output. Masking first means the
remaining positions renormalize among themselves, exactly as they should.
:::

## Multi-head attention with explicit Q, K, V projections

Chapter 10's `CausalSelfAttention` fused all three projections into one `Linear` call: fast,
but it hides what's happening. Here we keep them separate so the data flow stays visible.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads            # per-head dimension (Ch 5)
        self.num_heads = num_heads
        # Three separate projections: each input token → query, key, value space (Ch 5)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)   # output mix across heads (Ch 5)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) → (B, num_heads, T, d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, Q_in, K_in, V_in, mask=None):
        B = Q_in.size(0)
        Q = self.split_heads(self.W_Q(Q_in))    # project then reshape into heads
        K = self.split_heads(self.W_K(K_in))
        V = self.split_heads(self.W_V(V_in))
        out = scaled_dot_product_attention(Q, K, V, mask)          # Ch 5: the five steps
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_O(out)                     # merge and mix head outputs
```

::: {.callout .idea}
When Q, K, and V all come from the same sequence (`mha(x, x, x)`) it is **self-attention**:
each token asks questions about, and answers from, the same sequence. When Q comes from
a different sequence than K and V (`mha(decoder_state, enc_out, enc_out)`) it is
**cross-attention**: the decoder queries the encoder's memory. The module is the same.
The caller decides which form it becomes.
:::

## The feed-forward network

After attention, each token's vector passes through a small two-layer network independently.
The 4× expansion gives the model more capacity to transform each token before contracting
back, a high-dimensional scratch pad, as Chapter 6 described.

The source architecture uses ReLU here. The modern alternative is GELU or SwiGLU (Chapter 2,
Chapter 9). Both work the same way in the surrounding structure.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),    # expand: d_model → d_ff (typically 4×) (Ch 6)
            nn.ReLU(),                   # nonlinearity (Ch 2; modern: GELU or SwiGLU, Ch 9)
            nn.Dropout(dropout),         # regularization: randomly zero values during training
            nn.Linear(d_ff, d_model),    # compress back (Ch 6)
        )

    def forward(self, x):
        return self.net(x)   # applied identically at every sequence position
```

::: {.callout .note}
**Dropout.** During training, `nn.Dropout(p)` randomly zeroes a fraction `p` of values at each
forward pass, forcing the model to learn redundant representations rather than relying on any
one neuron. At inference time (`model.eval()`), dropout is disabled automatically and all values
pass through unchanged.
:::

## The encoder layer

One encoder layer is the template from Chapter 6, in **post-norm** form: transform, then add
the residual, then normalize. The original 2017 paper used this ordering. Modern transformers
use **pre-norm** (normalize first, then transform), which trains more stably for deep stacks
(Chapter 6).

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)   # self-attention (Ch 5)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)    # post-norm (Ch 6); modern: pre-norm
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Sub-layer 1: self-attention + residual + normalize
        attn_out = self.self_attn(x, x, x, mask)      # Q = K = V = x (self-attention)
        x = self.norm1(x + self.dropout(attn_out))    # residual: gradient flows directly (Ch 6)
        # Sub-layer 2: FFN + residual + normalize
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
```

::: {.callout .deepdive}
**Post-norm vs pre-norm.** The residual equation is `output = Norm(x + Sublayer(x))` in
post-norm and `output = x + Sublayer(Norm(x))` in pre-norm. The difference matters at
initialization: with post-norm, the unnormalized residual passes directly through the first
few layers and can destabilize very deep stacks without a careful warm-up schedule. Pre-norm
puts the normalization inside the residual branch, which keeps activations stable regardless
of depth. Chapter 6's block and Chapter 10's GPT both use pre-norm. Swap `norm1`/`norm2` calls
to `self.norm1(x)` before the sublayer to modernize this encoder layer.
:::

## Positional encoding

Transformers process all tokens in parallel, so without positional information, "dog bites man"
and "man bites dog" produce identical representations. We fix this by adding a position vector
to each embedding. Chapter 4 covers the full rationale and modern alternatives (RoPE, ALiBi).
Here is the sinusoidal implementation from the original paper.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)              # position indices
        # frequencies decrease geometrically with dimension index (Ch 4)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)    # even dimensions: sine wave (Ch 4)
        pe[:, 1::2] = torch.cos(pos * div)    # odd dimensions: cosine wave (Ch 4)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]    # add position to each token embedding
```

::: {.callout .plain}
`register_buffer` is how you tell PyTorch "keep this tensor with the model, but never train
it." These sine and cosine values are fixed by the formula, so there is nothing to learn.
Registering them means they move to the GPU with the model and get saved alongside it, while
the optimizer leaves them alone.
:::

## The encoder stack

Stack `num_layers` identical encoder layers. After passing through all layers, each token
vector encodes the token's own meaning plus contextual information gathered from the
entire sequence through attention at every layer.

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)    # token lookup (Ch 3)
        self.pos_enc = PositionalEncoding(d_model, max_len)   # positional vectors (Ch 4)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Scale embeddings before adding positions: prevents PE from dominating
        x = self.embedding(src) * math.sqrt(self.d_model)    # (B, T) → (B, T, d_model)
        x = self.dropout(self.pos_enc(x))
        for layer in self.layers:
            x = layer(x, src_mask)
        return x   # (B, T, d_model) — contextualized representations of the source
```

We inline this stack into the full `Transformer` below rather than nesting the class, so the
encoder and decoder paths sit side by side in one place.

::: {.callout .lens}
**Representation.** Earlier encoder layers tend to capture local syntax, meaning nearby word
relationships. Middle layers build phrase-level structure. Final layers encode full
sentence semantics. This specialization is not programmed. It emerges from training on
enough data. By the time the encoder stack finishes, each token vector carries a
summary of the entire input sequence as seen from that token's perspective.
:::

## Causal masking

The decoder generates output one token at a time. During training we feed the entire target
sequence at once for efficiency, but the model must not peek at future tokens. To predict the
token at position 4, it may read positions 0–3, so the row for position 3 attends to 0 through 3.

A causal mask enforces this with a lower-triangular matrix:

```
Causal mask, T = 4:

        pos0  pos1  pos2  pos3
pos0  [  1,    0,    0,    0  ]   pos0 sees only itself
pos1  [  1,    1,    0,    0  ]   pos1 sees pos0 and itself
pos2  [  1,    1,    1,    0  ]
pos3  [  1,    1,    1,    1  ]   pos3 sees all previous
```

Where the mask is 0, the attention score is set to −∞ before softmax, giving exactly zero
attention weight to those positions.

```python
def generate_causal_mask(seq_len: int, device) -> torch.Tensor:
    # torch.tril: keep the lower triangle (1s on and below the diagonal)
    return torch.tril(torch.ones(seq_len, seq_len, device=device))
```

## Cross-attention

Cross-attention is what distinguishes a decoder layer from an encoder layer. In self-attention,
queries, keys, and values all come from the same sequence. In cross-attention the **query comes
from the decoder** ("what am I trying to generate next?") and the **keys and values come from
the encoder's output**: "what did the input say?" The decoder can attend to any position in the
source sequence at every decoder step.

```
Cross-attention:
  Q  ←  decoder's current state     "What am I looking for in the source?"
  K  ←  encoder output              "What does each source position offer?"
  V  ←  encoder output              "What information do I read from there?"

  attn_weights = softmax(Q @ Kᵀ / √d_k)   (Ch 5)
  output       = attn_weights @ V
```

::: {.figure}
![](assets/figures/ch11/fig-self-vs-cross.svg)
:::

::: {.caption}
**Figure 11.1.** The same attention module does both jobs. In self-attention the query, key, and value all come from one sequence. In cross-attention the query comes from the decoder while the keys and values come from the encoder's output.
:::

This is how a translation model generates a German word by attending back to the relevant
English tokens. The encoder ran once. Cross-attention reads it at every decoder step.

## The decoder layer

Each decoder layer has three sub-layers: masked self-attention over the generated target so
far, cross-attention into the encoder output, and the FFN.

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)  # self-attention (Ch 5)
        self.cross_attn       = MultiHeadAttention(d_model, num_heads)  # cross-attention
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)    # post-norm (Ch 6)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_out, tgt_mask, src_mask=None):
        # Sub-layer 1: masked self-attention — only attend to past target positions
        tgt = self.norm1(tgt + self.dropout(
            self.masked_self_attn(tgt, tgt, tgt, tgt_mask)   # causal mask applied
        ))
        # Sub-layer 2: cross-attention — Q from decoder, K and V from encoder output
        tgt = self.norm2(tgt + self.dropout(
            self.cross_attn(tgt, encoder_out, encoder_out, src_mask)
        ))
        # Sub-layer 3: position-wise FFN
        tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))
        return tgt
```

::: {.callout .caution}
**The mask arguments matter.** `tgt_mask` is the causal mask, the lower-triangular matrix
that prevents future positions from leaking into the present. `src_mask` is a padding mask
marking which source positions to ignore (not needed when sequences in a batch are the same
length, but essential in production). Passing the wrong mask to the wrong attention layer is one
of the easiest mistakes to make here. Often you get a loud shape error, because the causal mask
is `tgt_len × tgt_len` while cross-attention scores are `tgt_len × src_len`. When source and
target happen to be the same length, though, the shapes line up, the forward pass completes, and
the model quietly learns a broken information flow.
:::

## Output projection and logits

After the decoder stack, one linear layer projects each `d_model`-dimensional token vector up
to `vocab_size` scores. Softmax over those scores gives a probability distribution over all
possible next tokens.

```python
# decoder output: (B, tgt_len, d_model)
# self.out_proj is nn.Linear(d_model, vocab_size)
logits = self.out_proj(decoder_out)   # → (B, tgt_len, vocab_size)
probs  = F.softmax(logits, dim=-1)

# logits = [3.2, -1.1, 5.7, 0.2, ...]   raw un-normalized scores
# probs  = [0.068, 0.001, 0.83, 0.003, ...]  sum to 1.0 across the full vocabulary
```

The token with the highest probability is the model's most confident next-token prediction. In
practice, sampling with temperature or top-k (Chapter 8) produces more varied and natural output
than always picking the argmax (the single highest-scoring token).

## Loss function and backpropagation

Training minimizes cross-entropy (Chapter 2): how surprised is the model by the correct token?
PyTorch's `loss.backward()` then propagates the error signal backward through every layer via
the chain rule, computing a gradient for every parameter. The optimizer applies those gradients
to update the weights.

```python
# targets: token IDs the model should have predicted, shape (B, tgt_len)
# logits:  model's raw scores,                       shape (B, tgt_len, vocab_size)
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),   # flatten to (B×tgt_len, vocab_size)
    targets.view(-1),                   # flatten to (B×tgt_len,)
)
# loss = -log(probability assigned to the correct token) (Ch 2)

optimizer.zero_grad()   # clear accumulated gradients from the previous step
loss.backward()         # backprop: chain rule from loss → every weight (Ch 2)
optimizer.step()        # apply gradients: nudge weights toward lower loss
```

## The complete encoder–decoder transformer

All components assembled into a single, runnable class. The full forward pass is four lines.

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab,
                 d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model)       # source token embeddings (Ch 3)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)       # target token embeddings (Ch 3)
        self.pos_enc = PositionalEncoding(d_model)             # shared PE module (Ch 4)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, tgt_vocab)         # LM head (Ch 8)

    def encode(self, src, src_mask=None):
        x = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x   # (B, src_len, d_model)

    def decode(self, tgt, enc_out, tgt_mask, src_mask=None):
        x = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        for layer in self.dec_layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return x   # (B, tgt_len, d_model)

    def forward(self, src, tgt, tgt_mask, src_mask=None):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, tgt_mask, src_mask)
        return self.out_proj(dec_out)   # (B, tgt_len, vocab_size)
```

Build and smoke-test with a tiny configuration:

```python
torch.manual_seed(0)   # seed before weight init, so these numbers reproduce exactly

model = Transformer(src_vocab=32000, tgt_vocab=32000,
                    d_model=64, num_heads=4, num_layers=2, d_ff=256, dropout=0.0)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")
# ~6.4M for this tiny config — the three embedding/projection tables (3 × 32,000 × 64)
# dominate. Scaled to the paper's d_model=512, N=6, vocab~37K this class gives ~101M.
# the paper's ~65M shares one matrix across both embeddings and the output projection
# (the weight tying of Chapter 8, which Chapter 10's GPT also uses).

B, src_len, tgt_len = 4, 10, 8
src = torch.randint(0, 32000, (B, src_len))
tgt = torch.randint(0, 32000, (B, tgt_len))
tgt_mask = generate_causal_mask(tgt_len, src.device)

logits = model(src, tgt, tgt_mask)
print(f"Logits shape: {logits.shape}")   # torch.Size([4, 8, 32000])
```

A full training step to confirm backpropagation works:

```python
targets = torch.randint(0, 32000, (B, tgt_len))
opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

losses = []
for step in range(10):
    logits = model(src, tgt, tgt_mask)
    loss = F.cross_entropy(logits.view(-1, 32000), targets.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())

print(f"Step 0 loss: {losses[0]:.4f}")   # ~10.4 (random, close to log(32000))
print(f"Step 9 loss: {losses[-1]:.4f}")  # lower — backprop works
# Step 0 loss: 10.5752
# Step 9 loss: 7.8185
```

## Key hyperparameters

| Parameter | Original paper | What it controls |
|---|---|---|
| `d_model` | 512 | Size of every token vector, bigger = more expressive, more compute |
| `num_heads` | 8 | Parallel attention perspectives, each head learns different relationships (Ch 5) |
| `d_k` | 64 (= 512 / 8) | Dimension per head, `d_model` divided equally across heads |
| `num_layers` | 6 | Depth of encoder and decoder stacks, deeper = more abstract representations |
| `d_ff` | 2048 (= 4 × 512) | Width of the FFN inner layer, expansion ratio typically 4× (Ch 6) |
| `dropout` | 0.1 | Fraction zeroed during training, prevents overfitting |
| `vocab_size` | ~37,000 (BPE) | Tokens the model can handle, BPE balances coverage and vocabulary size (Ch 3) |

::: {.caption}
**Table 11.1.** The original paper's hyperparameters, and what each one controls.
:::

::: {.callout .plain}
The original paper's 65 million parameters trained in ~12 hours on 8 P100 GPUs. Modern
frontier models have 70 billion to 2 trillion parameters and train for months on thousands of
accelerators, but the architecture you just built is the direct ancestor of all of them.
:::

## Summary

- **Self-attention** (`Q = K = V = x`) and **cross-attention** (`Q` from the decoder,
  `K` and `V` from the encoder) use the same module. The caller determines which form it takes.
- The **encoder** reads the full source sequence and produces contextualized representations.
  The **decoder** generates target tokens one at a time, attending to its own past and to the
  encoder's output through cross-attention.
- **Causal masking** enforces the autoregressive constraint: position `t` can only attend to
  positions 0 through `t`, its own position included, never any position after it.
- The original paper uses **post-norm** (add residual, then normalize). Modern models use
  **pre-norm** (normalize, then add residual), which trains deeper stacks more stably.
- A training step is: forward pass → cross-entropy loss → `loss.backward()` → `opt.step()`.

> **Coming up:** The model works. The question now is how to make it useful. Chapter 12 opens
> Part IV with a map of the full training pipeline: the seven stages from pre-training through
> reasoning RL, and what each one achieves. Chapter 13 then goes deep inside the first and most
> computationally intensive stage: pre-training at scale.
