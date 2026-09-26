# Building a GPT from Scratch in PyTorch

Part II explained every piece of a transformer. This chapter turns that understanding into code: a
small but complete decoder-only GPT (the architecture from Chapter 9) that you can read in one
sitting, train, and sample from. The code is deliberately minimal and modern. Each component
implements a concept from Part II, and we point back to those chapters rather than re-explaining the
theory.

**In this chapter**

- The model: causal self-attention, the block, and the full GPT, in idiomatic PyTorch.
- A training loop, from batches to gradient steps.
- Generating text from the trained model.

::: {.callout .note}
This chapter is code-first. Everything here is standard `torch` (no external model libraries) and
mirrors the design of Andrej Karpathy's nanoGPT. You need to be able to read Python. The PyTorch
idioms are explained as they appear. The companion script in `code/` uses a smaller configuration
than the listings below, so that it trains in seconds on a laptop CPU.
:::

## Configuration

A small dataclass holds the model's shape. These are the design trade-offs from Chapter 9 (vocabulary
size, context length (`block_size`), depth, heads, and width) gathered into one config object.

```python
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257     # GPT-2 BPE vocabulary (Chapter 3)
    block_size: int = 256       # maximum context length
    n_layer: int = 6            # number of transformer blocks
    n_head: int = 6             # attention heads per block
    n_embd: int = 384           # embedding dimension (d_model); must be divisible by n_head
    dropout: float = 0.0
```

## Causal self-attention

This is Chapter 5 in code. One `Linear` projects the input to Q, K, and V at once. We then reshape into
heads, run scaled dot-product attention with a **causal** mask, then project back. We call PyTorch's
built-in `F.scaled_dot_product_attention`. On a CUDA GPU in half precision (fp16/bf16) this can
dispatch to a fused **FlashAttention** kernel (Chapter 8) that never materializes the full `T × T`
score matrix. At the default fp32, or on CPU (where the companion script runs), it uses an equivalent
non-fused kernel instead.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0                # d_model divisible by heads (Ch 5)
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)   # Q, K, V (Ch 5)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)      # output W_O (Ch 5)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape                                  # batch, sequence, d_model
        q, k, v = self.qkv(x).split(C, dim=2)              # project to Q, K, V (Ch 5)
        # reshape into the H parallel heads of Ch 5: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # the five steps of Ch 5 — softmax(QK^T / sqrt(d_k)) @ V, causal-masked — as one
        # fused FlashAttention kernel (Ch 8). The 1/sqrt(d_k) scaling is applied internally.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )                                                  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).reshape(B, T, C)             # re-merge heads
        return self.proj(y)
```

::: {.callout .plain}
Two lines here look like they should be the same and aren't. `.view` reshapes a tensor for free,
but only when its values are still laid out in memory in reading order. After `.transpose(1, 2)`
they aren't, so the merge step on the way out uses `.reshape`, which copies if it has to.

The one `Linear` producing `3 * n_embd` outputs is a speed trick, not a new idea. Q, K and V each
need their own projection of the input, and doing all three as one matrix multiply is much faster
on a GPU than three. `.split(C, dim=2)` then chops the wide result back into the three pieces
Chapter 5 described.
:::

::: {.figure}
![](assets/figures/ch10/fig-shape-ladder.svg)
:::

::: {.caption}
**Figure 10.1.** The shape ladder inside one attention call. The token vectors are regrouped into H heads, attention runs on all heads in parallel, then the heads are merged back. Because C = H × d_k, no numbers are lost.
:::

## The transformer block

Chapter 6, in code. A **pre-norm** block: normalize, attend, add, then normalize, feed-forward, add. The
feed-forward network expands by 4×, applies a nonlinearity, and compresses back.

```python
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.n_embd)             # LayerNorm (Ch 6)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(                         # feed-forward network (Ch 6)
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False),   # expand x4
            nn.GELU(),                                            # nonlinearity (Ch 2)
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),   # compress back
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))     # pre-norm + residual around attention (Ch 6)
        x = x + self.mlp(self.norm2(x))      # pre-norm + residual around the FFN (Ch 6)
        return x
```

::: {.callout .deepdive}
**Bringing it up to a 2025 model.** Three swaps turn this classic block into the modern stack from
Chapter 9: replace `nn.LayerNorm` with `nn.RMSNorm`; replace the `Linear → GELU → Linear` MLP with a
**SwiGLU** FFN (`w_down(F.silu(w_gate(x)) * w_up(x))`, with the hidden width scaled to ~⅔ × 4·d_model);
and add **RoPE** (Chapter 4) inside attention instead of the learned positional embedding below. For
serving, switch attention to **GQA** by giving K and V fewer heads than Q (`enable_gqa=True` in
`scaled_dot_product_attention`). The training loop and the rest of the model stay the same.
:::

## The full model

Chapter 9 assembled: token embeddings plus (learned) positional embeddings, a stack of blocks, a final
norm, and the language-model head. We **tie weights** (Chapter 8) so the embedding table and the LM
head share one matrix.

```python
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token embeddings (Ch 3)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # positions (Ch 4)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])   # depth (Ch 6)
        self.norm = nn.LayerNorm(cfg.n_embd)                      # final norm
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)   # LM head (Ch 8)
        self.tok_emb.weight = self.lm_head.weight                # weight tying (Ch 8)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B, T, n_embd)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))                    # scores over the vocabulary (Ch 8)
        loss = None
        if targets is not None:                                # training loss: cross-entropy (Ch 2)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

We use a *learned* positional embedding here (GPT-2 style) to keep the build small. Chapter 4's RoPE is
the modern choice and slots into `CausalSelfAttention`.

::: {.callout .plain}
That `+` in the forward pass is doing something worth noticing. `self.tok_emb(idx)` is shaped
`(B, T, n_embd)`, one vector per token, per sequence in the batch. `self.pos_emb(pos)` is only
`(T, n_embd)`, one vector per position. PyTorch stretches the smaller one across the batch for
you rather than making you write the loop. That is *broadcasting*, and you will see it constantly.
:::

## Training

Language modeling is next-token prediction: the target is the input shifted by one. A batch is just
random windows of a long token array. The loss is cross-entropy (Chapter 2), optimized with AdamW.

```python
def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    # each target is the token that follows its input — next-token prediction (Ch 8)
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = GPTConfig()
model = GPT(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# encoded_corpus: the whole corpus run through the Chapter 3 tokenizer — one long list of token IDs.
# (Illustrative: the companion script substitutes synthetic random IDs so it runs with no dataset.)
encoded_corpus = tokenizer.encode(corpus_text)     # Chapter 3
train_data = torch.tensor(encoded_corpus, dtype=torch.long)
max_steps = 5000

for step in range(max_steps):
    x, y = get_batch(train_data, cfg.block_size, batch_size=32, device=device)
    logits, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
```

That loop is the entire training algorithm. The details that make it *work at scale* (learning-rate
schedules, gradient clipping, mixed precision, the full pipeline of pre-training and alignment) are the
subject of Part IV.

## Generating text

Generation is the loop from Chapter 8: repeatedly run the model, take the last position's logits, apply
temperature (Chapter 2) and optional top-k (Chapter 8), sample, and append.

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):                  # the generation loop (Ch 8)
        idx_cond = idx[:, -block_size:]              # never exceed the context window (Ch 8)
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature      # last step — temp (Ch 2)
        if top_k is not None:                        # top-k filtering (Ch 8)
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)            # logits -> probabilities (Ch 2)
        next_id = torch.multinomial(probs, num_samples=1)   # sample next token (Ch 8)
        idx = torch.cat([idx, next_id], dim=1)       # append and repeat
    model.train()                                    # restore training mode (dropout back on)
    return idx
```

`model.eval()` switches dropout off, which is what you want while sampling. Putting the model back
into training mode on the way out matters if you sample mid-run to watch the model improve,
otherwise dropout stays off for the rest of your training.

That's a complete GPT (under 100 lines of model code) implementing everything from Part II: BPE
tokens in, embeddings and positions, causal multi-head attention, pre-norm blocks with residuals, a
tied LM head, cross-entropy training, and autoregressive sampling out.

## Summary

- A working decoder-only GPT is small: a `CausalSelfAttention` module (SDPA with `is_causal=True`), a
  pre-norm `Block`, and a `GPT` that stacks blocks between a tied embedding/LM-head.
- **Training** is next-token prediction: shift the input by one, minimize cross-entropy with AdamW.
- **Generation** crops to the context window, samples the last position with temperature/top-k, and
  appends, one token at a time.
- The modern stack (RMSNorm, SwiGLU, RoPE, GQA) is a handful of drop-in swaps away.

> **Coming up:** This chapter built a model quickly, top-down. Chapter 11 goes the other way: a slow,
> annotated walk through every component, including the encoder–decoder pieces (cross-attention,
> masking) we skipped here, to cement exactly what each line computes.
