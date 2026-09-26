# Copyright (c) 2026 Athanasios Arvanitis
# Released under the MIT License. See LICENSE.md in the book root (the MIT section).
"""
Chapter 10 — Building a GPT from Scratch in PyTorch
"Inside the Transformer"

Run:
    python3 ch10_gpt.py

Verified with PyTorch 2.14 (CPU). No GPU required.
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 100        # tiny vocab for this demo (Ch 3)
    block_size: int = 16         # context window
    n_layer: int = 4             # transformer blocks
    n_head: int = 4              # attention heads (Ch 5)
    n_embd: int = 64             # d_model (Ch 5)
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0          # d_model divisible by heads (Ch 5)
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)   # Q, K, V (Ch 5)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)      # output W_O (Ch 5)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # five steps of Ch 5 as one fused FlashAttention kernel (Ch 8)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):   # pre-norm block (Ch 6)
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.n_embd)        # LayerNorm (Ch 6)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False),   # expand x4 (Ch 6)
            nn.GELU(),                                             # nonlinearity (Ch 2)
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))     # pre-norm + residual around attention (Ch 6)
        x = x + self.mlp(self.norm2(x))      # pre-norm + residual around the FFN (Ch 6)
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token embeddings (Ch 3)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # positions (Ch 4)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)   # LM head (Ch 8)
        self.tok_emb.weight = self.lm_head.weight                          # weight tying (Ch 8)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:                          # training loss: cross-entropy (Ch 2)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


if __name__ == "__main__":
    # Seed before anything random happens, including weight init, so the
    # numbers printed below reproduce exactly.
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Check weight tying
    assert model.tok_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()
    print("Weight tying confirmed")

    # Forward pass check
    dummy = torch.zeros(2, 4, dtype=torch.long, device=device)
    logits, _ = model(dummy)
    print(f"Logits shape: {logits.shape}")   # (2, 4, 100)

    # Synthetic training data
    train_data = torch.randint(0, cfg.vocab_size, (10000,))

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    model.train()
    print("\nTraining (50 steps):")
    for step in range(50):
        x, y = get_batch(train_data, cfg.block_size, batch_size=32, device=device)
        logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"  step {step:2d}  loss {loss.item():.4f}")

    # Generate
    prompt = torch.zeros(1, 1, dtype=torch.long, device=device)
    out = generate(model, prompt, max_new_tokens=20, block_size=cfg.block_size, top_k=10)
    print(f"\nGenerated token IDs: {out[0].tolist()}")
