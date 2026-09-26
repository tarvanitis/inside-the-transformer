# Copyright (c) 2026 Athanasios Arvanitis
# Released under the MIT License. See LICENSE.md in the book root (the MIT section).
"""
Chapter 11 — Every Component, Annotated
"Inside the Transformer"

A complete encoder-decoder transformer — the architecture from Vaswani et al. (2017),
annotated with references to the chapters where each component is explained.

Run:
    python3 ch11_transformer.py

Verified with PyTorch 2.14 (CPU). No GPU required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V  (Ch 5)"""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # scale by 1/sqrt(d_k) (Ch 5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    return F.softmax(scores, dim=-1) @ V                 # weighted sum of values (Ch 5)


def generate_causal_mask(seq_len: int, device) -> torch.Tensor:
    """Lower-triangular mask: position t can only attend to positions 0..t."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with explicit Q, K, V projections (Ch 5)."""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)   # query projection (Ch 5)
        self.W_K = nn.Linear(d_model, d_model, bias=False)   # key projection   (Ch 5)
        self.W_V = nn.Linear(d_model, d_model, bias=False)   # value projection (Ch 5)
        self.W_O = nn.Linear(d_model, d_model, bias=False)   # output mix       (Ch 5)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, Q_in, K_in, V_in, mask=None):
        B = Q_in.size(0)
        Q = self.split_heads(self.W_Q(Q_in))
        K = self.split_heads(self.W_K(K_in))
        V = self.split_heads(self.W_V(V_in))
        out = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_O(out)


class FeedForward(nn.Module):
    """Position-wise FFN: expand 4x → nonlinearity → compress (Ch 6)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),     # expand (Ch 6)
            nn.ReLU(),                    # nonlinearity (Ch 2; modern: GELU or SwiGLU, Ch 9)
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),     # compress back (Ch 6)
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """One encoder layer: self-attention + FFN, each wrapped in residual + LayerNorm (Ch 6)."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)    # post-norm (Ch 6)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # self-attention: Q = K = V = x
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """One decoder layer: masked self-attn → cross-attn → FFN (Ch 9)."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_out, tgt_mask, src_mask=None):
        # 1: masked self-attention over the target sequence so far
        tgt = self.norm1(tgt + self.dropout(self.masked_self_attn(tgt, tgt, tgt, tgt_mask)))
        # 2: cross-attention — Q from decoder, K and V from encoder output
        tgt = self.norm2(tgt + self.dropout(self.cross_attn(tgt, encoder_out, encoder_out, src_mask)))
        # 3: FFN
        tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))
        return tgt


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from Vaswani et al. (2017) — Chapter 4."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)   # even dims: sine wave (Ch 4)
        pe[:, 1::2] = torch.cos(pos * div)   # odd dims:  cosine wave (Ch 4)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    """Complete encoder-decoder transformer (Vaswani et al. 2017).

    Source vocabulary → encoder → contextualised representations
    Target (so far)   → decoder (cross-attends to encoder) → next-token logits
    """
    def __init__(self, src_vocab, tgt_vocab,
                 d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model)    # source embeddings (Ch 3)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)    # target embeddings (Ch 3)
        self.pos_enc = PositionalEncoding(d_model)          # positional encoding (Ch 4)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, tgt_vocab)      # LM head (Ch 8)

    def encode(self, src, src_mask=None):
        x = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x   # (B, src_len, d_model) — contextualised source representations

    def decode(self, tgt, enc_out, tgt_mask, src_mask=None):
        x = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        for layer in self.dec_layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return x   # (B, tgt_len, d_model)

    def forward(self, src, tgt, tgt_mask, src_mask=None):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, tgt_mask, src_mask)
        return self.out_proj(dec_out)   # (B, tgt_len, tgt_vocab) — logits over next tokens


if __name__ == "__main__":
    # Seed before anything random happens, including weight init, so the
    # numbers printed below reproduce exactly.
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build a tiny model for smoke-testing
    model = Transformer(
        src_vocab=32000, tgt_vocab=32000,
        d_model=64, num_heads=4, num_layers=2, d_ff=256, dropout=0.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")   # 6,407,936 — the three 32k tables dominate

    # Forward pass
    B, src_len, tgt_len = 4, 10, 8
    src = torch.randint(0, 32000, (B, src_len), device=device)
    tgt = torch.randint(0, 32000, (B, tgt_len), device=device)
    tgt_mask = generate_causal_mask(tgt_len, device)

    logits = model(src, tgt, tgt_mask)
    print(f"Logits shape: {logits.shape}")   # torch.Size([4, 8, 32000])

    # Training step — loss should drop as the model fits the random targets
    targets = torch.randint(0, 32000, (B, tgt_len), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    print("\nTraining (10 steps):")
    for step in range(10):
        logits = model(src, tgt, tgt_mask)
        loss = F.cross_entropy(logits.view(-1, 32000), targets.view(-1))
        opt.zero_grad()
        loss.backward()     # backpropagation: chain rule from loss to every weight (Ch 2)
        opt.step()
        if step % 2 == 0:
            print(f"  step {step}  loss {loss.item():.4f}")

    print("\nEncoder-decoder transformer verified!")
