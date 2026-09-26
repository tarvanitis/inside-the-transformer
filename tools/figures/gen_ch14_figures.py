#!/usr/bin/env python3
"""Generate the Chapter 14 (SFT) figures as static SVGs.

Two figures: response masking, and LoRA. Shared helpers in svgkit.py.

Usage:
    python3 tools/figures/gen_ch14_figures.py [--png]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch14"


# ============================================================================
# Figure 14.1 — Response masking
# ============================================================================
def fig_response_masking():
    W, H = 640, 290
    s = header(W, H)
    s += text(W/2, 30, "SFT masks the instruction: only the response tokens produce a learning signal.",
              size=11.5, col=CHARCOAL)
    # instruction block
    ix, iw, iy, ih = 26, 176, 150, 40
    s += rect(ix, iy, iw, ih, fill=GRAYFILL, stroke=CHARCOAL, sw=1, rx=5)
    s += text(ix + iw/2, iy + 15, "What is the capital", size=10, col=INK)
    s += text(ix + iw/2, iy + 29, "of France?", size=10, col=INK)
    s += text(ix + iw/2, iy - 10, "instruction (masked)", size=9.5, col=MUTE, style="italic")
    s += arrow(ix + iw + 4, iy + ih/2, ix + iw + 30, iy + ih/2, col=GRAY, marker="gray", w=1.5)
    # response tokens + loss bars
    toks = [("The", 0.33), ("capital", 0.60), ("of", 0.09), ("France", 0.39), ("is", 0.14), ("Paris", 1.17)]
    x0, tw, gap = 244, 58, 5
    ty, th = iy, ih
    for i, (tok, loss) in enumerate(toks):
        x = x0 + i*(tw + gap)
        red = tok == "Paris"
        bh = loss*46
        s += rect(x, ty - 12 - bh, tw, bh, fill=(REDFILL if red else "#ece7dd"),
                  stroke=(RED if red else "#cfcabf"), sw=1)
        s += text(x + tw/2, ty - 12 - bh - 5, f"{loss:.2f}", size=8, col=(RED if red else MUTE), family=MONO)
        s += rect(x, ty, tw, th, fill=(REDFILL if red else PAPER),
                  stroke=(RED if red else CHARCOAL), sw=(1.4 if red else 1), rx=4)
        s += text(x + tw/2, ty + th/2, tok, size=9.5, col=(RED if red else INK), family=MONO)
    s += text(x0 + 3*(tw+gap), 70, "loss per response token", size=9.5, col=MUTE, style="italic")
    s += text(x0 + 5*(tw+gap) + tw/2, ty + th + 16, "biggest signal", size=9, col=RED)
    s += text(W/2, H - 14, "response (loss computed here)  —  SFT loss = average over these six = 0.45",
              size=10, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-response-masking.svg", s)


# ============================================================================
# Figure 14.2 — LoRA
# ============================================================================
def fig_lora():
    W, H = 620, 280
    s = header(W, H)
    s += text(W/2, 30, "LoRA freezes the big weight matrix and trains a small low-rank patch.",
              size=11.5, col=CHARCOAL)
    midY = 148
    wx, ws = 44, 116
    s += rect(wx, midY - ws/2, ws, ws, fill=GRAYFILL, stroke=CHARCOAL, sw=1.2)
    s += text(wx + ws/2, midY, "W", size=22, col=CHARCOAL, weight="bold")
    s += text(wx + ws/2, midY - ws/2 - 10, "pre-trained, frozen", size=9.5, col=MUTE, style="italic")
    s += text(wx + ws/2, midY + ws/2 + 16, "d × d", size=9.5, col=MUTE, family=MONO)
    s += text(wx + ws + 24, midY, "+", size=22, col=INK)
    # B (tall thin) x A (wide thin)
    bx, bw, bh = wx + ws + 44, 20, 116
    s += rect(bx, midY - bh/2, bw, bh, fill=REDFILL, stroke=RED, sw=1.3)
    s += text(bx + bw/2, midY, "B", size=13, col=RED, weight="bold")
    s += text(bx + bw/2, midY + bh/2 + 14, "d × r", size=9, col=RED, family=MONO)
    s += text(bx + bw + 14, midY, "×", size=16, col=INK)
    ax, aw, ah = bx + bw + 28, 116, 20
    s += rect(ax, midY - ah/2, aw, ah, fill=REDFILL, stroke=RED, sw=1.3)
    s += text(ax + aw/2, midY, "A", size=13, col=RED, weight="bold")
    s += text(ax + aw/2, midY - ah/2 - 8, "r × d", size=9, col=RED, family=MONO)
    s += text(ax + aw + 20, midY, "=", size=20, col=INK)
    ux, us = ax + aw + 40, 116
    s += rect(ux, midY - us/2, us, us, fill="#ffffff", stroke=RED, sw=1.3)
    s += text(ux + us/2, midY, "W + BA", size=13, col=RED, family=MONO)
    s += text(ux + us/2, midY + us/2 + 16, "updated weights", size=9.5, col=MUTE)
    s += text(W/2, H - 14,
              "r far smaller than d, so only B and A are trained — about 100M parameters instead of 70B.",
              size=10.5, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-lora.svg", s)


ALL = [fig_response_masking, fig_lora]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
