#!/usr/bin/env python3
"""Generate the Chapter 11 (annotated transformer) figures as static SVGs.

One figure: self-attention vs cross-attention, the same module fed different
inputs. Shared helpers live in svgkit.py.

Usage:
    python3 tools/figures/gen_ch11_figures.py [--png]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch11"


def _box(cx, y, label, w=120, h=32, fill=PAPER, stroke=CHARCOAL, tc=INK, size=11):
    return (rect(cx - w/2, y, w, h, fill=fill, stroke=stroke, sw=1.2, rx=5)
            + text(cx, y + h/2, label, size=size, col=tc))


# ============================================================================
# Figure 11.1 — Self-attention vs cross-attention
# ============================================================================
def fig_self_vs_cross():
    W, H = 640, 300
    s = header(W, H)
    s += text(W/2, 28, "Self-attention and cross-attention are the same module, fed different inputs.",
              size=11.5, col=CHARCOAL)

    # ---- self-attention (left) ----
    s += panel(16, 50, 300, 232, "Self-attention")
    cx = 166
    s += _box(cx, 74, "sequence x", w=150, fill=CREAM, stroke=RED, tc=INK)
    mhaY = 190
    s += _box(cx, mhaY, "attention module", w=180, fill=PAPER, stroke=CHARCOAL)
    for dx, lab in ((-58, "Q"), (0, "K"), (58, "V")):
        s += arrow(cx + dx*0.5, 106, cx + dx, mhaY - 2, col=CHARCOAL, marker="char", w=1.5)
        s += text(cx + dx + (8 if dx >= 0 else -8), 150, lab, size=11, col=RED, family=MONO,
                  anchor="start" if dx >= 0 else "end")
    s += arrow(cx, mhaY + 32, cx, mhaY + 50, col=RED, marker="red", w=1.6)
    s += text(cx, mhaY + 60, "output", size=11, col=RED)
    s += text(cx, 262, "Q = K = V = x", size=10.5, col=MUTE, family=MONO, style="italic")

    # ---- cross-attention (right) ----
    s += panel(324, 50, 300, 232, "Cross-attention")
    lx, rx = 410, 540
    s += _box(lx, 74, "decoder state", w=120, fill=CREAM, stroke=RED, tc=INK, size=10)
    s += _box(rx, 74, "encoder output", w=120, fill=PAPER, stroke=CHARCOAL, size=10)
    cxc = 474
    mhaY2 = 190
    s += _box(cxc, mhaY2, "attention module", w=180, fill=PAPER, stroke=CHARCOAL)
    s += arrow(lx, 106, cxc - 40, mhaY2 - 2, col=RED, marker="red", w=1.6)
    s += text(lx - 6, 150, "Q", size=11, col=RED, family=MONO, anchor="end")
    s += arrow(rx - 14, 106, cxc + 30, mhaY2 - 2, col=CHARCOAL, marker="char", w=1.5)
    s += text(rx + 8, 150, "K, V", size=11, col=CHARCOAL, family=MONO, anchor="start")
    s += arrow(cxc, mhaY2 + 32, cxc, mhaY2 + 50, col=RED, marker="red", w=1.6)
    s += text(cxc, mhaY2 + 60, "output", size=11, col=RED)
    s += text(cxc, 262, "Q from decoder,  K, V from encoder", size=10.5, col=MUTE, family=MONO, style="italic")

    s += footer()
    save(OUT, "fig-self-vs-cross.svg", s)


ALL = [fig_self_vs_cross]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
