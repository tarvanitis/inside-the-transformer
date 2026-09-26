#!/usr/bin/env python3
"""Generate the Chapter 10 (build a GPT) figures as static SVGs.

One figure: the tensor-shape ladder for the multi-head reshape. Shared helpers
live in svgkit.py.

Usage:
    python3 tools/figures/gen_ch10_figures.py [--png]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch10"


# ============================================================================
# Figure 10.1 — Multi-head reshape: the shape ladder
# ============================================================================
def fig_shape_ladder():
    W, H = 680, 210
    s = header(W, H)
    s += text(W/2, 30, "How one attention call reshapes the tensor to run H heads in parallel.",
              size=11.5, col=CHARCOAL)
    chips = [("(B, T, C)", None),
             ("(B, T, H, d_k)", "view"),
             ("(B, H, T, d_k)", "transpose"),
             ("(B, H, T, d_k)", "attention"),
             ("(B, T, C)", "merge")]
    cw, gap, cy, ch = 92, 38, 96, 40
    x0 = 28
    for i, (shape, op) in enumerate(chips):
        cx = x0 + i*(cw + gap)
        edge = i in (0, len(chips)-1)
        s += rect(cx, cy, cw, ch, fill=(REDFILL if edge else PAPER),
                  stroke=(RED if edge else CHARCOAL), sw=1.2, rx=5)
        s += text(cx + cw/2, cy + ch/2, shape, size=10.5, family=MONO, col=INK)
        if op:
            ax = x0 + (i-1)*(cw + gap) + cw
            s += arrow(ax + 2, cy + ch/2, cx - 2, cy + ch/2, col=GRAY, marker="gray", w=1.4)
            s += text((ax + cx)/2, cy - 8, op, size=9.5, col=RED, style="italic")
    s += text(x0 + cw/2, cy + ch + 22, "one vector per token", size=9, col=MUTE, style="italic")
    s += text(x0 + 2*(cw+gap) + cw/2, cy + ch + 22, "H heads, each of width d_k", size=9, col=MUTE, style="italic")
    s += text(W/2, H - 12, "C = H × d_k, so no numbers are lost — the vector is only regrouped.",
              size=10, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-shape-ladder.svg", s)


ALL = [fig_shape_ladder]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
