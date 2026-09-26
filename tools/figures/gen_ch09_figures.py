#!/usr/bin/env python3
"""Generate the Chapter 9 (architecture trends) figures as static SVGs.

Chapter 9 already has Figures 9.1-9.2 (mermaid); these are 9.3-9.4.
Shared helpers live in svgkit.py.

Usage:
    python3 tools/figures/gen_ch09_figures.py [--png]
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch09"


# ============================================================================
# Figure 9.3 — Mixture-of-Experts routing
# ============================================================================
def fig_moe_routing():
    W, H = 680, 320
    s = header(W, H)
    s += text(W/2, 28, "Mixture-of-Experts: a router sends each token to only its top-k experts.",
              size=11.5, col=CHARCOAL)
    # token + router
    s += rect(30, 128, 92, 32, fill=CREAM, stroke=RED, sw=1.2, rx=5)
    s += text(76, 144, '"programming"', size=10, family=MONO, col=INK)
    s += arrow(122, 144, 150, 144, col=GRAY, marker="gray", w=1.5)
    s += rect(152, 126, 60, 36, fill=PAPER, stroke=CHARCOAL, sw=1.2, rx=5)
    s += text(182, 144, "router", size=11, col=INK)
    # experts with score bars
    scores = [0.3, 0.1, 0.8, 0.2, 0.7, 0.1, 0.05, 0.15]
    top = {2, 4}
    ex0, ew, eg, ey = 262, 42, 9, 120
    s += arrow(212, 144, ex0 - 6, ey + 30, col=GRAY, marker="gray", w=1.2)
    for i, sc in enumerate(scores):
        ex = ex0 + i*(ew + eg)
        red = i in top
        bh = sc*46
        s += rect(ex, ey - bh, ew, bh, fill=(REDFILL if red else "#ece7dd"),
                  stroke=(RED if red else "#cfcabf"), sw=1)
        s += text(ex + ew/2, ey - bh - 6, f"{sc:.2f}", size=8, col=(RED if red else MUTE), family=MONO)
        s += rect(ex, ey + 10, ew, 40, fill=(REDFILL if red else "#f4f1ea"),
                  stroke=(RED if red else "#d8d2c6"), sw=(1.5 if red else 0.8), rx=3)
        s += text(ex + ew/2, ey + 30, f"E{i}", size=10, col=(RED if red else MUTE), family=MONO)
    s += text(ex0 + 4*(ew + eg) - eg, ey - 58, "router scores over experts", size=10, col=MUTE, style="italic")
    # combine top-2
    plusx, plusy = ex0 + 3*(ew + eg) + ew/2, ey + 118
    for i in top:
        ex = ex0 + i*(ew + eg) + ew/2
        s += arrow(ex, ey + 50, plusx, plusy - 9, col=RED, marker="red", w=1.4)
    s += circle(plusx, plusy, 11, fill="#ffffff", stroke=RED, sw=1.7)
    s += text(plusx, plusy, "+", size=15, col=RED, weight="bold")
    s += arrow(plusx, plusy + 11, plusx, plusy + 32, col=RED, marker="red", w=1.5)
    s += rect(plusx - 95, plusy + 34, 190, 30, fill=CREAM, stroke=RED, sw=1.2, rx=5)
    s += text(plusx, plusy + 49, "0.53·E2 + 0.47·E4", size=11, col=RED, family=MONO)
    s += text(W/2, H - 12,
              "Only 2 of 8 experts run for this token, so total capacity grows without growing the per-token cost.",
              size=10, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-moe-routing.svg", s)


# ============================================================================
# Figure 9.4 — Test-time compute
# ============================================================================
def fig_testtime():
    W, H = 560, 300
    s = header(W, H)
    s += text(W/2, 30, "Spending more compute per query: think longer, answer better.", size=12, col=CHARCOAL)
    f = Frame(ox=80, oy=250, sx=42, sy=190)   # x in [0,10], y in [0,1]
    # axes
    s += arrow(f.X(0), f.Y(0), f.X(10.4), f.Y(0), col=GRAY, marker="gray", w=1.4)
    s += arrow(f.X(0), f.Y(0), f.X(0), f.Y(1.08), col=GRAY, marker="gray", w=1.4)
    s += text(f.X(5.2), f.Y(0) + 30, "reasoning tokens spent per query →", size=11, col=MUTE)
    s += ('<text x="%.1f" y="%.1f" font-size="11" fill="%s" text-anchor="middle" font-family="%s" '
          'transform="rotate(-90 %.1f %.1f)">accuracy on hard problems</text>\n'
          % (f.X(0) - 34, f.Y(0.5), MUTE, SERIF, f.X(0) - 34, f.Y(0.5)))
    for yv in (0.5, 1.0):
        s += line(f.X(0), f.Y(yv), f.X(10), f.Y(yv), col=GRID, w=1, dash="2,3")
        s += text(f.X(0) - 8, f.Y(yv), f"{yv:.1f}", size=9, col=MUTE, anchor="end", family=MONO)
    base, cap = 0.30, 0.92
    # "answer immediately" baseline
    s += line(f.X(0), f.Y(base), f.X(10), f.Y(base), col=CHARCOAL, w=1.3, dash="5,3")
    s += text(f.X(10), f.Y(base) - 8, "answer immediately", size=10, col=CHARCOAL, anchor="end")
    # rising curve
    pts = []
    x = 0.0
    while x <= 10.0001:
        y = base + (cap - base)*(1 - math.exp(-x/2.2))
        pts.append(f"{f.X(x):.2f},{f.Y(y):.2f}"); x += 0.2
    s += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{RED}" stroke-width="2.6"/>\n'
    s += text(f.X(7.2), f.Y(base + (cap-base)*(1-math.exp(-7.2/2.2))) - 12,
              "with a chain of thought", size=10.5, col=RED)
    s += footer()
    save(OUT, "fig-test-time-compute.svg", s)


def _dashrect(x, y, w, h, col=RED):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="none" '
            f'stroke="{col}" stroke-width="1.1" stroke-dasharray="5,4"/>\n')


def fig_decoder_only():
    W, H = 380, 520
    s = header(W, H)
    s += text(W/2, 22, "The modern decoder-only architecture, end to end.", size=11, col=CHARCOAL)
    cx = 176

    def B(y, lab, fill=PAPER, st=CHARCOAL, tc=INK):
        return node(cx, y, 214, 26, lab, fill=fill, stroke=st, tc=tc, size=9.5)

    def P(y):
        return circle(cx, y, 10, fill="#ffffff", stroke=RED, sw=1.5) + text(cx, y, "+", size=13, col=RED, weight="bold")
    s += _dashrect(42, 116, 268, 214)
    s += text(302, 128, "× N", size=10, col=RED, anchor="end", weight="bold")
    rows = [(44, "Input tokens → IDs", CREAM, RED), (82, "Token embedding", PAPER, CHARCOAL),
            (142, "RMSNorm", REDFILL, RED), (178, "Multi-head attention (causal)", PAPER, CHARCOAL),
            (212, "+", None, None), (246, "RMSNorm", REDFILL, RED),
            (280, "Feed-forward (SwiGLU)", PAPER, CHARCOAL), (314, "+", None, None),
            (366, "Final RMSNorm", REDFILL, RED), (402, "LM head", PAPER, CHARCOAL),
            (438, "Softmax over vocabulary", PAPER, CHARCOAL), (474, "Next-token probabilities", CREAM, RED)]
    ys = [r[0] for r in rows]
    for y, lab, fill, st in rows:
        s += P(y) if lab == "+" else B(y, lab, fill=fill, st=st)
    for y1, y2 in zip(ys, ys[1:]):
        a = y1 + (10 if rows[ys.index(y1)][1] == "+" else 13)
        b = y2 - (10 if rows[ys.index(y2)][1] == "+" else 13)
        s += arrow(cx, a, cx, b, col=GRAY, marker="gray", w=1.4)
    # residual skips
    s += line(cx, 95, 298, 95, col=CHARCOAL, w=1.2, dash="4,3")
    s += line(298, 95, 298, 212, col=CHARCOAL, w=1.2, dash="4,3")
    s += arrow(298, 212, cx + 10, 212, col=CHARCOAL, marker="char", w=1.2, dash="4,3")
    s += line(cx, 224, 54, 224, col=CHARCOAL, w=1.2, dash="4,3")
    s += line(54, 224, 54, 314, col=CHARCOAL, w=1.2, dash="4,3")
    s += arrow(54, 314, cx - 10, 314, col=CHARCOAL, marker="char", w=1.2, dash="4,3")
    s += footer()
    save(OUT, "fig-decoder-only.svg", s)


def fig_encoder_decoder():
    W, H = 620, 420
    s = header(W, H)
    s += text(W/2, 22, "The original 2017 encoder–decoder transformer: two stacks joined by cross-attention.",
              size=10.5, col=CHARCOAL)
    ecx, dcx = 150, 452

    def B(cx, y, lab, fill=PAPER, st=CHARCOAL, tc=INK):
        return node(cx, y, 184, 24, lab, fill=fill, stroke=st, tc=tc, size=9)

    def chain(cx, items):
        out = ""
        for y, lab, fill, st in items:
            out += B(cx, y, lab, fill=fill, st=st)
        for a, b in zip(items, items[1:]):
            out += arrow(cx, a[0] + 12, cx, b[0] - 12, col=GRAY, marker="gray", w=1.4)
        return out
    # encoder
    s += text(ecx, 52, "Encoder", size=11, col=CHARCOAL, weight="bold")
    enc = [(78, "Source tokens + PE", CREAM, RED), (132, "Self-attention", PAPER, CHARCOAL),
           (164, "Add & Norm", PAPER, CHARCOAL), (196, "Feed-forward", PAPER, CHARCOAL),
           (228, "Add & Norm", PAPER, CHARCOAL), (280, "Encoder output (memory)", REDFILL, RED)]
    s += _dashrect(ecx - 100, 116, 200, 128, col=CHARCOAL)
    s += text(ecx + 92, 128, "× N", size=9.5, col=CHARCOAL, anchor="end")
    s += chain(ecx, enc)
    # decoder
    s += text(dcx, 52, "Decoder", size=11, col=CHARCOAL, weight="bold")
    dec = [(78, "Target tokens + PE", CREAM, RED), (132, "Masked self-attention", PAPER, CHARCOAL),
           (164, "Add & Norm", PAPER, CHARCOAL), (196, "Cross-attention", REDFILL, RED),
           (228, "Add & Norm", PAPER, CHARCOAL), (260, "Feed-forward", PAPER, CHARCOAL),
           (292, "Add & Norm", PAPER, CHARCOAL), (338, "Linear + Softmax", PAPER, CHARCOAL),
           (374, "Output probabilities", CREAM, RED)]
    s += _dashrect(dcx - 100, 116, 200, 192, col=CHARCOAL)
    s += text(dcx + 92, 128, "× N", size=9.5, col=CHARCOAL, anchor="end")
    s += chain(dcx, dec)
    # cross-attention edge: memory -> decoder cross-attention
    s += arrow(ecx + 92, 280, dcx - 94, 196, col=RED, marker="red", w=1.6)
    s += text((ecx + dcx)/2, 236, "K, V", size=9.5, col=RED, weight="bold")
    s += footer()
    save(OUT, "fig-encoder-decoder.svg", s)


ALL = [fig_decoder_only, fig_encoder_decoder, fig_moe_routing, fig_testtime]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
