#!/usr/bin/env python3
"""Generate the Chapter 6 (transformer block) figures as static SVGs.

The chapter's assembled-block diagram is a mermaid figure that now sits last in
the figure order (renumbered to 6.4); these three static figures are 6.1-6.3 and
appear before it. Shared helpers live in svgkit.py.

Usage:
    python3 tools/figures/gen_ch06_figures.py            # write SVGs
    python3 tools/figures/gen_ch06_figures.py --png      # also render PNG previews
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch06"


# ============================================================================
# Figure 6.1 — Feed-forward network: expand -> activate -> compress
# ============================================================================
def fig_ffn_shape():
    W, H = 620, 320
    s = header(W, H)
    midY = 178
    inX, inW, inHH = 80, 26, 34
    hidX, hidW, hidHH = 285, 36, 98
    outX, outW, outHH = 500, 26, 34

    def bar(x, w, hh, fill, stroke):
        return rect(x, midY - hh, w, 2*hh, fill=fill, stroke=stroke, sw=1.2, rx=3)

    # W1 expand trapezoid
    s += (f'<polygon points="{inX+inW},{midY-inHH} {hidX},{midY-hidHH} {hidX},{midY+hidHH} '
          f'{inX+inW},{midY+inHH}" fill="{REDFILL}" stroke="{RED}" stroke-width="1" opacity="0.5"/>\n')
    # W2 compress trapezoid
    s += (f'<polygon points="{hidX+hidW},{midY-hidHH} {outX},{midY-outHH} {outX},{midY+outHH} '
          f'{hidX+hidW},{midY+hidHH}" fill="{REDFILL}" stroke="{RED}" stroke-width="1" opacity="0.5"/>\n')
    # bars
    s += bar(inX, inW, inHH, PAPER, CHARCOAL)
    s += bar(hidX, hidW, hidHH, CREAM, RED)
    s += bar(outX, outW, outHH, PAPER, CHARCOAL)
    # ReLU glyph inside hidden bar (flat then rising)
    cx = hidX + hidW/2
    s += (f'<polyline points="{cx-11},{midY+9} {cx},{midY+9} {cx+11},{midY-9}" '
          f'fill="none" stroke="{RED}" stroke-width="2"/>\n')
    # labels
    s += text((inX+inW+hidX)/2 - 6, midY - 44, "W₁  expand", size=11, col=RED)
    s += text((hidX+hidW+outX)/2 + 6, midY - 44, "W₂  compress", size=11, col=RED)
    s += text(cx, midY - hidHH - 12, "nonlinearity", size=10.5, col=MUTE, style="italic")
    s += text(inX+inW/2, midY + inHH + 20, "d_model", size=11, col=INK, family=MONO)
    s += text(cx, midY + hidHH + 20, "d_ff ≈ 4 × d_model", size=11, col=INK, family=MONO)
    s += text(outX+outW/2, midY + outHH + 20, "d_model", size=11, col=INK, family=MONO)
    s += text(W/2, 28,
              "The feed-forward network widens each token's vector, adds a nonlinearity, then compresses it back.",
              size=11.5, col=CHARCOAL)
    s += footer()
    save(OUT, "fig-ffn-shape.svg", s)


# ============================================================================
# Figure 6.2 — The residual stream as a highway
# ============================================================================
def fig_residual_stream():
    W, H = 640, 240
    s = header(W, H)
    hy = 92
    s += text(W/2, 32, "Each sub-layer reads from the stream, computes a correction, and adds it back.",
              size=11.5, col=CHARCOAL)
    # the highway
    s += arrow(36, hy, 604, hy, col=RED, marker="red", w=3.2)
    s += text(30, hy - 14, "x₀", size=12, col=RED, family=MONO, anchor="start")
    s += text(600, hy - 14, "x_final", size=12, col=RED, family=MONO, anchor="end")
    s += text(150, hy - 14, "residual stream", size=11, col=RED, style="italic", anchor="middle")

    def detour(readx, addx, label):
        out = ""
        boxy, bw, bh = 150, 128, 34
        bx = (readx + addx)/2 - bw/2
        out += line(readx, hy, bx + 8, boxy + bh/2, col=CHARCOAL, w=1.4)          # read down
        out += rect(bx, boxy, bw, bh, fill=PAPER, stroke=CHARCOAL, sw=1.1, rx=5)
        out += text(bx + bw/2, boxy + bh/2, label, size=11.5, col=INK)
        out += arrow(bx + bw - 8, boxy + bh/2, addx, hy + 6, col=CHARCOAL, marker="char", w=1.4)  # write up
        out += circle(addx, hy, 9, fill="#ffffff", stroke=RED, sw=1.6)
        out += text(addx, hy, "+", size=13, col=RED, weight="bold")
        return out
    s += detour(150, 250, "Multi-head attention")
    s += detour(380, 480, "Feed-forward")
    s += text(W/2, H - 14,
              "The line runs straight through each +, so the signal (and its gradient) survives many layers.",
              size=10.5, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-residual-stream.svg", s)


# ============================================================================
# Figure 6.3 — Pre-norm vs post-norm
# ============================================================================
def fig_pre_post_norm():
    W, H = 640, 320
    s = header(W, H)

    def box(cx, y, label, w=118, h=30, fill=PAPER, stroke=CHARCOAL, tc=INK, bold="normal"):
        return (rect(cx - w/2, y, w, h, fill=fill, stroke=stroke, sw=1.2, rx=5)
                + text(cx, y + h/2, label, size=11.5, col=tc, weight=bold))

    def node(cx, y):
        return circle(cx, y, 9, fill="#ffffff", stroke=RED, sw=1.6) + text(cx, y, "+", size=13, col=RED, weight="bold")

    def down(cx, y1, y2):
        return arrow(cx, y1, cx, y2, col=GRAY, marker="gray", w=1.5)

    def skip(cx, skipx, ytop, yadd):
        # x branches right, down, and into the + node
        return (line(cx, ytop, skipx, ytop, col=CHARCOAL, w=1.3)
                + line(skipx, ytop, skipx, yadd, col=CHARCOAL, w=1.3)
                + arrow(skipx, yadd, cx + 9, yadd, col=CHARCOAL, marker="char", w=1.3))

    # ---- post-norm (left) ----
    s += panel(16, 20, 300, 300, "Post-norm  (original 2017)")
    cx = 128; skipx = 250
    s += text(cx, 70, "x", size=13, col=RED, family=MONO)
    s += down(cx, 78, 92)
    s += box(cx, 92, "Attention")
    s += down(cx, 122, 150)
    s += node(cx, 159)
    s += down(cx, 168, 196)
    s += box(cx, 196, "Norm", fill=REDFILL, stroke=RED, tc=RED, bold="bold")
    s += down(cx, 226, 250)
    s += text(cx, 262, "out", size=12, col=INK, family=MONO)
    s += skip(cx, skipx, 70, 159)
    s += text(cx, 300, "Norm wraps the residual add", size=10, col=MUTE, style="italic")

    # ---- pre-norm (right) ----
    s += panel(336, 20, 288, 300, "Pre-norm  (modern default)")
    cx2 = 440; skipx2 = 566
    s += text(cx2, 70, "x", size=13, col=RED, family=MONO)
    s += down(cx2, 78, 92)
    s += box(cx2, 92, "Norm", fill=REDFILL, stroke=RED, tc=RED, bold="bold")
    s += down(cx2, 122, 140)
    s += box(cx2, 140, "Attention")
    s += down(cx2, 170, 205)
    s += node(cx2, 214)
    s += down(cx2, 223, 250)
    s += text(cx2, 262, "out", size=12, col=INK, family=MONO)
    s += skip(cx2, skipx2, 70, 214)
    s += text(cx2, 300, "the skip carries the un-normalized x", size=10, col=MUTE, style="italic")

    s += footer()
    save(OUT, "fig-pre-post-norm.svg", s)


def fig_block():
    W, H = 440, 356
    s = header(W, H)
    cx, sx = 160, 320
    s += text(cx, 30, "x  (residual stream)", size=11.5, col=RED, family=MONO)
    rows = [("RMSNorm", 74, REDFILL, RED),
            ("Multi-head attention", 118, PAPER, CHARCOAL),
            ("+", 166, None, None),
            ("RMSNorm", 212, REDFILL, RED),
            ("Feed-forward (SwiGLU)", 256, PAPER, CHARCOAL),
            ("+", 304, None, None)]
    ys = {}
    for lab, y, fill, st in rows:
        ys[lab if lab != "+" else f"+{y}"] = y
        if lab == "+":
            s += circle(cx, y, 11, fill="#ffffff", stroke=RED, sw=1.6)
            s += text(cx, y, "+", size=15, col=RED, weight="bold")
        else:
            s += node(cx, y, 190, 30, lab, fill=fill, stroke=st, size=10.5)
    s += text(cx, 342, "x  (updated)", size=11.5, col=RED, family=MONO)
    # main downward arrows
    seq = [40, 74, 118, 166, 212, 256, 304, 342]
    tops = [59, 74-15, 118-15, 166-11, 212-15, 256-15, 304-11, 342-6]
    # draw arrows between consecutive element boundaries
    pairs = [(46, 59), (89, 103), (133, 155), (177, 197), (227, 241), (271, 293), (315, 334)]
    for y1, y2 in pairs:
        s += arrow(cx, y1, cx, y2, col=GRAY, marker="gray", w=1.5)
    # residual skips on the right
    def skip(y_from, y_to, label):
        out = line(cx, y_from, sx, y_from, col=CHARCOAL, w=1.3, dash="4,3")
        out += line(sx, y_from, sx, y_to, col=CHARCOAL, w=1.3, dash="4,3")
        out += arrow(sx, y_to, cx + 11, y_to, col=CHARCOAL, marker="char", w=1.3, dash="4,3")
        out += text(sx + 8, (y_from + y_to)/2, label, size=8.5, col=MUTE, anchor="start", style="italic")
        return out
    s += skip(52, 166, "skip")
    s += skip(190, 304, "skip")
    s += footer()
    save(OUT, "fig-block.svg", s)


ALL = [fig_ffn_shape, fig_residual_stream, fig_pre_post_norm, fig_block]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
