#!/usr/bin/env python3
"""Generate the Chapter 20 (offline / local inference) figures as static SVGs.

Two figures: the memory-bandwidth wall, and the quantization size/quality
trade-off. Shared helpers in svgkit.py.

Usage:
    python3 tools/figures/gen_ch20_figures.py [--png]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch20"


# ============================================================================
# Figure 20.1 — The memory-bandwidth wall
# ============================================================================
def fig_bandwidth_wall():
    W, H = 620, 260
    s = header(W, H)
    s += text(W/2, 30, "To make one token, the CPU reads the whole model from RAM.", size=12, col=CHARCOAL)
    s += text(W/2, 50, "tokens/sec  ≈  memory bandwidth  ÷  model size", size=11.5, col=RED, family=MONO)
    rows = [("FP16  —  14 GB", 14.0, 3.5, GRAYFILL, CHARCOAL, "char"),
            ("Q4_K_M  —  4.1 GB", 4.1, 12.0, REDFILL, RED, "red")]
    x0, y, bh = 60, 92, 40
    sc = 380/14.0
    for i, (lab, size, tps, fill, st, mk) in enumerate(rows):
        yy = y + i*76
        s += text(x0, yy - 8, lab, size=10.5, col=INK, anchor="start", family=MONO)
        s += rect(x0, yy, size*sc, bh, fill=fill, stroke=st, sw=1.2)
        s += text(x0 + size*sc/2, yy + bh/2, f"stream {size:g} GB / token", size=9.5,
                  col=(RED if st == RED else INK))
        s += arrow(x0 + size*sc + 6, yy + bh/2, x0 + size*sc + 46, yy + bh/2, col=st, marker=mk, w=1.5)
        s += text(x0 + size*sc + 52, yy + bh/2, f"≈ {tps:g} tok/s", size=11.5, col=st,
                  anchor="start", weight="bold")
    s += text(W/2, H - 14,
              "At ~50 GB/s: 50÷14 ≈ 3.5, 50÷4.1 ≈ 12. A smaller model is less to stream, so it runs faster.",
              size=10, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-bandwidth-wall.svg", s)


# ============================================================================
# Figure 20.2 — Quantization size/quality trade-off
# ============================================================================
def fig_quant_tradeoff():
    W, H = 600, 320
    s = header(W, H)
    s += text(W/2, 30, "Quantization trade-off for a 7B model: quality holds down to ~4 bits, then drops.",
              size=11, col=CHARCOAL)
    x0, y0 = 80, 258
    xlo, xhi = 2.0, 15.0
    ylo, yhi = 60.0, 104.0
    px = 460/(xhi - xlo)
    py = 190/(yhi - ylo)
    def X(gb): return x0 + (gb - xlo)*px
    def Y(q): return y0 - (q - ylo)*py
    # axes
    s += arrow(x0, y0, x0 + 475, y0, col=GRAY, marker="gray", w=1.4)
    s += arrow(x0, y0, x0, y0 - 205, col=GRAY, marker="gray", w=1.4)
    s += text(x0 + 235, y0 + 32, "model size on disk (GB) →", size=11, col=MUTE)
    s += ('<text x="%.1f" y="%.1f" font-size="11" fill="%s" text-anchor="middle" font-family="%s" '
          'transform="rotate(-90 %.1f %.1f)">quality (relative)</text>\n'
          % (x0 - 42, Y(82), MUTE, SERIF, x0 - 42, Y(82)))
    for q in (60, 80, 100):
        s += line(x0, Y(q), x0 + 475, Y(q), col=GRID, w=1, dash="2,3")
        s += text(x0 - 8, Y(q), str(q), size=9, col=MUTE, anchor="end", family=MONO)
    for gb in (2, 5, 10, 14):
        s += text(X(gb), y0 + 15, str(gb), size=9, col=MUTE, family=MONO)
    # points (size, quality, label, is_sweet)
    pts = [(2.7, 68, "Q2_K", False), (3.3, 85, "Q3_K_M", False), (4.1, 95, "Q4_K_M", True),
           (4.9, 98, "Q5_K_M", True), (5.8, 99, "Q6_K", False), (7.5, 100, "Q8_0", False),
           (14.0, 100, "FP16", False)]
    poly = " ".join(f"{X(gb):.1f},{Y(q):.1f}" for gb, q, _, _ in pts)
    s += f'<polyline points="{poly}" fill="none" stroke="{CHARCOAL}" stroke-width="2"/>\n'
    for gb, q, lab, sweet in pts:
        s += circle(X(gb), Y(q), 4.2, fill=(RED if sweet else "#ffffff"), stroke=(RED if sweet else CHARCOAL), sw=1.5)
        dy = 16 if lab in ("Q4_K_M",) else -12
        s += text(X(gb), Y(q) + dy, lab, size=8.5, col=(RED if sweet else CHARCOAL), family=MONO,
                  anchor="middle")
    # sweet-spot band
    s += (f'<rect x="{X(3.9):.1f}" y="{Y(104):.1f}" width="{(X(5.0)-X(3.9)):.1f}" '
          f'height="{(Y(60)-Y(104)):.1f}" fill="{RED}" opacity="0.06"/>\n')
    s += text(X(4.5), Y(74), "sweet spot", size=9.5, col=RED, style="italic")
    s += text(X(11), Y(96), "bigger, barely better", size=9.5, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-quant-tradeoff.svg", s)


ALL = [fig_bandwidth_wall, fig_quant_tradeoff]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
