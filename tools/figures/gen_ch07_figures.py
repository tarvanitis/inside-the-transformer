#!/usr/bin/env python3
"""Generate the Chapter 7 (mechanistic interpretability) figures as static SVGs.

Chapter 7 already has Figure 7.1 (the mermaid induction-head circuit); these two
static figures make the two abstract mechanisms concrete and are numbered 7.2-7.3.
Shared helpers live in svgkit.py.

Usage:
    python3 tools/figures/gen_ch07_figures.py            # write SVGs
    python3 tools/figures/gen_ch07_figures.py --png      # also render PNG previews
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch07"


def carrow(x1, y1, x2, y2, bow, col, mk, w=1.8, dash=None):
    """Curved arrow from (x1,y1) to (x2,y2); bow>0 arcs upward, <0 downward."""
    mx, my = (x1 + x2)/2, min(y1, y2) - bow
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M{x1:.1f},{y1:.1f} Q{mx:.1f},{my:.1f} {x2:.1f},{y2:.1f}" '
            f'fill="none" stroke="{col}" stroke-width="{w}" marker-end="url(#a-{mk})"{d}/>\n')


# ============================================================================
# Figure 7.2 — An induction head in action (the "Mr Dursley" trace)
# ============================================================================
def fig_induction_trace():
    W, H = 640, 250
    s = header(W, H)
    s += text(W/2, 30, "Pattern:  … A  B  …  A  →  predict B  (what followed A last time)",
              size=11.5, col=CHARCOAL)
    toks = [("Mr", "red"), ("Dursley", "char"), ("…", None), ("Mr", "red"), ("?", "red")]
    bw = [44, 82, 28, 44, 40]
    gap, y, bh = 20, 138, 34
    xs = []
    x = 70
    for w in bw:
        xs.append(x); x += w + gap
    for (lab, kind), bx, w in zip(toks, xs, bw):
        if lab == "…":
            s += text(bx + w/2, y + bh/2, "…", size=16, col=MUTE)
            continue
        red = kind == "red"
        fill = REDFILL if red else PAPER
        st = RED if red else "#c9c4b8"
        s += rect(bx, y, w, bh, fill=fill, stroke=st, sw=1.2, rx=4)
        s += text(bx + w/2, y + bh/2, lab, size=13, col=(RED if red else INK), family=MONO)
    c0 = xs[0] + bw[0]/2          # first "Mr"
    cD = xs[1] + bw[1]/2          # "Dursley"
    c3 = xs[3] + bw[3]/2          # second "Mr"
    c4 = xs[4] + bw[4]/2          # "?"
    # (1) second Mr -> first Mr : find the earlier copy
    s += carrow(c3, y - 2, c0, y - 2, 74, RED, "red")
    s += text((c0 + c3)/2, y - 90, "① find the earlier “Mr”", size=11, col=RED)
    # (2) first Mr -> Dursley : what followed it
    s += carrow(c0, y - 2, cD, y - 2, 30, CHARCOAL, "char")
    s += text((c0 + cD)/2, y - 44, "② what followed it", size=10.5, col=CHARCOAL)
    # prediction under "?"
    s += arrow(c4, y + bh + 2, c4, y + bh + 18, col=RED, marker="red", w=1.5)
    s += rect(c4 - 42, y + bh + 20, 84, 28, fill=CREAM, stroke=RED, sw=1.2, rx=4)
    s += text(c4, y + bh + 34, "Dursley", size=12, col=RED, family=MONO, weight="bold")
    s += text(c4, y + bh + 62, "copied forward", size=9.5, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-induction-trace.svg", s)


# ============================================================================
# Figure 7.3 — A feed-forward layer as key-value memory
# ============================================================================
def fig_ffn_memory():
    W, H = 640, 300
    s = header(W, H)
    s += text(W/2, 30, "A neuron fires when the token matches its key, then writes its value back.",
              size=11.5, col=CHARCOAL)
    # input vector
    ix, iy, ih = 44, 96, 84
    s += rect(ix, iy, 24, ih, fill=PAPER, stroke=CHARCOAL, sw=1.2, rx=3)
    s += text(ix + 12, iy + ih + 15, "token", size=10, col=INK, family=MONO)
    s += text(ix + 12, iy + ih + 28, "(Eiffel Tower)", size=9, col=MUTE)
    # keys = columns of W1
    keys = ["k₁", "k₂", "k₃", "k₄"]
    firing = 1
    kx0, kw, kg, ky, kh = 190, 22, 40, 100, 84
    fx = None
    for i, k in enumerate(keys):
        kx = kx0 + i*(kw + kg)
        red = i == firing
        s += rect(kx, ky, kw, kh, fill=(REDFILL if red else "#f0eee8"),
                  stroke=(RED if red else "#c9c4b8"), sw=(1.4 if red else 0.9), rx=3)
        s += text(kx + kw/2, ky + kh + 14, k, size=10.5, col=(RED if red else MUTE), family=MONO)
        s += arrow(ix + 24, iy + ih/2, kx - 3, ky + kh/2,
                   col=(RED if red else "#dbd5c9"), marker=("red" if red else "gray"),
                   w=(1.5 if red else 0.9))
        if red:
            s += text(kx + kw/2, ky - 8, "fires", size=9.5, col=RED, weight="bold")
            fx = kx + kw
    s += text(kx0 + 1.5*(kw + kg), ky - 26, "keys = columns of W₁", size=10.5, col=MUTE, style="italic")
    # value = row of W2
    vx, vy, vw = 452, 100, 156
    s += arrow(fx + 3, ky + kh/2, vx - 3, vy + 22, col=RED, marker="red", w=1.7)
    s += rect(vx, vy, vw, 46, fill=CREAM, stroke=RED, sw=1.2, rx=5)
    s += text(vx + vw/2, vy + 16, "value = row of W₂", size=9.5, col=MUTE)
    s += text(vx + vw/2, vy + 32, "Paris · France · tall", size=11, col=RED, family=MONO)
    # residual stream
    ry = 262
    s += arrow(44, ry, 600, ry, col=RED, marker="red", w=2.6)
    s += text(44, ry - 12, "residual stream", size=10, col=RED, style="italic", anchor="start")
    s += arrow(vx + vw/2, vy + 46, vx + vw/2, ry - 5, col=RED, marker="red", w=1.5)
    s += text(vx + vw/2 + 12, (vy + 46 + ry)/2, "written back", size=9.5, col=MUTE,
              anchor="start", style="italic")
    s += footer()
    save(OUT, "fig-ffn-memory.svg", s)


def fig_induction_circuit():
    W, H = 680, 180
    s = header(W, H)
    s += text(W/2, 26, "The induction-head circuit: a previous-token head, then an induction head.",
              size=11.5, col=CHARCOAL)
    cy = 100
    ns = [(70, 88, ["token t"], CREAM, RED),
          (236, 150, ["Layer ℓ", "previous-token head", "(copy t−1 onto t)"], PAPER, CHARCOAL),
          (428, 150, ["Layer ℓ+1", "induction head", "(find earlier copy)"], REDFILL, RED),
          (606, 122, ["attend to what", "followed → predict"], CREAM, RED)]
    for cx, w, lines, fill, st in ns:
        h = 20 + len(lines)*15
        s += node(cx, cy, w, h, lines, fill=fill, stroke=st, size=10)
    for a, b in zip(ns, ns[1:]):
        s += arrow(a[0] + a[1]/2 + 2, cy, b[0] - b[1]/2 - 2, cy, col=GRAY, marker="gray", w=1.6)
    s += footer()
    save(OUT, "fig-induction-circuit.svg", s)


ALL = [fig_induction_circuit, fig_induction_trace, fig_ffn_memory]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
