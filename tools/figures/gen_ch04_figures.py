#!/usr/bin/env python3
"""Generate the Chapter 4 (positional encoding) figures as static SVGs.

Shared drawing helpers live in svgkit.py. Content-based filenames; the in-book
Figure 4.x number lives only in the caption.

Frequency convention follows the standard sinusoidal formula
PE(pos, 2i) = sin(pos / 10000^(2i/d)): LOW dimension index -> short wavelength
(fast), HIGH index -> long wavelength (slow).

Usage:
    python3 tools/figures/gen_ch04_figures.py            # write SVGs
    python3 tools/figures/gen_ch04_figures.py --png      # also render PNG previews
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch04"

STEEL = "#33577B"   # negative end of the diverging heat scale


def _lerp_hex(c1, c2, t):
    a = [int(c1[i:i+2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i+2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(a[i] + (b[i]-a[i])*t)) for i in range(3))

def _heat(v):   # v in [-1, 1] -> steel(-)..white(0)..red(+)
    v = max(-1.0, min(1.0, v))
    return _lerp_hex("#ffffff", RED, v) if v >= 0 else _lerp_hex("#ffffff", STEEL, -v)


# ============================================================================
# Figure 4.1 — The clock analogy: position as a set of hands at different speeds
# ============================================================================
def fig_clock_analogy():
    W, H = 600, 330
    s = header(W, H)
    cols = [("fast hand", "low dim", 2.6, 165),
            ("medium hand", "middle dim", 1.1, 320),
            ("slow hand", "high dim", 0.4, 475)]
    rows = [("position 2", 2, 110), ("position 5", 5, 235)]
    R = 40
    for name, dim, omega, cx in cols:
        s += text(cx, 40, name, size=12.5, col=CHARCOAL, weight="bold")
        s += text(cx, 56, f"({dim})", size=10.5, col=MUTE, style="italic")
    for rlabel, pos, cy in rows:
        s += text(70, cy, rlabel, size=12, col=INK, anchor="middle", weight="bold")
        for name, dim, omega, cx in cols:
            s += circle(cx, cy, R, fill=CREAM, stroke=RED, sw=1.3)
            for tick in range(12):
                a = math.radians(tick*30)
                s += line(cx + (R-4)*math.cos(a), cy - (R-4)*math.sin(a),
                          cx + R*math.cos(a), cy - R*math.sin(a), col="#d8c9b0", w=0.8)
            s += circle(cx, cy, 2.4, fill=INK)
            th = omega*pos
            s += arrow(cx, cy, cx + (R-8)*math.cos(th), cy - (R-8)*math.sin(th),
                       col=RED, marker="red", w=2.2)
    s += text(W/2, H - 20,
              "Each dimension is one hand. The whole set of hand angles is a signature unique to a position;",
              size=11, col=MUTE, style="italic")
    s += text(W/2, H - 6,
              "nearby positions give nearby signatures (the fast hands separate them).",
              size=11, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-clock-analogy.svg", s)


# ============================================================================
# Figure 4.2 — Sinusoidal positional-encoding heatmap
# ============================================================================
def fig_sinusoidal_heatmap():
    W, H = 600, 360
    s = header(W, H)
    D, N = 24, 20                 # dimensions (cols), positions (rows)
    gx, gy = 90, 60
    cw, ch = 460/D, 240/N
    for pos in range(N):
        for d in range(D):
            pair = d // 2
            freq = 1.0 / (10000 ** (2*pair / D))
            v = math.sin(pos*freq) if d % 2 == 0 else math.cos(pos*freq)
            s += (f'<rect x="{gx + d*cw:.2f}" y="{gy + pos*ch:.2f}" width="{cw:.2f}" '
                  f'height="{ch:.2f}" fill="{_heat(v)}" stroke="none"/>\n')
    s += rect(gx, gy, D*cw, N*ch, fill="none", stroke=CHARCOAL, sw=1.1)
    # axes
    s += text(gx + D*cw/2, gy + N*ch + 26, "dimension  (0 → d_model)", size=12, col=INK)
    s += text(gx + cw, gy + N*ch + 42, "fast, short wavelength", size=10, col=MUTE, anchor="start", style="italic")
    s += text(gx + D*cw - cw, gy + N*ch + 42, "slow, long wavelength", size=10, col=MUTE, anchor="end", style="italic")
    s += ('<text x="%.1f" y="%.1f" font-size="12" fill="%s" text-anchor="middle" '
          'font-family="%s" transform="rotate(-90 %.1f %.1f)">position  (0 ↓)</text>\n'
          % (gx - 24, gy + N*ch/2, INK, SERIF, gx - 24, gy + N*ch/2))
    # legend colorbar
    lx, ly, lw = gx + D*cw + 24, gy + 10, 16
    steps = 40
    for k in range(steps):
        v = 1 - 2*k/(steps-1)          # +1 at top .. -1 at bottom
        s += (f'<rect x="{lx}" y="{ly + k*(220/steps):.2f}" width="{lw}" '
              f'height="{220/steps + 0.5:.2f}" fill="{_heat(v)}" stroke="none"/>\n')
    s += rect(lx, ly, lw, 220, fill="none", stroke=CHARCOAL, sw=0.9)
    for val, yy in ((" +1", ly), (" 0", ly + 110), (" −1", ly + 220)):
        s += text(lx + lw + 3, yy, val, size=10, col=MUTE, anchor="start", family=MONO)
    s += text(W/2, 34, "Each position (a row) has a unique pattern across the dimensions.",
              size=12, col=CHARCOAL)
    s += footer()
    save(OUT, "fig-sinusoidal-heatmap.svg", s)


# ============================================================================
# Figure 4.3 — RoPE: the score depends on the angle gap (relative position)
# ============================================================================
def fig_rope_circle():
    W, H = 600, 360
    s = header(W, H)
    cx, cy, R = 250, 200, 130
    # unit circle + axes
    s += (f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="{GRID}" stroke-width="1.2"/>\n')
    s += line(cx - R - 16, cy, cx + R + 16, cy, col=GRID, w=1)
    s += line(cx, cy - R - 16, cx, cy + R + 16, col=GRID, w=1)
    s += circle(cx, cy, 2.6, fill=INK)
    def vec(rad, col, mk, lab, dash=None, lr=1.0):
        x2 = cx + R*math.cos(rad); y2 = cy - R*math.sin(rad)
        out = arrow(cx, cy, x2, y2, col=col, marker=mk, w=2.4, dash=dash)
        out += text(x2 + 10*math.cos(rad), y2 - 10*math.sin(rad), lab, size=12, col=col,
                    family=MONO, anchor="start" if math.cos(rad) >= -0.2 else "end")
        return out
    # k at pos 3 (1.5 rad), q at pos 2 (1.0 rad), q at pos 5 (2.5 rad)
    s += vec(1.0, RED, "red", "q, pos 2")
    s += vec(1.5, CHARCOAL, "char", "k, pos 3")
    s += vec(2.5, RED, "red", "q, pos 5", dash="6,3")
    # gap arcs
    s += arc(cx, cy, 52, math.degrees(1.0), math.degrees(1.5), col=MUTE, w=1.6)
    s += arc(cx, cy, 78, math.degrees(1.5), math.degrees(2.5), col=MUTE, w=1.6)
    s += text(cx + 44, cy - 44, "0.5", size=11, col=MUTE, family=MONO)
    s += text(cx - 30, cy - 86, "1.0", size=11, col=MUTE, family=MONO)
    # score readouts
    bx = 452
    s += text(bx, 150, "1 position apart", size=12, col=INK, anchor="middle", weight="bold")
    s += text(bx, 168, "gap 0.5 rad", size=11, col=MUTE, anchor="middle", family=MONO)
    s += text(bx, 186, "score = cos 0.5", size=11.5, col=RED, anchor="middle", family=MONO)
    s += text(bx, 202, "= 0.88", size=11.5, col=RED, anchor="middle", family=MONO, weight="bold")
    s += text(bx, 236, "2 positions apart", size=12, col=INK, anchor="middle", weight="bold")
    s += text(bx, 254, "gap 1.0 rad", size=11, col=MUTE, anchor="middle", family=MONO)
    s += text(bx, 272, "score = cos 1.0", size=11.5, col=RED, anchor="middle", family=MONO)
    s += text(bx, 288, "= 0.54", size=11.5, col=RED, anchor="middle", family=MONO, weight="bold")
    s += text(W/2, 32, "Rotate q and k by their positions; the dot product depends only on the gap.",
              size=12, col=CHARCOAL)
    s += text(cx, cy + R + 34, "the same score wherever the pair sits, as long as the gap is the same",
              size=10.5, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-rope-circle.svg", s)


ALL = [fig_clock_analogy, fig_sinusoidal_heatmap, fig_rope_circle]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
