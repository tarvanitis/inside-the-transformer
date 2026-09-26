#!/usr/bin/env python3
"""Generate the Chapter 17 (evaluation) figures as static SVGs.

One figure: benchmark saturation over time. Shared helpers in svgkit.py.

Usage:
    python3 tools/figures/gen_ch17_figures.py [--png]
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch17"


# ============================================================================
# Figure 17.1 — Benchmark saturation
# ============================================================================
def fig_saturation():
    W, H = 600, 300
    s = header(W, H)
    s += text(W/2, 30, "Benchmarks saturate: as models improve, harder ones must replace them.",
              size=11.5, col=CHARCOAL)
    x0, y0 = 70, 250
    xlo, xhi = 2019, 2026
    px = 470/(xhi - xlo)
    py = 190/100.0
    def X(t): return x0 + (t - xlo)*px
    def Y(a): return y0 - a*py
    # axes
    s += arrow(x0, y0, x0 + 480, y0, col=GRAY, marker="gray", w=1.4)
    s += arrow(x0, y0, x0, y0 - 205, col=GRAY, marker="gray", w=1.4)
    s += text(x0 + 240, y0 + 30, "year →", size=11, col=MUTE)
    s += ('<text x="%.1f" y="%.1f" font-size="11" fill="%s" text-anchor="middle" font-family="%s" '
          'transform="rotate(-90 %.1f %.1f)">benchmark accuracy (%%)</text>\n'
          % (x0 - 40, Y(50), MUTE, SERIF, x0 - 40, Y(50)))
    for a in (0, 50, 100):
        s += line(x0, Y(a), x0 + 480, Y(a), col=GRID, w=1, dash=("1,0" if a == 0 else "2,3"))
        s += text(x0 - 8, Y(a), str(a), size=9, col=MUTE, anchor="end", family=MONO)
    for t in (2019, 2021, 2023, 2025):
        s += text(X(t), y0 + 15, str(t), size=9, col=MUTE, family=MONO)
    s += text(X(2026) + 4, Y(100) - 6, "ceiling", size=9, col=MUTE, anchor="end", style="italic")
    # classic (saturating) curve
    def classic(t): return 15 + 77/(1 + math.exp(-1.6*(t - 2022)))
    pts = []
    t = 2019.0
    while t <= 2026.001:
        pts.append(f"{X(t):.2f},{Y(classic(t)):.2f}"); t += 0.1
    s += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{CHARCOAL}" stroke-width="2.4"/>\n'
    s += text(X(2024.2), Y(classic(2024.2)) - 12, "classic (MMLU, GSM8K)", size=10, col=CHARCOAL, anchor="middle")
    s += text(X(2025.4), Y(92) + 14, "saturated", size=9, col=MUTE, style="italic")
    # frontier (still climbing) curve, introduced 2024
    def frontier(t): return 8 + 50/(1 + math.exp(-1.4*(t - 2025.2)))
    pts = []
    t = 2024.0
    while t <= 2026.001:
        pts.append(f"{X(t):.2f},{Y(frontier(t)):.2f}"); t += 0.1
    s += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{RED}" stroke-width="2.6"/>\n'
    s += circle(X(2024), Y(frontier(2024)), 3.4, fill=RED)
    s += text(X(2024) + 6, Y(frontier(2024)) + 4, "GPQA, SWE-bench, ARC-AGI introduced", size=9.5, col=RED, anchor="start")
    s += text(X(2026), Y(frontier(2026)) - 10, "still discriminating", size=9.5, col=RED, anchor="end")
    s += footer()
    save(OUT, "fig-benchmark-saturation.svg", s)


ALL = [fig_saturation]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
