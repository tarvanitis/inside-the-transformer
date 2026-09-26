#!/usr/bin/env python3
"""Generate the Chapter 13 (pre-training) figures as static SVGs.

Two figures: the learning-rate schedule (warm-up then cosine decay), and
Chinchilla scaling (tokens grow with parameters, ~20:1). Shared helpers in svgkit.py.

Usage:
    python3 tools/figures/gen_ch13_figures.py [--png]
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch13"


# ============================================================================
# Figure 13.1 — The learning-rate schedule
# ============================================================================
def fig_lr_schedule():
    W, H = 600, 300
    s = header(W, H)
    s += text(W/2, 30, "The learning-rate schedule: a short warm-up, then a long cosine decay.",
              size=11.5, col=CHARCOAL)
    peak, warm, total, minfrac = 3.0, 2.0, 100.0, 0.1   # in units: LR e-4, steps in thousands
    f = Frame(ox=70, oy=250, sx=(470/total), sy=(180/peak))

    def lr(step):
        mn = peak*minfrac
        if step < warm:
            return peak*step/warm
        decay = (step - warm)/(total - warm)
        return mn + (peak - mn)*0.5*(1 + math.cos(math.pi*decay))
    # axes
    s += arrow(f.X(0), f.Y(0), f.X(total*1.02), f.Y(0), col=GRAY, marker="gray", w=1.4)
    s += arrow(f.X(0), f.Y(0), f.X(0), f.Y(peak*1.12), col=GRAY, marker="gray", w=1.4)
    s += text(f.X(total/2), f.Y(0) + 30, "training step →", size=11, col=MUTE)
    s += ('<text x="%.1f" y="%.1f" font-size="11" fill="%s" text-anchor="middle" font-family="%s" '
          'transform="rotate(-90 %.1f %.1f)">learning rate</text>\n'
          % (f.X(0) - 40, f.Y(peak/2), MUTE, SERIF, f.X(0) - 40, f.Y(peak/2)))
    s += line(f.X(0), f.Y(peak), f.X(total), f.Y(peak), col=GRID, w=1, dash="2,3")
    s += text(f.X(0) - 8, f.Y(peak), "peak", size=9, col=MUTE, anchor="end", family=MONO)
    s += line(f.X(0), f.Y(peak*minfrac), f.X(total), f.Y(peak*minfrac), col=GRID, w=1, dash="2,3")
    s += text(f.X(0) - 8, f.Y(peak*minfrac), "10%", size=9, col=MUTE, anchor="end", family=MONO)
    # curve
    pts = []
    x = 0.0
    while x <= total + 0.001:
        pts.append(f"{f.X(x):.2f},{f.Y(lr(x)):.2f}"); x += 0.5
    s += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{RED}" stroke-width="2.6"/>\n'
    # warm-up marker
    s += line(f.X(warm), f.Y(0), f.X(warm), f.Y(peak), col=CHARCOAL, w=1, dash="3,3")
    s += circle(f.X(warm), f.Y(peak), 3.6, fill=RED)
    s += text(f.X(warm) + 4, f.Y(peak) - 10, "end of warm-up", size=9.5, col=CHARCOAL, anchor="start")
    s += text(f.X(warm/2), f.Y(peak) + 22, "warm-up", size=9.5, col=MUTE, style="italic")
    s += text(f.X(total*0.55), f.Y(lr(total*0.55)) - 14, "cosine decay", size=10, col=RED)
    s += footer()
    save(OUT, "fig-lr-schedule.svg", s)


# ============================================================================
# Figure 13.2 — Chinchilla scaling (log-log)
# ============================================================================
def fig_scaling():
    W, H = 600, 320
    s = header(W, H)
    s += text(W/2, 30, "Chinchilla scaling: train tokens should grow with model size (about 20 : 1).",
              size=11.5, col=CHARCOAL)
    # log-log frame: x = log10(params in B) 0..3 ; y = log10(tokens in T) -1..1.4
    x0, y0 = 80, 262
    xlo, xhi = 0.0, 3.0
    ylo, yhi = -1.2, 1.4
    px = (470)/(xhi - xlo)
    py = (196)/(yhi - ylo)
    def X(pB): return x0 + (math.log10(pB) - xlo)*px
    def Y(tT): return y0 - (math.log10(tT) - ylo)*py
    # axes
    s += arrow(x0, y0, x0 + 480, y0, col=GRAY, marker="gray", w=1.4)
    s += arrow(x0, y0, x0, y0 - 210, col=GRAY, marker="gray", w=1.4)
    s += text(x0 + 240, y0 + 32, "model parameters →", size=11, col=MUTE)
    s += ('<text x="%.1f" y="%.1f" font-size="11" fill="%s" text-anchor="middle" font-family="%s" '
          'transform="rotate(-90 %.1f %.1f)">training tokens</text>\n'
          % (x0 - 46, y0 - 100, MUTE, SERIF, x0 - 46, y0 - 100))
    for pB, lab in ((1, "1B"), (10, "10B"), (100, "100B"), (1000, "1T")):
        s += text(X(pB), y0 + 16, lab, size=9, col=MUTE, family=MONO)
        s += line(X(pB), y0, X(pB), y0 - 4, col=GRAY, w=1)
    for tT, lab in ((0.1, "0.1T"), (1, "1T"), (10, "10T")):
        s += text(x0 - 8, Y(tT), lab, size=9, col=MUTE, anchor="end", family=MONO)
        s += line(x0, Y(tT), x0 + 480, Y(tT), col=GRID, w=1, dash="2,3")
    # 20:1 line: tokens_T = 0.02 * params_B
    s += line(X(1), Y(0.02*1), X(1000), Y(0.02*1000), col=CHARCOAL, w=1.6, dash="6,3")
    s += text(X(300), Y(0.02*300) + 18, "compute-optimal (≈20:1)", size=10, col=CHARCOAL)
    # optimal points
    for pB, tT, lab in ((7, 0.14, "7B"), (13, 0.26, ""), (70, 1.4, "70B"), (405, 8.1, "405B")):
        s += circle(X(pB), Y(tT), 4, fill=CHARCOAL)
        if lab:
            s += text(X(pB), Y(tT) - 12, lab, size=8.5, col=CHARCOAL, family=MONO)
    # overtrained point: Llama 3 8B on 15T
    s += arrow(X(8), Y(0.16) - 4, X(8), Y(15) + 6, col=RED, marker="red", w=1.5, dash="3,2")
    s += circle(X(8), Y(15), 5, fill=RED)
    s += text(X(8) + 8, Y(15), "Llama 3 8B: 15T", size=9.5, col=RED, anchor="start", family=MONO)
    s += text(X(8) + 8, Y(15) + 14, "≈100× optimal (cheaper to serve)", size=9, col=MUTE, anchor="start", style="italic")
    s += footer()
    save(OUT, "fig-chinchilla-scaling.svg", s)


ALL = [fig_lr_schedule, fig_scaling]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
