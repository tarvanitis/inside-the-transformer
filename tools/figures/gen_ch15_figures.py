#!/usr/bin/env python3
"""Generate the Chapter 15 (preferences) figures as static SVGs.

The existing RLHF-loop mermaid is renumbered to 15.2; these are 15.1
(Bradley-Terry) and 15.3 (the KL leash). Shared helpers in svgkit.py.

Usage:
    python3 tools/figures/gen_ch15_figures.py [--png]
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch15"


# ============================================================================
# Figure 15.1 — Bradley-Terry: score gap -> preference probability
# ============================================================================
def fig_bradley_terry():
    W, H = 580, 300
    s = header(W, H)
    s += text(W/2, 30, "Bradley-Terry: the score gap sets the probability of preferring the winner.",
              size=11.5, col=CHARCOAL)
    f = Frame(ox=300, oy=250, sx=52, sy=188)   # x in [-4,4], y in [0,1]
    # gridlines
    for yv in (0.5, 1.0):
        s += line(f.X(-4), f.Y(yv), f.X(4), f.Y(yv), col=GRID, w=1, dash="2,3")
        s += text(f.X(-4) - 8, f.Y(yv), f"{yv:.1f}", size=9, col=MUTE, anchor="end", family=MONO)
    # axes
    s += arrow(f.X(-4.3), f.Y(0), f.X(4.3), f.Y(0), col=GRAY, marker="gray", w=1.4)
    s += arrow(f.X(0), f.Y(0), f.X(0), f.Y(1.12), col=GRAY, marker="gray", w=1.3)
    s += text(f.X(0) + 96, f.Y(0) + 30, "score gap  (r_w − r_l)  →", size=10.5, col=MUTE)
    s += ('<text x="%.1f" y="%.1f" font-size="10.5" fill="%s" text-anchor="middle" font-family="%s" '
          'transform="rotate(-90 %.1f %.1f)">P(prefer winner)</text>\n'
          % (f.X(-4) - 34, f.Y(0.5), MUTE, SERIF, f.X(-4) - 34, f.Y(0.5)))
    # sigmoid curve
    pts = []
    x = -4.0
    while x <= 4.0001:
        pts.append(f"{f.X(x):.2f},{f.Y(1/(1+math.exp(-x))):.2f}"); x += 0.1
    s += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{RED}" stroke-width="2.6"/>\n'
    # coin-flip marker at gap 0
    s += circle(f.X(0), f.Y(0.5), 3.4, fill=GRAY)
    s += text(f.X(0) + 6, f.Y(0.5) + 16, "gap 0 → coin flip", size=9.5, col=MUTE, anchor="start")
    # worked-example points
    s += circle(f.X(-0.4), f.Y(0.401), 4.4, fill="none", stroke=RED, sw=1.8)
    s += text(f.X(-0.4) - 6, f.Y(0.401) - 10, "before: gap −0.4, 0.40", size=9, col=RED, anchor="end")
    s += circle(f.X(1.4), f.Y(0.802), 4.4, fill=RED)
    s += text(f.X(1.4) + 8, f.Y(0.802) - 6, "after: gap +1.4, 0.80", size=9, col=RED, anchor="start")
    s += arrow(f.X(-0.4) + 4, f.Y(0.401) - 2, f.X(1.4) - 4, f.Y(0.802) + 2, col=RED, marker="red", w=1.2, dash="3,2")
    s += text(f.X(0.5), f.Y(0.62), "training", size=9, col=RED, style="italic")
    s += footer()
    save(OUT, "fig-bradley-terry.svg", s)


# ============================================================================
# Figure 15.3 — The KL leash
# ============================================================================
def fig_kl_leash():
    W, H = 620, 250
    s = header(W, H)
    s += text(W/2, 30, "The KL leash keeps the policy near the SFT model while it chases reward.",
              size=11.5, col=CHARCOAL)
    midY = 140
    # reward direction
    s += arrow(60, 74, 560, 74, col="#d8d2c6", marker="gray", w=1.3)
    s += text(310, 64, "higher reward →", size=10, col=MUTE, style="italic")
    # anchor
    rx = 150
    s += circle(rx, midY, 7, fill=CHARCOAL)
    s += text(rx, midY + 22, "π_ref", size=12, col=CHARCOAL, family=MONO)
    s += text(rx, midY + 37, "frozen SFT anchor", size=9, col=MUTE, style="italic")
    # policy
    px = 320
    s += circle(px, midY, 7, fill=RED)
    s += text(px, midY + 22, "π_θ", size=12, col=RED, family=MONO)
    s += text(px, midY + 37, "policy", size=9, col=RED, style="italic")
    # leash
    s += line(rx + 7, midY, px - 7, midY, col=RED, w=1.7, dash="4,3")
    s += text((rx + px)/2, midY - 12, "KL leash (β)", size=10, col=RED)
    # pull toward reward
    s += arrow(px + 8, midY, px + 74, midY, col=RED, marker="red", w=1.5)
    s += text(px + 44, midY - 10, "pull", size=9, col=RED, style="italic")
    # reward-hacking zone (unreachable)
    hz = 500
    s += rect(hz - 40, midY - 26, 130, 52, fill="#f5e9ec", stroke="#e2bcc6", sw=1, rx=6)
    s += text(hz + 25, midY - 6, "reward hacking", size=10, col="#a85a70")
    s += text(hz + 25, midY + 10, "(sycophancy, padding)", size=8.5, col=MUTE)
    s += text((px + hz - 40)/2 + 6, midY + 30, "the leash holds it back", size=9, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-kl-leash.svg", s)


def fig_rlhf_loop():
    W, H = 700, 210
    s = header(W, H)
    s += text(W/2, 24, "The RLHF loop: policy, reward model, and the KL leash to the frozen reference.",
              size=11, col=CHARCOAL)
    cy = 74
    ns = {"prompt": (56, 60, ["prompt x"], CREAM, RED),
          "policy": (176, 108, ["policy π_θ"], REDFILL, RED),
          "resp":   (312, 104, ["response y"], PAPER, CHARCOAL),
          "rm":     (452, 118, ["reward model", "→ r(x, y)"], PAPER, CHARCOAL),
          "ppo":    (606, 96, ["PPO update"], REDFILL, RED)}
    for cx, w, lines, fill, st in ns.values():
        h = 18 + len(lines)*15
        s += node(cx, cy, w, h, lines, fill=fill, stroke=st, size=10)
    order = ["prompt", "policy", "resp", "rm", "ppo"]
    for a, b in zip(order, order[1:]):
        s += arrow(ns[a][0] + ns[a][1]/2 + 2, cy, ns[b][0] - ns[b][1]/2 - 2, cy, col=GRAY, marker="gray", w=1.5)
    # loop back PPO -> policy
    s += arrow(606, cy + 20, 606, 132, col=CHARCOAL, marker="char", w=1.4)
    s += line(606, 132, 176, 132, col=CHARCOAL, w=1.4)
    s += arrow(176, 132, 176, cy + 20, col=CHARCOAL, marker="char", w=1.4)
    s += text(390, 125, "update the policy toward higher reward", size=8.5, col=MUTE, style="italic")
    # frozen reference -> KL penalty -> PPO
    s += node(340, 175, 150, 26, "frozen reference π_ref", fill="#f0eee8", stroke=CHARCOAL, size=9.5)
    s += arrow(415, 168, 585, cy + 18, col=RED, marker="red", w=1.3, dash="4,3")
    s += text(520, 150, "KL penalty", size=9, col=RED, style="italic")
    s += footer()
    save(OUT, "fig-rlhf-loop.svg", s)


ALL = [fig_bradley_terry, fig_rlhf_loop, fig_kl_leash]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
