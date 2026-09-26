#!/usr/bin/env python3
"""Generate the Chapter 16 (reasoning RL / GRPO) figures as static SVGs.

Two figures: GRPO drops PPO's critic, and the group-relative advantage number
line. Shared helpers in svgkit.py.

Usage:
    python3 tools/figures/gen_ch16_figures.py [--png]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch16"


def _box(cx, y, label, w=132, h=30, fill=PAPER, stroke=CHARCOAL, tc=INK, size=10.5):
    return (rect(cx - w/2, y, w, h, fill=fill, stroke=stroke, sw=1.1, rx=5)
            + text(cx, y + h/2, label, size=size, col=tc))


# ============================================================================
# Figure 16.1 — GRPO drops the critic
# ============================================================================
def fig_grpo_baseline():
    W, H = 640, 280
    s = header(W, H)
    s += text(W/2, 28, "GRPO drops PPO's critic and uses the group's own average as the baseline.",
              size=11.5, col=CHARCOAL)
    # ---- PPO (left) ----
    s += panel(16, 50, 300, 214, "PPO: a trained critic")
    cx = 100
    s += _box(cx, 78, "policy", w=110)
    s += arrow(cx, 108, cx, 126, col=GRAY, marker="gray", w=1.4)
    s += _box(cx, 126, "response → reward r", w=150)
    # critic
    ccx = 240
    s += _box(ccx, 78, "value network", w=110, fill=REDFILL, stroke=RED, tc=RED)
    s += text(ccx, 68, "extra trained model", size=8.5, col=RED, style="italic")
    s += arrow(ccx, 108, ccx, 126, col=RED, marker="red", w=1.4)
    s += _box(ccx, 126, "baseline b", w=110, fill=CREAM, stroke=RED, tc=RED)
    s += _box(166, 196, "advantage = r − b", w=180, fill="#ffffff", stroke=CHARCOAL)
    s += arrow(cx, 156, 150, 194, col=GRAY, marker="gray", w=1.2)
    s += arrow(ccx, 156, 210, 194, col=RED, marker="red", w=1.2)
    # ---- GRPO (right) ----
    s += panel(324, 50, 300, 214, "GRPO: the group is the baseline")
    gx = 474
    s += _box(gx, 78, "policy", w=110)
    s += arrow(gx, 108, gx, 122, col=GRAY, marker="gray", w=1.4)
    # group of responses
    rewards = [1, 1, 0, 0]
    gw, gsx = 30, 8
    gx0 = gx - (4*gw + 3*gsx)/2
    for i, r in enumerate(rewards):
        x = gx0 + i*(gw + gsx)
        red = r == 1
        s += rect(x, 128, gw, 26, fill=(REDFILL if red else GRAYFILL),
                  stroke=(RED if red else "#c9c4b8"), sw=1)
        s += text(x + gw/2, 128 + 13, str(r), size=9.5, col=(RED if red else MUTE), family=MONO)
    s += text(gx, 170, "G responses, rewards {1,1,0,0}", size=9, col=MUTE, style="italic")
    s += _box(gx, 186, "baseline = group mean", w=180, fill=CREAM, stroke=RED, tc=RED)
    s += text(gx, 226, "no critic — no extra network to train", size=9.5, col=RED, style="italic")
    s += footer()
    save(OUT, "fig-grpo-baseline.svg", s)


# ============================================================================
# Figure 16.2 — Group-relative advantage number line
# ============================================================================
def fig_advantage():
    W, H = 600, 250
    s = header(W, H)
    s += text(W/2, 30, "Group-relative advantage on “17 × 24” (a group of four).",
              size=11.5, col=CHARCOAL)
    x0, x1, ry = 110, 490, 118
    def RX(r): return x0 + r*(x1 - x0)
    s += line(x0 - 12, ry, x1 + 12, ry, col=GRAY, w=1.4)
    s += text(x0, ry + 26, "reward 0 (wrong)", size=9, col=MUTE, family=MONO)
    s += text(x1, ry + 26, "reward 1 (correct)", size=9, col=MUTE, family=MONO, anchor="end")
    # mean line
    s += line(RX(0.5), ry - 46, RX(0.5), ry + 14, col=CHARCOAL, w=1.2, dash="4,3")
    s += text(RX(0.5), ry - 54, "group mean 0.5", size=10, col=CHARCOAL)
    # responses
    dots = [("y₃", 0, False, -16), ("y₄", 0, False, 16),
            ("y₁", 1, True, -16), ("y₂", 1, True, 16)]
    for lab, r, red, off in dots:
        s += circle(RX(r) + off, ry, 6, fill=(REDFILL if red else GRAYFILL),
                    stroke=(RED if red else CHARCOAL), sw=1.3)
        s += text(RX(r) + off, ry - 14, lab, size=9.5, col=(RED if red else CHARCOAL), family=MONO)
    # advantage annotations
    s += arrow(RX(0.5) - 8, ry + 62, RX(0.06), ry + 62, col=CHARCOAL, marker="char", w=1.3)
    s += text(RX(0.0) + 6, ry + 55, "advantage −1  (suppress)", size=9.5, col=CHARCOAL, anchor="start")
    s += arrow(RX(0.5) + 8, ry + 62, RX(0.94), ry + 62, col=RED, marker="red", w=1.3)
    s += text(RX(1.0) - 6, ry + 55, "advantage +1  (reinforce)", size=9.5, col=RED, anchor="end")
    s += text(W/2, H - 12,
              "Standardizing by the group mean and std turns a correct answer into +1σ, a wrong one into −1σ.",
              size=10, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-group-advantage.svg", s)


ALL = [fig_grpo_baseline, fig_advantage]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
