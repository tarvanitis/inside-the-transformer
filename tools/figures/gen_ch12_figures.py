#!/usr/bin/env python3
"""Generate the Chapter 12 (training pipeline) figures as static SVGs.

Chapter 12 already has Figure 12.1 (mermaid pipeline); this is 12.2.
Shared helpers live in svgkit.py.

Usage:
    python3 tools/figures/gen_ch12_figures.py [--png]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch12"


# ============================================================================
# Figure 12.2 — Where the compute goes
# ============================================================================
def fig_compute_budget():
    W, H = 640, 300
    s = header(W, H)
    s += text(W/2, 30, "Where the compute goes: pre-training dominates the budget.", size=12, col=CHARCOAL)
    bx, bw, bh = 40, 560, 42
    # top bar: overall
    by = 78
    pt = 0.92
    s += text(bx, by - 8, "total GPU-hours", size=10, col=MUTE, anchor="start", style="italic")
    s += rect(bx, by, bw*pt, bh, fill=REDFILL, stroke=RED, sw=1.2)
    s += text(bx + bw*pt/2, by + bh/2, "pre-training   ≈ 90–97%", size=12.5, col=RED, weight="bold")
    s += rect(bx + bw*pt, by, bw*(1-pt), bh, fill=GRAYFILL, stroke=CHARCOAL, sw=1)
    s += text(bx + bw*pt + bw*(1-pt)/2, by - 8, "post-training", size=9.5, col=MUTE)
    # zoom lines to the expanded bar
    by2 = 178
    s += line(bx + bw*pt, by + bh, bx, by2 - 6, col="#c9b8bf", w=1, dash="3,3")
    s += line(bx + bw, by + bh, bx + bw, by2 - 6, col="#c9b8bf", w=1, dash="3,3")
    s += text(W/2, by + bh + 34, "the post-training slice, expanded:", size=10.5, col=MUTE, style="italic")
    # bottom bar: post-training breakdown
    segs = [("SFT + reward model", "days", 0.20, GRAYFILL, CHARCOAL),
            ("RLHF / DPO", "weeks", 0.32, "#efc9d4", RED),
            ("reasoning RL", "weeks–months", 0.48, REDFILL, RED)]
    x = bx
    for lab, dur, frac, fill, st in segs:
        w = bw*frac
        s += rect(x, by2, w, bh, fill=fill, stroke=st, sw=1)
        s += text(x + w/2, by2 + bh + 14, lab, size=9.5, col=INK)
        s += text(x + w/2, by2 + bh + 28, dur, size=8.5, col=MUTE, style="italic")
        x += w
    s += text(W/2, H - 12, "You cannot fine-tune a weak base into a frontier model.",
              size=10.5, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-compute-budget.svg", s)


def fig_pipeline():
    W, H = 730, 230
    s = header(W, H)
    s += text(W/2, 26, "The seven training stages, from random weights to a deployed reasoning model.",
              size=11.5, col=CHARCOAL)
    ns = {"pt":  (62, 110, 96, ["① Pre-training"], REDFILL, RED),
          "sft": (188, 110, 78, ["② SFT"], PAPER, CHARCOAL),
          "rm":  (320, 64, 124, ["③ Reward model"], PAPER, CHARCOAL),
          "dpo": (320, 156, 100, ["⑤ DPO"], PAPER, CHARCOAL),
          "rlhf":(476, 64, 124, ["④ RLHF / PPO"], PAPER, CHARCOAL),
          "rl":  (612, 110, 116, ["⑥ Reasoning RL"], REDFILL, RED),
          "eval":(612, 188, 116, ["⑦ Evaluation"], CREAM, RED)}
    for cx, cy, w, lines, fill, st in ns.values():
        s += node(cx, cy, w, 30, lines, fill=fill, stroke=st, size=10)
    def E(a, b, col=GRAY, mk="gray"):
        (ax, ay, aw) = ns[a][0], ns[a][1], ns[a][2]
        (bx, by, bw) = ns[b][0], ns[b][1], ns[b][2]
        return arrow(ax + aw/2 + 2, ay + (6 if by > ay else (-6 if by < ay else 0)),
                     bx - bw/2 - 2, by, col=col, marker=mk, w=1.5)
    s += E("pt", "sft")
    s += E("sft", "rm")
    s += E("sft", "dpo")
    s += E("rm", "rlhf")
    s += E("rlhf", "rl")
    s += E("dpo", "rl")
    # rl -> eval (down)
    s += arrow(612, 125, 612, 173, col=GRAY, marker="gray", w=1.5)
    s += footer()
    save(OUT, "fig-pipeline.svg", s)


ALL = [fig_pipeline, fig_compute_budget]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
