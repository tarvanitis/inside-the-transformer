#!/usr/bin/env python3
"""Generate the Chapter 19 (agents) figures as static SVGs.

Chapter 19 already has Figure 19.1 (mermaid agent loop); these are 19.2-19.3.
Shared helpers in svgkit.py.

Usage:
    python3 tools/figures/gen_ch19_figures.py [--png]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch19"


def _box(cx, y, label, w=110, h=30, fill=PAPER, stroke=CHARCOAL, tc=INK, size=10.5):
    return (rect(cx - w/2, y, w, h, fill=fill, stroke=stroke, sw=1.1, rx=5)
            + text(cx, y + h/2, label, size=size, col=tc))


# ============================================================================
# Figure 19.2 — Two agent architectures
# ============================================================================
def fig_agent_architectures():
    W, H = 640, 300
    s = header(W, H)
    s += text(W/2, 28, "Two agent architectures: start simple, add structure only when needed.",
              size=11.5, col=CHARCOAL)
    # ---- single agent ----
    s += panel(16, 52, 300, 226, "Single agent")
    ax, ay = 166, 108
    s += _box(ax, ay, "LLM agent", w=130, fill=CREAM, stroke=RED)
    # loop arrow on the agent
    s += arc(ax + 78, ay + 15, 16, -60, 200, col=MUTE, w=1.3)
    s += text(ax + 104, ay + 4, "loop", size=8.5, col=MUTE, style="italic", anchor="start")
    tools = ["search", "run code", "files"]
    for i, t in enumerate(tools):
        tx = 62 + i*94
        s += _box(tx, 214, t, w=80, h=26, size=9.5)
        s += arrow(ax + (i-1)*40, ay + 30, tx, 214 - 2, col=GRAY, marker="gray", w=1.1)
    s += text(166, 254, "one loop, one tool set", size=9.5, col=MUTE, style="italic")
    # ---- orchestrator ----
    s += panel(324, 52, 300, 226, "Orchestrator + sub-agents")
    ox = 474
    s += _box(ox, 80, "orchestrator", w=140, fill=CREAM, stroke=RED)
    workers = ["research", "analyst", "writer"]
    for i, w in enumerate(workers):
        wx = 384 + i*94
        s += _box(wx, 176, w, w=84, h=28, size=9.5)
        s += arrow(ox + (i-1)*52, 110, wx, 176 - 2, col=GRAY, marker="gray", w=1.1)
    s += text(474, 224, "workers specialize, results return to the supervisor",
              size=9, col=MUTE, style="italic")
    s += text(474, 254, "use only when one agent hits its limits", size=9.5, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-agent-architectures.svg", s)


# ============================================================================
# Figure 19.3 — Why agent runs cost more (quadratic context)
# ============================================================================
def fig_agent_cost():
    W, H = 560, 300
    s = header(W, H)
    s += text(W/2, 28, "Each step resends the whole conversation, so cost grows with the square of the steps.",
              size=11, col=CHARCOAL)
    x0, y0 = 66, 246
    bw, gap = 32, 8
    sc = 170/20.0   # 0..20k tokens -> px
    for step in range(1, 11):
        h = (2*step)*sc
        x = x0 + (step-1)*(bw + gap)
        s += rect(x, y0 - h, bw, h, fill=REDFILL, stroke=RED, sw=1)
        s += text(x + bw/2, y0 + 12, str(step), size=8, col=MUTE, family=MONO)
    s += line(x0 - 8, y0, x0 + 10*(bw + gap), y0, col=GRAY, w=1.3)
    s += text(x0 + 5*(bw + gap), y0 + 28, "agent step", size=10, col=MUTE)
    s += text(x0 + 9*(bw+gap) + bw/2, y0 - (20*sc) - 8, "20k", size=8, col=RED, family=MONO)
    # totals box (upper-left, above the short bars)
    bxx = 70
    s += rect(bxx, 60, 224, 74, fill=PAPER, stroke=BORDER, sw=1, rx=6)
    s += text(bxx + 112, 80, "10-step run", size=10.5, col=CHARCOAL, weight="bold")
    s += text(bxx + 112, 98, "total ≈ 110k tokens", size=10.5, col=RED, family=MONO)
    s += text(bxx + 112, 114, "naive 10 × 2k = 20k", size=10, col=MUTE, family=MONO)
    s += text(bxx + 112, 128, "→ 5.5× more", size=10.5, col=RED, family=MONO, weight="bold")
    s += footer()
    save(OUT, "fig-agent-cost.svg", s)


def fig_agent_loop():
    W, H = 700, 210
    s = header(W, H)
    s += text(W/2, 26, "The agent loop: reason, act, observe, repeat until done.", size=11.5, col=CHARCOAL)
    cy = 78
    s += node(58, cy, 70, 30, "user goal", fill=CREAM, stroke=RED, size=10)
    s += node(190, cy, 110, 30, "LLM reasons / plans", fill=REDFILL, stroke=RED, size=10)
    s += diamond(330, cy, 96, 50, "", fill=CREAM, stroke=RED)
    s += text(330, cy - 4, "need a", size=9, col=INK)
    s += text(330, cy + 8, "tool?", size=9, col=INK)
    s += node(470, cy, 96, 30, "call tool", size=10)
    s += node(600, cy, 96, 30, "observe result", size=10)
    s += arrow(93, cy, 134, cy, col=GRAY, marker="gray", w=1.5)      # goal -> plan
    s += arrow(245, cy, 282, cy, col=GRAY, marker="gray", w=1.5)     # plan -> decision
    s += arrow(378, cy, 421, cy, col=RED, marker="red", w=1.5)       # yes -> tool
    s += text(400, cy - 8, "yes", size=8, col=RED)
    s += arrow(518, cy, 551, cy, col=GRAY, marker="gray", w=1.5)     # tool -> observe
    # observe loops back to plan
    s += arrow(600, cy + 15, 600, 150, col=CHARCOAL, marker="char", w=1.4)
    s += line(600, 150, 190, 150, col=CHARCOAL, w=1.4)
    s += arrow(190, 150, 190, cy + 15, col=CHARCOAL, marker="char", w=1.4)
    s += text(395, 143, "feed result back into the next step", size=8.5, col=MUTE, style="italic")
    # no -> final answer
    s += arrow(330, cy + 25, 330, 150, col=GRAY, marker="gray", w=1.4)
    s += text(340, 120, "no", size=8, col=MUTE, anchor="start")
    s += node(330, 168, 110, 28, "final answer", fill=CREAM, stroke=RED, size=10)
    s += footer()
    save(OUT, "fig-agent-loop.svg", s)


ALL = [fig_agent_loop, fig_agent_architectures, fig_agent_cost]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
