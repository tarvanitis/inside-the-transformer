#!/usr/bin/env python3
"""Generate the Chapter 8 (inference) figures as static SVGs.

Chapter 8 already has Figure 8.1 (the mermaid generation loop); these are 8.2-8.3.
Shared helpers live in svgkit.py.

Usage:
    python3 tools/figures/gen_ch08_figures.py [--png]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch08"


# ============================================================================
# Figure 8.2 — Why the KV-cache helps (recompute vs cache)
# ============================================================================
def fig_kv_cache():
    W, H = 640, 300
    s = header(W, H)
    s += text(W/2, 30, "The KV-cache stores past keys and values, so each step computes only the new token.",
              size=11.5, col=CHARCOAL)
    cell, gy = 30, 96

    def grid(x0, title, cached, note):
        out = text(x0 + 2*cell, gy - 16, title, size=12, col=CHARCOAL, weight="bold")
        for step in range(4):
            for pos in range(4):
                xx, yy = x0 + pos*cell, gy + step*cell
                if pos > step:
                    out += rect(xx, yy, cell, cell, fill="#ffffff", stroke="#ece9e1", sw=0.8)
                elif cached and pos < step:
                    out += rect(xx, yy, cell, cell, fill=GRAYFILL, stroke="#ffffff", sw=1)
                else:
                    out += rect(xx, yy, cell, cell, fill=REDFILL, stroke=RED, sw=1)
        for pos in range(4):
            out += text(x0 + pos*cell + cell/2, gy - 3, f"t{pos+1}", size=8, col=MUTE, family=MONO)
        for step in range(4):
            out += text(x0 - 6, gy + step*cell + cell/2, f"step {step+1}", size=8, col=MUTE,
                        anchor="end", family=MONO)
        out += text(x0 + 2*cell, gy + 4*cell + 16, note, size=9.5, col=MUTE, style="italic")
        return out

    s += grid(150, "without cache", False, "recompute every token each step")
    s += grid(440, "with KV-cache", True, "compute only the new token")
    # legend
    ly = gy + 4*cell + 40
    s += rect(150, ly, 15, 15, fill=REDFILL, stroke=RED, sw=1)
    s += text(170, ly + 8, "computed this step", size=10, col=INK, anchor="start")
    s += rect(340, ly, 15, 15, fill=GRAYFILL, stroke="#c9c4b8", sw=1)
    s += text(360, ly + 8, "read from cache", size=10, col=INK, anchor="start")
    s += footer()
    save(OUT, "fig-kv-cache.svg", s)


# ============================================================================
# Figure 8.3 — FlashAttention: full matrix vs streamed tiles
# ============================================================================
def fig_flashattention():
    W, H = 640, 280
    s = header(W, H)
    s += text(W/2, 30, "FlashAttention streams the score matrix through fast on-chip memory in tiles,",
              size=11.5, col=CHARCOAL)
    s += text(W/2, 46, "instead of building all of it at once in slow memory.", size=11.5, col=CHARCOAL)
    # left: standard
    s += panel(20, 66, 288, 190, "Standard attention")
    bx, by, bs = 95, 108, 130
    s += rect(bx, by, bs, bs, fill=REDFILL, stroke=RED, sw=1.3)
    s += text(bx + bs/2, by + bs/2 - 8, "full T × T scores", size=11, col=INK)
    s += text(bx + bs/2, by + bs/2 + 10, "in slow memory", size=10, col=MUTE)
    s += text(bx + bs/2, by + bs + 18, "gigabytes moved to and from DRAM", size=9.5, col=RED)
    # right: flash
    s += panel(332, 66, 288, 190, "FlashAttention")
    tx, ty, ts = 380, 100, 44
    for r in range(3):
        for c in range(3):
            hl = (r == 1 and c == 1)
            s += rect(tx + c*ts, ty + r*ts, ts, ts,
                      fill=(REDFILL if hl else "#f4f1ea"),
                      stroke=(RED if hl else "#d8d2c6"), sw=(1.5 if hl else 0.8))
    s += arrow(tx + 3*ts + 6, ty + 1.5*ts, tx + 3*ts + 44, ty + 1.5*ts, col=RED, marker="red", w=1.6)
    s += text(tx + 3*ts + 52, ty + 1.5*ts - 8, "online", size=10, col=RED, anchor="start")
    s += text(tx + 3*ts + 52, ty + 1.5*ts + 8, "softmax", size=10, col=RED, anchor="start")
    s += text(tx + 1.5*ts, ty + 3*ts + 20, "one tile at a time in fast SRAM", size=9.5, col=MUTE)
    s += footer()
    save(OUT, "fig-flashattention.svg", s)


def fig_generation_loop():
    W, H = 720, 210
    s = header(W, H)
    s += text(W/2, 26, "The generation loop: forward pass, sample, append, repeat.", size=11.5, col=CHARCOAL)
    cy = 74

    def N(cx, w, lines, fill=PAPER, st=CHARCOAL):
        h = 18 + len(lines)*15
        return node(cx, cy, w, h, lines, fill=fill, stroke=st, size=9.5)
    xs = {"prompt": (52, 56), "tok": (140, 78), "fwd": (244, 92), "log": (340, 66),
          "samp": (424, 66), "emit": (500, 58)}
    s += N(*xs["prompt"], ["prompt"], fill=CREAM, st=RED)
    s += N(*xs["tok"], ["tokenize"])
    s += N(*xs["fwd"], ["forward pass"], fill=REDFILL, st=RED)
    s += N(*xs["log"], ["logits"])
    s += N(*xs["samp"], ["sample"])
    s += N(*xs["emit"], ["emit"])
    s += diamond(586, cy, 84, 48, "", fill=CREAM, stroke=RED)
    s += text(586, cy - 4, "EOS or", size=8.5, col=INK)
    s += text(586, cy + 8, "max?", size=8.5, col=INK)
    s += N(672, 64, ["final", "text"], fill=CREAM, st=RED)
    order = ["prompt", "tok", "fwd", "log", "samp", "emit"]
    for a, b in zip(order, order[1:]):
        s += arrow(xs[a][0] + xs[a][1]/2 + 2, cy, xs[b][0] - xs[b][1]/2 - 2, cy, col=GRAY, marker="gray", w=1.5)
    s += arrow(529, cy, 544, cy, col=GRAY, marker="gray", w=1.5)   # emit -> diamond
    s += arrow(628, cy, 640, cy, col=RED, marker="red", w=1.5)     # yes -> final
    s += text(634, cy - 8, "yes", size=8, col=RED)
    # no -> append -> loop back to forward
    s += arrow(586, cy + 24, 586, 150, col=CHARCOAL, marker="char", w=1.4)
    s += text(596, 130, "no", size=8, col=CHARCOAL, anchor="start")
    s += node(586, 166, 110, 26, "append token", fill=PAPER, stroke=CHARCOAL, size=9.5)
    s += line(531, 166, 244, 166, col=CHARCOAL, w=1.4)
    s += arrow(244, 166, 244, cy + 24, col=CHARCOAL, marker="char", w=1.4)
    s += footer()
    save(OUT, "fig-generation-loop.svg", s)


ALL = [fig_generation_loop, fig_kv_cache, fig_flashattention]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
