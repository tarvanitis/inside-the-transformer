#!/usr/bin/env python3
"""Generate the Chapter 5 (attention) figures as static SVGs.

Chapter 5 already has Figure 5.1 (the mermaid flowchart), so these are numbered
5.2 onward in order of appearance. Shared drawing helpers live in svgkit.py.

Usage:
    python3 tools/figures/gen_ch05_figures.py            # write SVGs
    python3 tools/figures/gen_ch05_figures.py --png      # also render PNG previews
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch05"


def _cellval(v):
    if abs(v) < 1e-9:
        return "0"
    sv = f"{v:.2f}"
    return sv[1:] if sv.startswith("0.") else sv


def heatgrid(x0, y0, M, cell, vmax, title=None, masked=None, rowlab="q", collab="k",
             numbers=True):
    """A labeled heatmap of matrix M (list of rows). masked = set of (r,c) drawn as -inf."""
    masked = masked or set()
    n = len(M); m = len(M[0])
    out = ""
    if title:
        out += text(x0 + m*cell/2, y0 - 20, title, size=12, col=CHARCOAL, weight="bold")
    for r in range(n):
        for c in range(m):
            xx, yy = x0 + c*cell, y0 + r*cell
            if (r, c) in masked:
                out += rect(xx, yy, cell, cell, fill="#e3e0d8", stroke="#ffffff", sw=1)
                if numbers:
                    out += text(xx + cell/2, yy + cell/2, "−∞", size=9, col=MUTE)
            else:
                v = M[r][c]
                frac = min(max(v / vmax, 0), 1)
                out += rect(xx, yy, cell, cell, fill=mix("#ffffff", RED, frac), stroke="#ffffff", sw=1)
                if numbers:
                    tc = "#ffffff" if frac > 0.62 else INK
                    out += text(xx + cell/2, yy + cell/2, _cellval(v), size=8.5, col=tc, family=MONO)
    if collab:
        for c in range(m):
            out += text(x0 + c*cell + cell/2, y0 - 6, f"{collab}{c}", size=8, col=MUTE, family=MONO)
    if rowlab:
        for r in range(n):
            out += text(x0 - 7, y0 + r*cell + cell/2, f"{rowlab}{r}", size=8, col=MUTE, family=MONO, anchor="end")
    return out


# ============================================================================
# Figure 5.2 — The five-step pipeline as grids (worked example)
# ============================================================================
def fig_attention_steps():
    W, H = 680, 250
    s = header(W, H)
    scaled = [[0.58, 0.58, 1.15, 0.00],
              [0.58, 0.58, 0.00, 0.58],
              [1.15, 0.58, 0.58, 0.58],
              [0.58, 1.15, 0.58, 0.58]]
    mask = {(r, c) for r in range(4) for c in range(4) if c > r}
    weights = [[1.00, 0, 0, 0],
               [0.50, 0.50, 0, 0],
               [0.47, 0.26, 0.26, 0],
               [0.21, 0.37, 0.21, 0.21]]
    cell, gy = 30, 62
    x1, x2, x3 = 60, 290, 520
    s += heatgrid(x1, gy, scaled, cell, 1.15, title="scaled scores")
    s += heatgrid(x2, gy, scaled, cell, 1.15, title="after causal mask", masked=mask)
    s += heatgrid(x3, gy, weights, cell, 1.0, title="attention weights")
    ymid = gy + 2*cell
    s += arrow(x1 + 4*cell + 6, ymid, x2 - 12, ymid, col=GRAY, marker="gray", w=1.4)
    s += arrow(x2 + 4*cell + 6, ymid, x3 - 12, ymid, col=GRAY, marker="gray", w=1.4)
    s += text((x1 + 4*cell + x2)/2 + 3, ymid - 8, "mask", size=9.5, col=MUTE, style="italic")
    s += text((x2 + 4*cell + x3)/2 + 3, ymid - 8, "softmax", size=9.5, col=MUTE, style="italic")
    s += text(W/2, H - 16,
              "Each query row (q0–q3) attends over the keys (k0–k3). The mask blocks the future; "
              "softmax makes each row sum to 1.",
              size=10, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-attention-steps.svg", s)


# ============================================================================
# Figure 5.3 — Multi-head attention: parallel patterns, concatenated
# ============================================================================
def fig_multihead():
    W, H = 680, 320
    s = header(W, H)
    heads = [
        ("head 1", "subject–verb", [[1, 0, 0, 0], [.1, .9, 0, 0], [.1, .1, .8, 0], [.05, .05, .1, .8]]),
        ("head 2", "similarity",       [[1, 0, 0, 0], [.5, .5, 0, 0], [.33, .33, .34, 0], [.25, .25, .25, .25]]),
        ("head 3", "nearby",           [[1, 0, 0, 0], [.2, .8, 0, 0], [0, .3, .7, 0], [0, 0, .3, .7]]),
        ("head 4", "brackets",         [[1, 0, 0, 0], [.8, .2, 0, 0], [.7, 0, .3, 0], [.6, 0, 0, .4]]),
    ]
    cell = 17
    xs = [46, 210, 374, 538]
    gy = 74
    for (hlabel, role, M), x in zip(heads, xs):
        s += heatgrid(x, gy, M, cell, 1.0, rowlab="", collab="", numbers=False)
        s += rect(x, gy, 4*cell, 4*cell, fill="none", stroke="#ccc7bc", sw=1)
        s += text(x + 2*cell, gy - 20, hlabel, size=11.5, col=CHARCOAL, weight="bold")
        s += text(x + 2*cell, gy + 4*cell + 14, role, size=10, col=MUTE, style="italic")
    # converge: line under the heads -> concat -> W_O -> output
    braceY = gy + 4*cell + 30
    s += line(xs[0] + 2*cell, braceY, xs[-1] + 2*cell, braceY, col=GRAY, w=1.2)
    for x in xs:
        s += line(x + 2*cell, braceY - 6, x + 2*cell, braceY, col=GRAY, w=1.2)
    cxc = W/2
    s += line(cxc, braceY, cxc, braceY + 14, col=GRAY, w=1.2)
    def flowbox(cy, label, fill):
        w, h = 150, 30
        return (rect(cxc - w/2, cy, w, h, fill=fill, stroke=CHARCOAL, sw=1.1, rx=5)
                + text(cxc, cy + h/2, label, size=12, col=INK))
    y0 = braceY + 14
    s += flowbox(y0, "concatenate heads", PAPER)
    s += arrow(cxc, y0 + 30, cxc, y0 + 44, col=GRAY, marker="gray", w=1.4)
    s += flowbox(y0 + 44, "mix with W_O", CREAM)
    s += arrow(cxc, y0 + 74, cxc, y0 + 88, col=RED, marker="red", w=1.6)
    s += text(cxc, y0 + 100, "context-aware output", size=12, col=RED, weight="bold")
    s += text(W/2, 26, "Every head sees the whole input and learns its own attention pattern.",
              size=12, col=CHARCOAL)
    s += footer()
    save(OUT, "fig-multihead.svg", s)


# ============================================================================
# Figure 5.4 — MHA vs GQA vs MQA: sharing key/value heads
# ============================================================================
def fig_kv_sharing():
    W, H = 680, 250
    s = header(W, H)
    panels = [("Multi-head", 8, "KV cache: 8 units"),
              ("Grouped-query", 2, "KV cache: 2 units"),
              ("Multi-query", 1, "KV cache: 1 unit")]
    pw = W / 3
    qn = 8
    qsz = 15
    for i, (title, kvn, note) in enumerate(panels):
        cx0 = i*pw
        span = qn*qsz + (qn-1)*4
        qx0 = cx0 + (pw - span)/2
        qy = 78
        # query heads
        for q in range(qn):
            qx = qx0 + q*(qsz+4)
            s += rect(qx, qy, qsz, qsz, fill=REDFILL, stroke=RED, sw=1)
        # kv heads
        kvy = 168
        kvw = qsz + 6
        group = qn // kvn
        kv_centers = []
        for k in range(kvn):
            # center the KV box under its group of query heads
            first = k*group
            gx0 = qx0 + first*(qsz+4)
            gx1 = qx0 + (first+group-1)*(qsz+4) + qsz
            cxk = (gx0 + gx1)/2
            kv_centers.append(cxk)
            s += rect(cxk - kvw/2, kvy, kvw, qsz, fill=CREAM, stroke=RED, sw=1.1)
            s += text(cxk, kvy + qsz/2, "KV", size=8, col=RED, family=MONO)
        # connectors
        for q in range(qn):
            qx = qx0 + q*(qsz+4) + qsz/2
            cxk = kv_centers[q // group]
            s += line(qx, qy + qsz, cxk, kvy, col="#c9b8bf", w=0.9)
        s += text(cx0 + pw/2, 44, title, size=12.5, col=CHARCOAL, weight="bold")
        s += text(cx0 + pw/2, 60, f"{qn} Q, {kvn} KV", size=10, col=MUTE, family=MONO)
        s += text(cx0 + pw/2, 210, note, size=11, col=RED, weight="bold")
        if i:
            s += line(cx0, 36, cx0, 224, col=BORDER, w=1)
    s += text(W/2, 30, "Query heads (top) can share key/value heads (bottom).", size=12, col=CHARCOAL)
    s += footer()
    save(OUT, "fig-kv-sharing.svg", s)


def fig_attention_flow():
    W, H = 710, 200
    s = header(W, H)
    s += text(W/2, 24, "Scaled dot-product attention, from input vectors to weighted output.",
              size=11.5, col=CHARCOAL)

    def N(cx, cy, w, lines, fill=PAPER, st=CHARCOAL):
        h = 18 + len(lines)*15
        return node(cx, cy, w, h, lines, fill=fill, stroke=st, size=9.5)

    s += N(50, 90, 52, ["X"], fill=CREAM, st=RED)
    s += N(148, 58, 62, ["Q"])
    s += N(148, 94, 62, ["K"])
    s += N(148, 150, 62, ["V"])
    s += N(268, 78, 98, ["Q·Kᵀ / √d_k"], fill=REDFILL, st=RED)
    s += N(378, 78, 82, ["causal mask"])
    s += N(482, 78, 78, ["softmax"])
    s += N(588, 112, 62, ["× V"], fill=REDFILL, st=RED)
    s += N(668, 112, 62, ["output"], fill=CREAM, st=RED)
    g = dict(col=GRAY, marker="gray", w=1.4)
    s += arrow(76, 88, 116, 60, **g)
    s += arrow(76, 90, 116, 94, **g)
    s += arrow(76, 94, 116, 148, **g)
    s += arrow(180, 60, 219, 74, **g)
    s += arrow(180, 94, 219, 82, **g)
    s += arrow(317, 78, 336, 78, col=GRAY, marker="gray", w=1.5)
    s += arrow(419, 78, 442, 78, col=GRAY, marker="gray", w=1.5)
    s += arrow(521, 82, 556, 104, col=GRAY, marker="gray", w=1.5)
    s += line(179, 150, 588, 150, col=GRAY, w=1.3)
    s += arrow(588, 150, 588, 130, col=GRAY, marker="gray", w=1.3)
    s += arrow(619, 112, 636, 112, col=GRAY, marker="gray", w=1.5)
    s += footer()
    save(OUT, "fig-attention-flow.svg", s)


ALL = [fig_attention_flow, fig_attention_steps, fig_multihead, fig_kv_sharing]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
