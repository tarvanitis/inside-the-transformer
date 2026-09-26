#!/usr/bin/env python3
"""Generate the Chapter 3 (tokenization & embeddings) figures as static SVGs.

Shared drawing helpers live in svgkit.py. Content-based filenames; the in-book
Figure 3.x number lives only in the caption.

Usage:
    python3 tools/figures/gen_ch03_figures.py            # write SVGs
    python3 tools/figures/gen_ch03_figures.py --png      # also render PNG previews
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch03"


# ============================================================================
# Figure 3.1 — Tokenization pipeline: text -> subword tokens -> token IDs
# ============================================================================
def fig_pipeline():
    W, H = 600, 300
    s = header(W, H)
    tokens = ["The", "␣cat", "␣sat"]
    ids = ["464", "3797", "3332"]
    cols = [200, 340, 480]
    bw, bh = 96, 38
    lblx = 40
    # text box
    s += rect(W/2 - 95, 34, 190, 40, fill=CREAM, stroke=RED, sw=1.3, rx=6)
    s += text(W/2, 54, '"The cat sat"', size=15, family=MONO, col=INK)
    s += text(lblx, 54, "text", size=12, col=MUTE, anchor="start", style="italic")
    # arrow down to tokens
    s += arrow(W/2, 78, W/2, 112, col=GRAY, marker="gray", w=1.7)
    s += text(W/2 + 10, 96, "tokenize (BPE)", size=11, col=MUTE, anchor="start", style="italic")
    # token boxes
    for c, tok in zip(cols, tokens):
        s += rect(c - bw/2, 126, bw, bh, fill=PAPER, stroke=CHARCOAL, sw=1.1, rx=5)
        s += text(c, 145, tok, size=14, family=MONO, col=INK)
    s += text(lblx, 145, "tokens", size=12, col=MUTE, anchor="start", style="italic")
    # per-token arrows to IDs
    for c in cols:
        s += arrow(c, 166, c, 208, col=GRAY, marker="gray", w=1.4)
    s += text(cols[-1] + 58, 187, "look up ID", size=11, col=MUTE, anchor="start", style="italic")
    # id boxes
    for c, idv in zip(cols, ids):
        s += rect(c - bw/2, 222, bw, bh, fill=REDFILL, stroke=RED, sw=1.1, rx=5)
        s += text(c, 241, idv, size=14, family=MONO, col=RED, weight="bold")
    s += text(lblx, 241, "token IDs", size=12, col=MUTE, anchor="start", style="italic")
    s += text(W/2, 284, "␣ marks the space that belongs to the token.",
              size=10.5, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-tokenization-pipeline.svg", s)


# ============================================================================
# Figure 3.2 — One-hot vs dense embedding
# ============================================================================
def fig_onehot_dense():
    W, H = 620, 250
    s = header(W, H)
    x0 = 42
    # one-hot (top)
    s += text(x0, 34, "one-hot", size=13, col=CHARCOAL, weight="bold", anchor="start")
    n, cw, y = 24, 13, 48
    hot = 7
    for k in range(n):
        f = REDFILL if k == hot else "#ffffff"
        st = RED if k == hot else "#cfcabf"
        s += rect(x0 + k*cw, y, cw, cw, fill=f, stroke=st, sw=0.8)
        if k == hot:
            s += text(x0 + k*cw + cw/2, y + cw/2, "1", size=9, col=RED, family=MONO, weight="bold")
        elif k in (0, 1, 2, n-1, n-2):
            s += text(x0 + k*cw + cw/2, y + cw/2, "0", size=8, col="#b9b4a8", family=MONO)
    s += text(x0 + n*cw + 8, y + cw/2, "…", size=13, col=MUTE, anchor="start")
    s += text(x0, y + cw + 20, "50,000 dimensions: a single 1, everything else 0. Says nothing about similarity.",
              size=11, col=MUTE, anchor="start", style="italic")
    # dense (bottom)
    s += text(x0, 150, "dense embedding", size=13, col=CHARCOAL, weight="bold", anchor="start")
    dense = [0.23, -0.15, 0.78, 0.41, -0.33, 0.52, 0.11, -0.27]
    cw2, y2 = 52, 164
    for k, v in enumerate(dense):
        f = REDFILL if v >= 0 else BLUEFILL
        s += rect(x0 + k*cw2, y2, cw2, 30, fill=f, stroke="#c9c4b8", sw=0.8)
        s += text(x0 + k*cw2 + cw2/2, y2 + 15, f"{v:+.2f}", size=10.5, family=MONO, col=INK)
    s += text(x0 + len(dense)*cw2 + 8, y2 + 15, "…", size=13, col=MUTE, anchor="start")
    s += text(x0, y2 + 30 + 20, "A few thousand dimensions, every number free to vary and carry meaning.",
              size=11, col=MUTE, anchor="start", style="italic")
    s += footer()
    save(OUT, "fig-onehot-vs-dense.svg", s)


# ============================================================================
# Figure 3.3 — The embedding matrix is a lookup
# ============================================================================
def fig_embedding_lookup():
    W, H = 640, 320
    s = header(W, H)
    ids = ["464", "3797", "3332"]
    id_y = [80, 150, 220]
    for idv, y in zip(ids, id_y):
        s += rect(28, y, 64, 32, fill=REDFILL, stroke=RED, sw=1, rx=4)
        s += text(60, y + 16, idv, size=13, family=MONO, col=RED, weight="bold")
    s += text(60, 56, "token IDs", size=11, col=MUTE, style="italic")
    # matrix E
    ex, ey, ew, eh = 200, 56, 150, 210
    nrows = 12
    rh = eh / nrows
    hl = {2: 0, 6: 1, 9: 2}   # drawn-row index -> id index
    s += rect(ex, ey, ew, eh, fill="#ffffff", stroke=CHARCOAL, sw=1.2)
    for r in range(nrows):
        ry = ey + r*rh
        if r in hl:
            s += rect(ex, ry, ew, rh, fill=REDFILL, stroke=RED, sw=1)
        if r:
            s += line(ex, ry, ex + ew, ry, col=GRID, w=0.6)
    s += text(ex + ew/2, ey - 14, "E   (50000 × 768)", size=12, col=INK, weight="bold")
    s += text(ex + ew/2, ey + eh + 16, "one row per token", size=10.5, col=MUTE, style="italic")
    # arrows: id -> highlighted row
    for r, i in hl.items():
        ry = ey + r*rh + rh/2
        s += arrow(94, id_y[i] + 16, ex - 2, ry, col=GRAY, marker="gray", w=1.3)
    # output vectors on the right
    vx = 430
    vcells = [[0.23, -0.15, 0.78], [0.25, -0.14, 0.76], [-0.54, 0.82, -0.11]]
    cw = 44
    for i, row in enumerate(vcells):
        ry = ey + list(hl.keys())[i]*rh + rh/2 - 15
        # arrow from matrix row to this vector
        s += arrow(ex + ew + 2, ey + list(hl.keys())[i]*rh + rh/2, vx - 2, ry + 15, col=RED, marker="red", w=1.3)
        for k, v in enumerate(row):
            s += rect(vx + k*cw, ry, cw, 30, fill=REDFILL if v >= 0 else BLUEFILL, stroke="#c9c4b8", sw=0.7)
            s += text(vx + k*cw + cw/2, ry + 15, f"{v:+.2f}", size=9.5, family=MONO, col=INK)
        s += text(vx + 3*cw + 6, ry + 15, "…", size=12, col=MUTE, anchor="start")
    s += text(vx + 66, ey - 14, "embedding vectors  (3 × 768)", size=12, col=INK, weight="bold")
    s += text(W/2, H - 12, "Reading a token's vector is just picking its row from E.",
              size=10.5, col=MUTE, style="italic")
    s += footer()
    save(OUT, "fig-embedding-lookup.svg", s)


# ============================================================================
# Figure 3.4 — Meaning has shape: clusters + the analogy parallelogram
# ============================================================================
def fig_embedding_geometry():
    W, H = 640, 340
    s = header(W, H)
    # ---- left panel: clusters ----
    s += panel(16, 20, 300, 300, "Similar meanings sit close")
    fL = Frame(ox=70, oy=280, sx=210, sy=210)
    s += arrow(fL.X(0), fL.Y(0), fL.X(1.08), fL.Y(0), col=GRID, marker="gray", w=1.2)
    s += arrow(fL.X(0), fL.Y(0), fL.X(0), fL.Y(1.12), col=GRID, marker="gray", w=1.2)
    cluster = [("cat", 0.70, 0.74), ("kitten", 0.80, 0.66), ("feline", 0.62, 0.82)]
    for name, x, y in cluster:
        s += circle(fL.X(x), fL.Y(y), 3.4, fill=RED)
        s += text(fL.X(x) + (7 if name != "feline" else -7), fL.Y(y) + (-8 if name == "feline" else 12),
                  name, size=12, col=RED, family=MONO,
                  anchor="end" if name == "feline" else "start")
    s += (f'<ellipse cx="{fL.X(0.71):.1f}" cy="{fL.Y(0.74):.1f}" rx="40" ry="40" '
          f'fill="none" stroke="{RED}" stroke-width="1.1" stroke-dasharray="3,3" opacity="0.7"/>\n')
    s += circle(fL.X(0.22), fL.Y(0.24), 3.4, fill=CHARCOAL)
    s += text(fL.X(0.22) + 8, fL.Y(0.24) + 4, "democracy", size=12, col=CHARCOAL, family=MONO, anchor="start")
    s += text(166, 306, "far from the animal cluster", size=10.5, col=MUTE, style="italic")
    # ---- right panel: analogy parallelogram ----
    s += panel(336, 20, 288, 300, "Relationships are directions")
    fR = Frame(ox=372, oy=286, sx=232, sy=232)
    pts = {"man": (0.16, 0.20), "woman": (0.16, 0.66),
           "king": (0.66, 0.36), "queen": (0.66, 0.82)}
    # gender edges (charcoal), royalty edges (red)
    def seg(a, b, col, mk, dash=None):
        return arrow(fR.X(pts[a][0]), fR.Y(pts[a][1]), fR.X(pts[b][0]), fR.Y(pts[b][1]),
                     col=col, marker=mk, w=2.0, dash=dash)
    s += seg("man", "king", RED, "red")
    s += seg("woman", "queen", RED, "red", dash="5,3")
    s += seg("man", "woman", CHARCOAL, "char")
    s += seg("king", "queen", CHARCOAL, "char", dash="5,3")
    labels = {"man": (-6, 14, "end"), "woman": (-6, -8, "end"),
              "king": (8, 14, "start"), "queen": (8, -8, "start")}
    for name, (x, y) in pts.items():
        s += circle(fR.X(x), fR.Y(y), 3.6, fill=INK)
        dx, dy, anc = labels[name]
        s += text(fR.X(x) + dx, fR.Y(y) + dy, name, size=12.5, col=INK, family=MONO, anchor=anc)
    s += text(fR.X(0.42), fR.Y(0.28), "royalty", size=10.5, col=RED, style="italic")
    s += text(fR.X(0.05), fR.Y(0.43), "gender", size=10.5, col=CHARCOAL, style="italic", anchor="start")
    s += text(480, 306, "king − man + woman ≈ queen", size=12, col=INK, family=MONO)
    s += footer()
    save(OUT, "fig-embedding-geometry.svg", s)


ALL = [fig_pipeline, fig_onehot_dense, fig_embedding_lookup, fig_embedding_geometry]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
