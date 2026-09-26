#!/usr/bin/env python3
"""Generate the Chapter 18 (RAG) figures as static SVGs.

Chapter 18 already has Figure 18.1 (mermaid pipeline); these are 18.2-18.3.
Shared helpers in svgkit.py.

Usage:
    python3 tools/figures/gen_ch18_figures.py [--png]
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch18"


# ============================================================================
# Figure 18.2 — Semantic search geometry
# ============================================================================
def fig_semantic_search():
    W, H = 600, 320
    s = header(W, H)
    s += text(W/2, 30, "Semantic search: retrieve the chunks at the smallest angle to the query.",
              size=11.5, col=CHARCOAL)
    f = Frame(ox=80, oy=280, sx=210, sy=210)
    s += arrow(f.X(0), f.Y(0), f.X(1.18), f.Y(0), col=GRID, marker="gray", w=1.2)
    s += arrow(f.X(0), f.Y(0), f.X(0), f.Y(1.18), col=GRID, marker="gray", w=1.2)
    s += text(f.X(0.6), f.Y(0) + 24, "embedding space (2 of many dimensions)", size=9.5, col=MUTE, style="italic")
    # query
    qx, qy = 0.88, 0.66
    s += arrow(f.X(0), f.Y(0), f.X(qx), f.Y(qy), col=RED, marker="red", w=2.8)
    s += text(f.X(qx) + 6, f.Y(qy) - 4, "query", size=11.5, col=RED, anchor="start", weight="bold")
    s += text(f.X(qx) + 6, f.Y(qy) + 10, '"French capital"', size=8.5, col=RED, anchor="start", family=MONO)
    # relevant chunks (small angle)
    rel = [(0.66, 0.80, '"the French capital city"'), (0.92, 0.44, '"Paris, capital of France"')]
    for cx, cy, lab in rel:
        s += arrow(f.X(0), f.Y(0), f.X(cx), f.Y(cy), col=CHARCOAL, marker="char", w=2.0)
        s += text(f.X(cx) + 6, f.Y(cy) + (12 if cy < 0.6 else -4), lab, size=8, col=CHARCOAL, anchor="start", family=MONO)
    # small-angle arc between query and nearest relevant chunk
    aq = math.degrees(math.atan2(qy, qx))
    ar = math.degrees(math.atan2(0.80, 0.66))
    s += arc(f.X(0), f.Y(0), 60, aq, ar, col=MUTE, w=1.3)
    s += text(f.X(0.42), f.Y(0.42), "small angle", size=9, col=MUTE)
    s += text(f.X(0.42), f.Y(0.42) + 12, "high cosine", size=9, col=RED)
    # irrelevant (large angle)
    s += arrow(f.X(0), f.Y(0), f.X(0.12), f.Y(0.95), col="#c9c4b8", marker="gray", w=1.8)
    s += text(f.X(0.12) + 4, f.Y(0.98), '"banana bread recipe"', size=8, col=MUTE, anchor="start", family=MONO)
    s += text(f.X(0.09), f.Y(0.5), "large angle,", size=8.5, col=MUTE, anchor="start")
    s += text(f.X(0.09), f.Y(0.5) - 11, "low cosine", size=8.5, col=MUTE, anchor="start")
    s += footer()
    save(OUT, "fig-semantic-search.svg", s)


# ============================================================================
# Figure 18.3 — Retrieve, then rerank
# ============================================================================
def fig_rerank_funnel():
    W, H = 620, 250
    s = header(W, H)
    s += text(W/2, 30, "Retrieve a wide set quickly, then rerank it accurately.", size=12, col=CHARCOAL)
    stages = [("corpus", "millions of chunks", 130, "#ece7dd", CHARCOAL),
              ("bi-encoder", "top 20 — cosine, fast", 78, REDFILL, RED),
              ("cross-encoder", "top 4 — joint, accurate", 30, REDFILL, RED),
              ("prompt", "into the LLM", 30, CREAM, RED)]
    xs = [40, 200, 370, 520]
    bw, midY = 90, 130
    for i, (title, sub, h, fill, st) in enumerate(stages):
        x = xs[i]
        s += rect(x, midY - h/2, bw, h, fill=fill, stroke=st, sw=1.2, rx=4)
        s += text(x + bw/2, midY - h/2 - 10, title, size=11, col=CHARCOAL, weight="bold")
        s += text(x + bw/2, midY + h/2 + 14, sub, size=8.5, col=MUTE)
        if i:
            s += arrow(xs[i-1] + bw + 2, midY, x - 2, midY, col=GRAY, marker="gray", w=1.5)
    s += text((xs[1]+bw+xs[2])/2, midY - 78, "cheap,", size=9, col=MUTE, style="italic")
    s += text((xs[1]+bw+xs[2])/2, midY - 66, "wide net", size=9, col=MUTE, style="italic")
    s += text((xs[2]+bw+xs[3])/2, midY - 78, "reads query +", size=9, col=RED, style="italic")
    s += text((xs[2]+bw+xs[3])/2, midY - 66, "chunk together", size=9, col=RED, style="italic")
    s += footer()
    save(OUT, "fig-rerank-funnel.svg", s)


def fig_rag_pipeline():
    W, H = 700, 260
    s = header(W, H)
    s += text(W/2, 24, "The two-phase RAG pipeline: offline indexing, then online retrieval and generation.",
              size=11, col=CHARCOAL)
    # top container: indexing
    s += rect(28, 44, 648, 78, fill="#faf7f2", stroke=BORDER, sw=1, rx=8)
    s += text(48, 58, "Indexing — offline, one-time", size=9.5, col=MUTE, anchor="start", style="italic")
    top = [(112, 86, "Documents"), (232, 66, "Chunk"), (342, 78, "Embed"), (474, 108, "Vector store")]
    tcy = 92
    for cx, w, lab in top:
        s += node(cx, tcy, w, 28, lab, fill=(REDFILL if lab == "Vector store" else PAPER),
                  stroke=(RED if lab == "Vector store" else CHARCOAL), size=9.5)
    for a, b in zip(top, top[1:]):
        s += arrow(a[0] + a[1]/2 + 2, tcy, b[0] - b[1]/2 - 2, tcy, col=GRAY, marker="gray", w=1.4)
    # bottom container: query time
    s += rect(28, 150, 648, 92, fill="#faf7f2", stroke=BORDER, sw=1, rx=8)
    s += text(48, 164, "Query time — per request", size=9.5, col=MUTE, anchor="start", style="italic")
    bcy = 200
    bot = [(96, 76, "Query", PAPER, CHARCOAL), (220, 96, "embed query", PAPER, CHARCOAL),
           (360, 118, "similarity search", REDFILL, RED), (502, 66, "LLM", PAPER, CHARCOAL),
           (606, 86, "answer", CREAM, RED)]
    for cx, w, lab, fill, st in bot:
        s += node(cx, bcy, w, 28, lab, fill=fill, stroke=st, size=9.5)
    for a, b in zip(bot, bot[1:]):
        s += arrow(a[0] + a[1]/2 + 2, bcy, b[0] - b[1]/2 - 2, bcy, col=GRAY, marker="gray", w=1.4)
    s += text((360 + 59 + 502 - 33)/2, bcy - 12, "top-k chunks", size=8, col=MUTE, style="italic")
    # vector store -> similarity search
    s += arrow(474, 106, 380, bcy - 15, col=RED, marker="red", w=1.4, dash="4,3")
    s += footer()
    save(OUT, "fig-rag-pipeline.svg", s)


ALL = [fig_rag_pipeline, fig_semantic_search, fig_rerank_funnel]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
