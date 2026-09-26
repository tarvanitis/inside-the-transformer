#!/usr/bin/env python3
"""Generate the Chapter 2 (mathematics) figures as static SVGs.

Draft set for *Inside the Transformer*. Geometry and every plotted value are
computed here so the figures stay arithmetically exact and match the worked
examples in ch-02-mathematics.md. Shared drawing helpers live in svgkit.py.

Usage:
    python3 tools/figures/gen_ch02_figures.py            # write SVGs
    python3 tools/figures/gen_ch02_figures.py --png      # also render PNG previews

Output: assets/figures/ch02/fig-*.svg  (content-based names; the in-book
Figure 2.x number lives only in the caption, so it never drifts)
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svgkit import *  # noqa: F401,F403  palette, helpers, Frame, save, render_previews

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "figures" / "ch02"

def write(name, svg):
    save(OUT, name, svg)


# ============================================================================
# Figure 2.1 — Vectors in space: similar words sit near one another
# ============================================================================
def fig_2_1():
    W, H = 560, 420
    s = header(W, H)
    f = Frame(ox=90, oy=350, sx=280, sy=280)   # unit square 0..1 -> 280px
    # axes
    s += arrow(f.X(0), f.Y(0), f.X(1.08), f.Y(0), col=GRAY, marker="gray", w=1.6)
    s += arrow(f.X(0), f.Y(0), f.X(0), f.Y(1.12), col=GRAY, marker="gray", w=1.6)
    s += text(f.X(0.55), f.Y(0) + 34, "“animal-ness”  (dimension 1)", size=12, col=MUTE)
    s += ('<text x="%.1f" y="%.1f" font-size="12" fill="%s" text-anchor="middle" '
          'font-family="%s" transform="rotate(-90 %.1f %.1f)">“living-ness”  (dimension 2)</text>\n'
          % (f.X(0) - 44, f.Y(0.55), MUTE, SERIF, f.X(0) - 44, f.Y(0.55)))
    # gridlines at 0.5
    for t in (0.5, 1.0):
        s += line(f.X(t), f.Y(0), f.X(t), f.Y(1.1), col=GRID, w=1, dash="2,3")
        s += line(f.X(0), f.Y(t), f.X(1.08), f.Y(t), col=GRID, w=1, dash="2,3")
    pts = [("cat",  0.91, 0.88, RED,      "red",  ( 8, -10)),
           ("dog",  0.89, 0.80, RED,      "red",  (10,  16)),
           ("king", 0.12, 0.85, CHARCOAL, "char", (10, -12)),
           ("car",  0.14, 0.12, CHARCOAL, "char", (12,  -8))]
    for name, x, y, col, mk, (lx, ly) in pts:
        s += arrow(f.X(0), f.Y(0), f.X(x), f.Y(y), col=col, marker=mk, w=2.0)
        s += circle(f.X(x), f.Y(y), 3.4, fill=col)
        s += text(f.X(x) + lx, f.Y(y) + ly, name, size=13.5, col=col,
                  anchor="start" if lx >= 0 else "end", family=MONO, weight="bold")
    # cluster ring around cat/dog
    s += (f'<ellipse cx="{f.X(0.90):.1f}" cy="{f.Y(0.84):.1f}" rx="30" ry="34" '
          f'fill="none" stroke="{RED}" stroke-width="1.1" stroke-dasharray="3,3" opacity="0.7"/>\n')
    s += text(f.X(0.90), f.Y(0.84) - 46, "similar words", size=11.5, col=RED, style="italic")
    s += text(W/2, H - 12,
              "Toy 2-D space. A real model uses several thousand dimensions at once.",
              size=11, col=MUTE, style="italic")
    s += footer()
    write("fig-vectors-in-space.svg", s)

# ============================================================================
# Figure 2.2 — Dot product = alignment (same a, three different b)
# ============================================================================
def fig_2_2():
    W, H = 620, 300
    s = header(W, H)
    panels = [("point the same way", "a · b > 0", (2.0, 1.4), RED),
              ("perpendicular",      "a · b = 0", (-0.9, 2.4), RED),
              ("point apart",        "a · b < 0", (-2.2, -1.5), RED)]
    a = (2.6, 0.4)   # fixed reference vector (charcoal)
    pw = W / 3
    for i, (cap, val, b, _) in enumerate(panels):
        cx0 = i * pw
        f = Frame(ox=cx0 + pw/2, oy=180, sx=34, sy=34)
        ox, oy = f.X(0), f.Y(0)
        # light axes
        s += line(cx0 + 16, oy, cx0 + pw - 16, oy, col=GRID, w=1)
        s += line(f.X(0), 40, f.X(0), 250, col=GRID, w=1)
        # vectors
        s += arrow(ox, oy, f.X(a[0]), f.Y(a[1]), col=CHARCOAL, marker="char", w=2.2)
        s += arrow(ox, oy, f.X(b[0]), f.Y(b[1]), col=RED, marker="red", w=2.2)
        # angle arc
        a1 = math.degrees(math.atan2(a[1], a[0]))
        a2 = math.degrees(math.atan2(b[1], b[0]))
        s += arc(ox, oy, 22, a1, a2, col=GRAY, w=1.2)
        # labels
        s += text(f.X(a[0]) + 6, f.Y(a[1]) - 6, "a", size=14, col=CHARCOAL, family=MONO,
                  weight="bold", anchor="start")
        s += text(f.X(b[0]) + (8 if b[0] >= 0 else -8), f.Y(b[1]) + (-6 if b[1] >= 0 else 14),
                  "b", size=14, col=RED, family=MONO, weight="bold",
                  anchor="start" if b[0] >= 0 else "end")
        if abs(a2 - a1 - 90) < 25 or abs(a2 - a1 + 90) < 25:
            ua = (math.cos(math.radians(a1)), math.sin(math.radians(a1)))
            ub = (math.cos(math.radians(a2)), math.sin(math.radians(a2)))
            s += right_angle(ox, oy, ua, ub, size=12, col=GRAY)
        s += text(cx0 + pw/2, 268, val, size=14, col=RED, family=MONO, weight="bold")
        s += text(cx0 + pw/2, 40, cap, size=12.5, col=CHARCOAL, style="italic")
        if i:
            s += line(cx0, 30, cx0, 285, col=BORDER, w=1)
    s += footer()
    write("fig-dot-product-alignment.svg", s)

# ============================================================================
# Figure 2.3 — Cosine ignores length (left) + compass needles (right)
# ============================================================================
def fig_2_3():
    W, H = 620, 320
    s = header(W, H)
    # left panel: same direction, different length
    s += panel(16, 20, 300, 284, "Same direction, different length")
    f = Frame(ox=70, oy=250, sx=42, sy=42)
    s += line(f.X(0), f.Y(0), f.X(4.7), f.Y(0), col=GRID, w=1)
    s += line(f.X(0), f.Y(0), f.X(0), f.Y(3.4), col=GRID, w=1)
    a = (2, 1); b = (4, 2)
    s += arrow(f.X(0), f.Y(0), f.X(b[0]), f.Y(b[1]), col=RED, marker="red", w=2.4)
    s += arrow(f.X(0), f.Y(0), f.X(a[0]), f.Y(a[1]), col=CHARCOAL, marker="char", w=2.4)
    s += text(f.X(a[0]) + 6, f.Y(a[1]) + 15, "a = [2, 1]", size=12, col=CHARCOAL, family=MONO, anchor="start")
    s += text(f.X(b[0]) - 4, f.Y(b[1]) - 10, "b = [4, 2]", size=12, col=RED, family=MONO, anchor="end")
    s += arc(f.X(0), f.Y(0), 26, 0, 27, col=GRAY, w=1.1)
    s += text(166, 240, "angle between them = 0°", size=12, col=INK)
    s += text(166, 259, "cosine = 1.0  (same direction)", size=12.5, col=RED, weight="bold")
    s += text(166, 278, "but dot product 5 → 10  (b is twice as long)", size=11.5, col=MUTE)
    # right panel: compass needles
    s += panel(336, 20, 268, 284, "Cosine reads the angle only")
    trip = [(0, "1"), (90, "0"), (180, "−1")]
    bx = 470
    for i, (deg, val) in enumerate(trip):
        cy = 88 + i * 74
        s += circle(bx, cy, 26, fill=CREAM, stroke=BORDER, sw=1.2)
        # north needle (reference) + second needle
        s += arrow(bx, cy, bx, cy - 22, col=CHARCOAL, marker="char", w=2.0)
        dx = 22 * math.cos(math.radians(90 - deg)); dy = 22 * math.sin(math.radians(90 - deg))
        s += arrow(bx, cy, bx + dx, cy - dy, col=RED, marker="red", w=2.0)
        label = {0: "north vs north", 90: "north vs east", 180: "north vs south"}[deg]
        s += text(bx + 44, cy - 6, label, size=12, col=INK, anchor="start")
        s += text(bx + 44, cy + 12, "cos = " + val, size=12.5, col=RED, anchor="start", family=MONO, weight="bold")
    s += footer()
    write("fig-cosine-length-invariance.svg", s)

# ============================================================================
# Figure 2.4 — Transpose = reflection across the main diagonal
# ============================================================================
def fig_2_4():
    W, H = 600, 300
    s = header(W, H)
    cell = 46
    A = [[1, 2, 3], [4, 5, 6]]
    rowcol = [RED, CHARCOAL]     # row 0 red, row 1 charcoal
    # grid A (2x3) at left
    ax, ay = 70, 96
    s += text(ax + 3*cell/2, ay - 16, "A   (2 × 3)", size=13, col=INK, weight="bold")
    for r in range(2):
        for c in range(3):
            s += rect(ax + c*cell, ay + r*cell, cell, cell, fill=(REDFILL if r == 0 else GRAYFILL),
                      stroke="#bdb8ae", sw=1)
            s += text(ax + c*cell + cell/2, ay + r*cell + cell/2, A[r][c], size=15,
                      col=rowcol[r], family=MONO, weight="bold")
    # main diagonal on A
    s += line(ax, ay, ax + 2*cell, ay + 2*cell, col=RED, w=1.6, dash="4,3")
    s += text(ax + 2*cell + 30, ay + 2*cell + 4, "diagonal", size=11, col=RED, style="italic", anchor="start")
    # flip arrow
    s += arc(300, 150, 30, 210, 150, col=GRAY, w=1.6)
    s += ('<path d="M323,128 l6,-4 l-1,8 z" fill="%s"/>\n' % GRAY)
    s += text(300, 118, "flip", size=12, col=MUTE, style="italic")
    s += text(300, 196, "rows ↔ columns", size=11, col=MUTE, style="italic")
    # grid A^T (3x2) at right
    tx, ty = 380, 74
    s += text(tx + 2*cell/2, ty - 16, "Aᵀ   (3 × 2)", size=13, col=INK, weight="bold")
    for r in range(3):
        for c in range(2):
            v = A[c][r]
            s += rect(tx + c*cell, ty + r*cell, cell, cell, fill=(REDFILL if c == 0 else GRAYFILL),
                      stroke="#bdb8ae", sw=1)
            s += text(tx + c*cell + cell/2, ty + r*cell + cell/2, v, size=15,
                      col=rowcol[c], family=MONO, weight="bold")
    s += text(W/2, H - 16, "Row 1 of A (red) becomes column 1 of Aᵀ. The grid is reflected across its diagonal.",
              size=11.5, col=MUTE, style="italic")
    s += footer()
    write("fig-transpose-diagonal.svg", s)

# ============================================================================
# Figure 2.5 — Matrix multiply: row . column -> cell, and the shape rule
# ============================================================================
def fig_2_5():
    W, H = 640, 360
    s = header(W, H)
    cell = 40
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    C = [[58, 64], [139, 154]]
    ax, ay = 40, 70
    bx, by = 250, 40
    cx, cy = 250, 210
    # A (2x3), highlight row 0
    s += text(ax + 3*cell/2, ay - 14, "A", size=13, col=INK, weight="bold")
    for r in range(2):
        for c in range(3):
            hl = (r == 0)
            s += rect(ax + c*cell, ay + r*cell, cell, cell, fill=(REDFILL if hl else "#fff"),
                      stroke="#bdb8ae", sw=1)
            s += text(ax + c*cell + cell/2, ay + r*cell + cell/2, A[r][c], size=14,
                      col=(RED if hl else INK), family=MONO, weight=("bold" if hl else "normal"))
    # B (3x2), highlight col 0
    s += text(bx + 2*cell/2, by - 14, "B", size=13, col=INK, weight="bold")
    for r in range(3):
        for c in range(2):
            hl = (c == 0)
            s += rect(bx + c*cell, by + r*cell, cell, cell, fill=(REDFILL if hl else "#fff"),
                      stroke="#bdb8ae", sw=1)
            s += text(bx + c*cell + cell/2, by + r*cell + cell/2, B[r][c], size=14,
                      col=(RED if hl else INK), family=MONO, weight=("bold" if hl else "normal"))
    # C (2x2), highlight cell 0,0
    s += text(cx + 2*cell/2, cy - 14, "C = A × B", size=13, col=INK, weight="bold")
    for r in range(2):
        for c in range(2):
            hl = (r == 0 and c == 0)
            s += rect(cx + c*cell, cy + r*cell, cell, cell, fill=(REDFILL if hl else "#fff"),
                      stroke="#bdb8ae", sw=1)
            s += text(cx + c*cell + cell/2, cy + r*cell + cell/2, C[r][c], size=14,
                      col=(RED if hl else INK), family=MONO, weight=("bold" if hl else "normal"))
    # the arithmetic
    s += text(400, cy + cell/2, "row 1 · col 1", size=12.5, col=RED, anchor="start", weight="bold")
    s += text(400, cy + cell/2 + 20, "(1×7)+(2×9)+(3×11) = 58", size=12.5, col=INK, anchor="start", family=MONO)
    # shape rule strip
    yb = 300
    s += line(40, yb - 18, W - 40, yb - 18, col=BORDER, w=1)
    s += text(40, yb, "The shape rule:", size=12.5, col=CHARCOAL, anchor="start", weight="bold")
    def shape(x, y, txt, hl=False):
        return text(x, y, txt, size=13, col=(RED if hl else INK), anchor="start", family=MONO)
    s += shape(160, yb, "[2×", ); s += shape(196, yb, "3", hl=True); s += shape(210, yb, "]")
    s += shape(226, yb, "· [", ); s += shape(246, yb, "3", hl=True); s += shape(260, yb, "×2]  →  [2×2]")
    s += text(160, yb + 20, "inner numbers must match (3 = 3); the result is outer × outer.",
              size=11, col=MUTE, anchor="start", style="italic")
    s += text(160, yb + 40, "In attention:  Q [5×64] · Kᵀ [64×5]  →  [5×5] score grid.",
              size=11.5, col=CHARCOAL, anchor="start", family=MONO)
    s += footer()
    write("fig-matmul-shapes.svg", s)

# ============================================================================
# Figure 2.6 — Rotation of a vector, and the RoPE relative-angle idea
# ============================================================================
def fig_2_6():
    W, H = 620, 340
    s = header(W, H)
    # left: rotate [1,0] by 45
    s += panel(16, 20, 300, 300, "Rotating a vector by 45°")
    f = Frame(ox=90, oy=250, sx=120, sy=120)
    s += arrow(f.X(-0.15), f.Y(0), f.X(1.35), f.Y(0), col=GRID, marker="gray", w=1.2)
    s += arrow(f.X(0), f.Y(-0.15), f.X(0), f.Y(1.35), col=GRID, marker="gray", w=1.2)
    s += arrow(f.X(0), f.Y(0), f.X(1), f.Y(0), col=CHARCOAL, marker="char", w=2.4, dash="5,3")
    r = 0.70710678
    s += arrow(f.X(0), f.Y(0), f.X(r), f.Y(r), col=RED, marker="red", w=2.6)
    s += arc(f.X(0), f.Y(0), 40, 0, 45, col=GRAY, w=1.3)
    s += text(f.X(0) + 52, f.Y(0) - 20, "45°", size=12.5, col=MUTE)
    s += text(f.X(1) + 6, f.Y(0) + 16, "[1, 0]", size=12, col=CHARCOAL, family=MONO, anchor="start")
    s += text(f.X(r) + 8, f.Y(r) - 6, "[0.707, 0.707]", size=12, col=RED, family=MONO, anchor="start")
    # right: RoPE relative angle
    s += panel(336, 20, 268, 300, "Why position becomes an angle")
    cx, cy = 400, 232
    s += circle(cx, cy, 3, fill=INK)
    def ray(angle_deg, r=86):
        a = math.radians(angle_deg)
        return cx + r*math.cos(a), cy - r*math.sin(a)
    mx, my = ray(74); nx, ny = ray(30)
    s += arrow(cx, cy, mx, my, col=CHARCOAL, marker="char", w=2.3)
    s += arrow(cx, cy, nx, ny, col=RED, marker="red", w=2.3)
    s += text(mx + 6, my - 4, "position m", size=11.5, col=CHARCOAL, anchor="start", family=MONO)
    s += text(nx + 6, ny + 12, "position n", size=11.5, col=RED, anchor="start", family=MONO)
    s += arc(cx, cy, 40, 30, 74, col=GRAY, w=1.4)
    s += text(cx + 52, cy - 30, "gap", size=11.5, col=MUTE, anchor="middle", style="italic")
    s += text(470, 300, "the gap depends only on m − n,", size=11.5, col=INK)
    s += text(470, 316, "not where the pair sits in the text", size=11.5, col=INK)
    s += footer()
    write("fig-rotation-rope.svg", s)

# ============================================================================
# Figure 2.7 — Softmax turns raw scores into probabilities
# ============================================================================
def fig_2_7():
    W, H = 620, 320
    s = header(W, H)
    labels = ["mat", "floor", "table"]
    logits = [3.0, 1.5, 0.2]
    probs = [0.779, 0.174, 0.047]
    # left: logits (raw)
    s += text(150, 40, "raw scores (logits)", size=13, col=CHARCOAL, weight="bold")
    baseY = 250; bw = 46; gap = 34
    x0 = 70; sc = 56  # px per unit logit
    for i, (lb, v) in enumerate(zip(labels, logits)):
        x = x0 + i*(bw+gap); h = v*sc
        s += rect(x, baseY - h, bw, h, fill=GRAYFILL, stroke=CHARCOAL, sw=1.1)
        s += text(x + bw/2, baseY - h - 10, f"{v:.1f}", size=12, col=CHARCOAL, family=MONO)
        s += text(x + bw/2, baseY + 16, lb, size=11.5, col=INK, family=MONO)
    s += line(x0 - 10, baseY, x0 + 3*(bw+gap), baseY, col=GRAY, w=1.3)
    # arrow
    s += arrow(292, 170, 340, 170, col=RED, marker="red", w=2.4)
    s += text(316, 154, "softmax", size=12.5, col=RED, style="italic")
    # right: probabilities
    s += text(470, 40, "probabilities (sum = 1)", size=13, col=RED, weight="bold")
    x0 = 372; scp = 210  # px per unit prob (0..1)
    axisY0 = baseY; axisY1 = baseY - scp
    s += line(x0 - 12, axisY0, x0 - 12, axisY1, col=GRAY, w=1.2)
    for t in (0, 0.5, 1.0):
        yy = baseY - t*scp
        s += line(x0 - 16, yy, x0 - 8, yy, col=GRAY, w=1)
        s += text(x0 - 22, yy, f"{t:.1f}", size=10, col=MUTE, anchor="end", family=MONO)
    for i, (lb, v) in enumerate(zip(labels, probs)):
        x = x0 + i*(bw+gap); h = v*scp
        s += rect(x, baseY - h, bw, h, fill=REDFILL, stroke=RED, sw=1.2)
        s += text(x + bw/2, baseY - h - 10, f"{v*100:.0f}%", size=12, col=RED, family=MONO, weight="bold")
        s += text(x + bw/2, baseY + 16, lb, size=11.5, col=INK, family=MONO)
    s += line(x0 - 12, baseY, x0 + 3*(bw+gap), baseY, col=GRAY, w=1.3)
    s += text(W/2, H - 14, "A 2× lead in the scores (3.0 vs 1.5) becomes a 4.5× lead in probability (78% vs 17%).",
              size=11.5, col=MUTE, style="italic")
    s += footer()
    write("fig-softmax-bars.svg", s)

# ============================================================================
# Figure 2.8 — The sigmoid S-curve
# ============================================================================
def fig_2_8():
    W, H = 560, 340
    s = header(W, H)
    f = Frame(ox=295, oy=285, sx=37.5, sy=210)   # x in [-6,6], y in [0,1]
    # frame box / gridlines
    for yv in (0.0, 0.5, 1.0):
        s += line(f.X(-6), f.Y(yv), f.X(6), f.Y(yv), col=GRID, w=1, dash=("1,0" if yv==0 else "2,3"))
        s += text(f.X(-6) - 10, f.Y(yv), f"{yv:.1f}", size=11, col=MUTE, anchor="end", family=MONO)
    s += line(f.X(0), f.Y(0), f.X(0), f.Y(1.05), col=GRID, w=1)
    s += arrow(f.X(-6.4), f.Y(0), f.X(6.4), f.Y(0), col=GRAY, marker="gray", w=1.4)
    s += text(f.X(6.2), f.Y(0) + 18, "x", size=12, col=MUTE, family=MONO)
    s += text(f.X(0) + 12, f.Y(1.0) - 6, "σ(x)", size=12, col=MUTE, family=MONO, anchor="start")
    # curve
    pts = []
    x = -6.0
    while x <= 6.0001:
        y = 1/(1+math.exp(-x)); pts.append(f"{f.X(x):.2f},{f.Y(y):.2f}"); x += 0.1
    s += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{RED}" stroke-width="2.6"/>\n'
    # worked points
    for xv in (-5, -2, 0, 2, 5):
        yv = 1/(1+math.exp(-xv))
        s += circle(f.X(xv), f.Y(yv), 3.6, fill=RED)
        s += text(f.X(xv), f.Y(yv) + (-14 if xv >= 0 else 16),
                  f"σ({xv})={yv:.3f}".rstrip("0").rstrip("."), size=10.5, col=INK, family=MONO)
        s += line(f.X(xv), f.Y(0), f.X(xv), f.Y(0)+4, col=GRAY, w=1)
        s += text(f.X(xv), f.Y(0)+16, str(xv), size=10, col=MUTE, family=MONO)
    s += text(W/2, 26, "Squashes any number into (0, 1); crosses 0.5 at x = 0.", size=12, col=CHARCOAL)
    s += footer()
    write("fig-sigmoid.svg", s)

# ============================================================================
# Figure 2.9 — Gradient descent on L=(w-3)^2, plus learning-rate regimes
# ============================================================================
def fig_2_9():
    W, H = 640, 340
    s = header(W, H)
    # main parabola
    s += panel(16, 20, 380, 300, "Rolling downhill: L = (w − 3)²")
    f = Frame(ox=70, oy=280, sx=48, sy=26)   # w in [0,6] -> 288px; L in [0,9]
    s += arrow(f.X(-0.2), f.Y(0), f.X(6.4), f.Y(0), col=GRAY, marker="gray", w=1.3)
    s += arrow(f.X(0), f.Y(-0.3), f.X(0), f.Y(9.6), col=GRAY, marker="gray", w=1.3)
    s += text(f.X(6.2), f.Y(0)+16, "w", size=12, col=MUTE, family=MONO)
    s += text(f.X(0)-8, f.Y(9.2), "L", size=12, col=MUTE, family=MONO, anchor="end")
    pts = []
    w = 0.0
    while w <= 6.0001:
        L = (w-3)**2; pts.append(f"{f.X(w):.2f},{f.Y(L):.2f}"); w += 0.1
    s += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{CHARCOAL}" stroke-width="2.2"/>\n'
    # descent steps from w=5
    seq = [5.0]
    for _ in range(4):
        w = seq[-1]; seq.append(round(w - 0.1*2*(w-3), 3))
    for i, w in enumerate(seq):
        L = (w-3)**2
        s += circle(f.X(w), f.Y(L), 4.2, fill=RED)
        if i:
            pw = seq[i-1]; pL = (pw-3)**2
            s += arrow(f.X(pw), f.Y(pL), f.X(w), f.Y(L), col=RED, marker="red", w=1.6)
    s += text(f.X(5), f.Y(4) - 12, "start w=5", size=11, col=RED)
    s += circle(f.X(3), f.Y(0), 4.5, fill="none", stroke=RED, sw=1.6)
    s += text(f.X(3), f.Y(0) - 14, "minimum w=3", size=11, col=RED)
    # right: three learning-rate regimes
    s += panel(410, 20, 214, 300, "Choosing the step size α")
    reg = [("α too small", "crawls", GRAY),
           ("α good", "converges", RED),
           ("α too large", "overshoots → oscillates", CHARCOAL)]
    for i, (name, note, col) in enumerate(reg):
        yy = 74 + i*80
        g = Frame(ox=440, oy=yy+30, sx=17, sy=7)
        pts = []
        w = 0.2
        while w <= 6.0001:
            pts.append(f"{g.X(w):.1f},{g.Y((w-3)**2):.1f}"); w += 0.2
        s += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{GRID}" stroke-width="1.6"/>\n'
        if i == 0:
            xs = [5, 4.8, 4.62, 4.46]
        elif i == 1:
            xs = [5, 4.0, 3.4, 3.1, 3.0]
        else:
            xs = [5, 1, 5, 1]
        for j, w in enumerate(xs):
            s += circle(g.X(w), g.Y((w-3)**2), 3, fill=col)
            if j:
                pw = xs[j-1]
                s += arrow(g.X(pw), g.Y((pw-3)**2), g.X(w), g.Y((w-3)**2), col=col, marker=("red" if col==RED else "char" if col==CHARCOAL else "gray"), w=1.3)
        s += text(590, yy - 2, name, size=11.5, col=col, anchor="end", weight="bold")
        s += text(590, yy + 12, note, size=10.5, col=MUTE, anchor="end", style="italic")
    s += footer()
    write("fig-gradient-descent.svg", s)

# ============================================================================
# Figure 2.10 — Temperature reshapes the distribution
# ============================================================================
def fig_2_10():
    W, H = 620, 300
    s = header(W, H)
    groups = [("τ = 0.5", "sharper", [0.84, 0.11, 0.04]),
              ("τ = 1.0", "default", [0.63, 0.23, 0.14]),
              ("τ = 1.5", "flatter", [0.53, 0.27, 0.20])]
    labels = ["A", "B", "C"]
    pw = W/3
    for i, (title, note, ps) in enumerate(groups):
        cx0 = i*pw
        baseY = 220; sc = 150; bw = 34; x0 = cx0 + 40
        s += line(x0 - 10, baseY, x0 + 3*bw + 20, baseY, col=GRAY, w=1.2)
        for j, p in enumerate(ps):
            x = x0 + j*(bw+12); h = p*sc
            col = RED if j == 0 else GRAYFILL
            stroke = RED if j == 0 else CHARCOAL
            s += rect(x, baseY - h, bw, h, fill=(REDFILL if j == 0 else GRAYFILL), stroke=stroke, sw=1.1)
            s += text(x + bw/2, baseY - h - 9, f"{p:.2f}", size=10.5,
                      col=(RED if j == 0 else CHARCOAL), family=MONO)
            s += text(x + bw/2, baseY + 15, labels[j], size=10.5, col=MUTE, family=MONO)
        s += text(cx0 + pw/2, 44, title, size=14, col=INK, weight="bold", family=MONO)
        s += text(cx0 + pw/2, 62, note, size=11.5, col=RED if i != 1 else MUTE, style="italic")
        if i:
            s += line(cx0, 32, cx0, 250, col=BORDER, w=1)
    s += text(W/2, H - 12, "Same logits [2.0, 1.0, 0.5]. Low τ sharpens toward the top token; high τ flattens.",
              size=11.5, col=MUTE, style="italic")
    s += footer()
    write("fig-temperature.svg", s)

# ============================================================================
# Figure 2.11 — Quantization: snapping to the nearest grid value
# ============================================================================
def fig_2_11():
    W, H = 620, 260
    s = header(W, H)
    s += text(W/2, 30, "Quantization: round each weight to the nearest step on a coarse grid",
              size=13, col=CHARCOAL)
    y = 150
    x0, x1 = 60, 560
    lo, hi = -1.0, 1.0
    def MX(v): return x0 + (v - lo)/(hi - lo) * (x1 - x0)
    s += arrow(x0 - 16, y, x1 + 16, y, col=GRAY, marker="gray", w=1.4)
    # illustrative coarse grid (step 0.125 -> 17 ticks; real int8 has 256)
    step = 0.125
    n = int(round((hi - lo)/step))
    for k in range(n + 1):
        v = lo + k*step
        s += line(MX(v), y - 6, MX(v), y + 6, col=GRAY, w=1.1)
    s += text(MX(0), y + 26, "0", size=10.5, col=MUTE, family=MONO)
    s += text(MX(-1), y + 26, "−max", size=10.5, col=MUTE, family=MONO)
    s += text(MX(1), y + 26, "+max", size=10.5, col=MUTE, family=MONO)
    # weights, snapped
    weights = [0.312, -0.891, 0.544, -0.127]
    for w in weights:
        snapped = round(w/step)*step
        s += circle(MX(w), y - 34, 3.6, fill="none", stroke=CHARCOAL, sw=1.6)   # original
        s += text(MX(w), y - 48, f"{w:+.3f}", size=9.5, col=CHARCOAL, family=MONO)
        s += arrow(MX(w), y - 30, MX(snapped), y - 8, col=RED, marker="red", w=1.4)
        s += circle(MX(snapped), y - 4, 3.6, fill=RED)                          # snapped
    s += text(x0, H - 22, "hollow = original float    ● = nearest grid value    the small gap is the rounding error",
              size=10.5, col=MUTE, anchor="start", style="italic")
    s += text(x0, H - 8, "(grid shown coarse for clarity; int8 provides 256 steps, int4 only 16)",
              size=10.5, col=MUTE, anchor="start", style="italic")
    s += footer()
    write("fig-quantization.svg", s)


ALL = [fig_2_1, fig_2_2, fig_2_3, fig_2_4, fig_2_5, fig_2_6,
       fig_2_7, fig_2_8, fig_2_9, fig_2_10, fig_2_11]


def main():
    for fn in ALL:
        fn()
    if "--png" in sys.argv:
        render_previews(OUT)


if __name__ == "__main__":
    main()
