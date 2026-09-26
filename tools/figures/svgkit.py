#!/usr/bin/env python3
"""Shared SVG toolkit for the book's static geometry/coordinate figures.

Palette and helpers used by the per-chapter figure generators
(gen_ch02_figures.py, gen_ch03_figures.py, gen_ch04_figures.py). Keeping this in
one place means every figure looks the same and coordinates stay exact.

Palette matches assets/classic-book.css: dark-red #A80036 accent, cream /
warm-paper fills, serif labels, monospace for numbers and vectors. Convention:
dark-red = the thing to look at; gray/charcoal = context.
"""
import math
import pathlib

# --- palette (from assets/classic-book.css + mermaid-config.json) ------------
RED      = "#A80036"   # focus / accent
INK      = "#1b1b1b"   # primary text
CHARCOAL = "#4a4a4a"   # secondary vector / secondary text
GRAY     = "#8a8a8a"   # axes
GRID     = "#d7d2c8"   # gridlines
CREAM    = "#FEFECE"
PAPER    = "#faf6f1"
NOTE     = "#FBFB77"
MUTE     = "#6b6b6b"
BORDER   = "#d9d4cc"
REDFILL  = "#f3d9e0"   # light tint of the accent for bars
GRAYFILL = "#e9e6df"
BLUEFILL = "#dfe7f0"   # cool tint, for a second data series when one is needed

SERIF = "'TeX Gyre Termes','Liberation Serif','DejaVu Serif',Georgia,serif"
MONO  = "'TeX Gyre Cursor','Courier New','DejaVu Sans Mono',monospace"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def header(w, h):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" font-family="{SERIF}">\n'
        + defs()
    )

def defs():
    m = ['<defs>']
    for name, col in (("red", RED), ("ink", INK), ("char", CHARCOAL), ("gray", GRAY)):
        m.append(
            f'<marker id="a-{name}" viewBox="0 0 10 10" refX="8.5" refY="5" '
            f'markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>'
        )
    m.append('</defs>\n')
    return "".join(m)

def footer():
    return '</svg>\n'

def line(x1, y1, x2, y2, col=GRAY, w=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{col}" stroke-width="{w}"{d}/>\n')

def arrow(x1, y1, x2, y2, col=INK, marker="ink", w=2.2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{col}" stroke-width="{w}" marker-end="url(#a-{marker})"{d}/>\n')

def text(x, y, s, size=13, col=INK, anchor="middle", family=SERIF,
         weight="normal", style="normal", baseline="middle"):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{col}" '
            f'text-anchor="{anchor}" font-family="{family}" font-weight="{weight}" '
            f'font-style="{style}" dominant-baseline="{baseline}">{esc(s)}</text>\n')

def circle(x, y, r, fill=RED, stroke="none", sw=0):
    return (f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>\n')

def rect(x, y, w, h, fill="none", stroke=BORDER, sw=1.0, rx=0):
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n')

def panel(x, y, w, h, title=None):
    s = rect(x, y, w, h, fill=PAPER, stroke=BORDER, sw=1.0, rx=6)
    if title:
        s += text(x + w / 2, y + 16, title, size=12.5, col=CHARCOAL, weight="bold")
    return s

def arc(cx, cy, r, a1, a2, col=GRAY, w=1.3):
    """Arc from math-angle a1 to a2 (degrees, CCW from +x, y-up)."""
    x1 = cx + r * math.cos(math.radians(a1)); y1 = cy - r * math.sin(math.radians(a1))
    x2 = cx + r * math.cos(math.radians(a2)); y2 = cy - r * math.sin(math.radians(a2))
    large = 1 if abs(a2 - a1) > 180 else 0
    sweep = 0 if a2 > a1 else 1   # y-down flips sweep sense
    return (f'<path d="M{x1:.2f},{y1:.2f} A{r},{r} 0 {large} {sweep} {x2:.2f},{y2:.2f}" '
            f'fill="none" stroke="{col}" stroke-width="{w}"/>\n')

def right_angle(cx, cy, u, v, size=13, col=GRAY):
    """Small square marking a right angle between unit dirs u,v (each (dx,dy) y-up)."""
    ax, ay = cx + u[0]*size, cy - u[1]*size
    bx, by = cx + v[0]*size, cy - v[1]*size
    dx, dy = (u[0]+v[0])*size, (u[1]+v[1])*size
    return (f'<path d="M{ax:.2f},{ay:.2f} L{cx+dx:.2f},{cy-dy:.2f} L{bx:.2f},{by:.2f}" '
            f'fill="none" stroke="{col}" stroke-width="1.2"/>\n')

class Frame:
    """Map math coords (x right, y up) to SVG pixels."""
    def __init__(self, ox, oy, sx, sy): self.ox, self.oy, self.sx, self.sy = ox, oy, sx, sy
    def X(self, x): return self.ox + x * self.sx
    def Y(self, y): return self.oy - y * self.sy

def node(cx, cy, w, h, lines, fill=PAPER, stroke=CHARCOAL, tc=INK, size=10.5, rx=6, sw=1.2,
         lh=13, family=SERIF, weight="normal"):
    """A rounded-rect flowchart node centered at (cx,cy) with one or more centered text lines."""
    if isinstance(lines, str):
        lines = [lines]
    out = rect(cx - w/2, cy - h/2, w, h, fill=fill, stroke=stroke, sw=sw, rx=rx)
    y0 = cy - (len(lines) - 1)*lh/2
    for i, ln in enumerate(lines):
        out += text(cx, y0 + i*lh, ln, size=size, col=tc, family=family, weight=weight)
    return out

def diamond(cx, cy, w, h, label, fill=CREAM, stroke=RED, tc=INK, size=10):
    """A decision (diamond) node centered at (cx,cy)."""
    pts = (f"{cx:.1f},{cy-h/2:.1f} {cx+w/2:.1f},{cy:.1f} "
           f"{cx:.1f},{cy+h/2:.1f} {cx-w/2:.1f},{cy:.1f}")
    return (f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>\n'
            + text(cx, cy, label, size=size, col=tc))

def elabel(x, y, s, col=MUTE, size=8.5):
    """A small italic edge label."""
    return text(x, y, s, size=size, col=col, style="italic")

def mix(c1, c2, t):
    """Linear blend between two #rrggbb colors; t in [0,1]."""
    t = max(0.0, min(1.0, t))
    a = [int(c1[i:i+2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i+2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(a[i] + (b[i]-a[i])*t)) for i in range(3))

def save(outdir, name, svg):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / name).write_text(svg, encoding="utf-8")
    print("wrote", name)

def render_previews(outdir, scale=2.0):
    """Render every SVG in outdir to a throwaway PNG in outdir/_preview (needs cairosvg)."""
    import cairosvg
    outdir = pathlib.Path(outdir)
    pngdir = outdir / "_preview"; pngdir.mkdir(exist_ok=True)
    for f in sorted(outdir.glob("*.svg")):
        cairosvg.svg2png(url=str(f), write_to=str(pngdir / (f.stem + ".png")), scale=scale)
    print("rendered PNG previews to", pngdir)
