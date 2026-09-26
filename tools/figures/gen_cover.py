from PIL import ImageFont
import math
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "assets" / "cover"

CREAM, CRIMSON, INK, GREY = "#F9F6F0", "#A80036", "#1A1A1A", "#9A958C"
W, H, M = 1200, 1800, 150
COL = W - 2*M
TITLE_STACK = ("Georgia, &apos;Iowan Old Style&apos;, &apos;Palatino Linotype&apos;, "
               "Palatino, &apos;Times New Roman&apos;, Times, serif")
# Times-first for the sentence: Liberation Serif (used for measuring) shares its metrics,
# so the arcs land on the right words in the reader's renderer too.
SENT_STACK = "&apos;Times New Roman&apos;, Times, &apos;Liberation Serif&apos;, serif"
# The measuring font must share Times metrics, or the arcs land on the wrong words.
# Liberation Serif is metric-compatible with Times New Roman. A serif font with
# different metrics (DejaVu Serif, for instance) will silently misplace every arc.
TTF_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",   # Debian, Ubuntu
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/liberation-serif/LiberationSerif-Regular.ttf",      # Fedora
    "/usr/share/fonts/TTF/LiberationSerif-Regular.ttf",                   # Arch
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",             # macOS
    "/Library/Fonts/Times New Roman.ttf",
]


def find_ttf():
    """Locate a Times-metric serif font, or explain how to install one."""
    override = os.environ.get("COVER_TTF")
    if override:
        if not pathlib.Path(override).exists():
            raise SystemExit(f"COVER_TTF points at {override}, which does not exist.")
        return override
    for candidate in TTF_CANDIDATES:
        if pathlib.Path(candidate).exists():
            return candidate
    raise SystemExit(
        "No Times-metric serif font found.\n"
        "The arc positions are measured with this font, so a substitute with different\n"
        "metrics would draw the arcs over the wrong words. Install Liberation Serif:\n"
        "  Debian/Ubuntu:  sudo apt install fonts-liberation\n"
        "  Fedora:         sudo dnf install liberation-serif-fonts\n"
        "  Arch:           sudo pacman -S ttf-liberation\n"
        "Or point COVER_TTF at a metric-compatible font:\n"
        "  COVER_TTF=/path/to/LiberationSerif-Regular.ttf python3 tools/figures/gen_cover.py"
    )


TTF = None   # resolved in main(), so importing this module never fails

WORDS = ["Machines", "learned", "to", "read", "us.", "Now", "they", "answer."]
FOCUS = 7                                    # the word doing the attending

# one head's weights for the focus word, hand-set to fit the line. Sums to 1.
ALPHA = {0: 0.20, 1: 0.09, 2: 0.05, 3: 0.22, 4: 0.13, 5: 0.08, 6: 0.15, 7: 0.08}

BASE = 1438                                  # baseline of the sentence


def layout():
    probe = 100
    f = ImageFont.truetype(TTF, probe)
    widths = [f.getlength(w) for w in WORDS]
    space = f.getlength(" ")
    total = sum(widths) + space*(len(WORDS)-1)
    size = probe * (COL / total)
    f = ImageFont.truetype(TTF, round(size))
    widths = [f.getlength(w) for w in WORDS]
    space = f.getlength(" ")
    scale = (COL - sum(widths)) / (len(WORDS)-1)      # justify to the full column
    xs, x = [], float(M)
    for w in widths:
        xs.append((x, x + w/2, x + w))
        x += w + scale
    return round(size), xs


def arc(x1, x2, y, width, opacity):
    span = abs(x2 - x1)
    h = min(span*1.42, 1050)
    return (f'<path d="M {x1:.1f} {y:.1f} C {x1:.1f} {y-h:.1f}, {x2:.1f} {y-h:.1f}, '
            f'{x2:.1f} {y:.1f}" fill="none" stroke="{CRIMSON}" stroke-width="{width:.2f}" '
            f'stroke-linecap="round" opacity="{opacity:.2f}"/>')


def build():
    size, xs = layout()
    top = BASE - size*0.72                   # arcs spring from the top of the letters
    mx = max(ALPHA.values())
    p, add = [], lambda s: p.append(s)

    add(f'<rect width="{W}" height="{H}" fill="{CREAM}"/>')
    add(f'<g font-family="{TITLE_STACK}">')
    add(f'<text x="{M}" y="252" font-size="118" fill="{INK}">Inside the</text>')
    add(f'<text x="{M}" y="382" font-size="118" fill="{INK}">Transformer</text>')
    add(f'<rect x="{M}" y="424" width="{COL}" height="4" fill="{CRIMSON}"/>')
    add(f'<text x="{M}" y="487" font-size="30" font-weight="bold" letter-spacing="4.6" '
        f'fill="{CRIMSON}">FROM ATTENTION TO REASONING</text>')
    add('</g>')

    fx = xs[FOCUS][1]
    for j in sorted(ALPHA, key=lambda j: ALPHA[j]):
        if j == FOCUS:
            continue
        a = ALPHA[j]
        add(arc(xs[j][1], fx, top, 1.0 + 11.0*a, 0.30 + 0.70*a/mx))
    # self-attention: a short arch over the word itself, weighted like the rest of the fan
    a = ALPHA[FOCUS]
    add(f'<path d="M {fx-16:.1f} {top:.1f} C {fx-16:.1f} {top-52:.1f}, {fx+16:.1f} '
        f'{top-52:.1f}, {fx+16:.1f} {top:.1f}" fill="none" stroke="{CRIMSON}" '
        f'stroke-width="{1.0+11.0*a:.2f}" stroke-linecap="round" '
        f'opacity="{0.30+0.70*a/mx:.2f}"/>')

    add(f'<g font-family="{SENT_STACK}" font-size="{size}">')
    for i, word in enumerate(WORDS):
        if i == FOCUS:
            add(f'<text x="{xs[i][0]:.1f}" y="{BASE}" fill="{CRIMSON}" '
                f'font-style="italic">{word}</text>')
        else:
            a = ALPHA.get(i, 0.0)
            lift = min(1.0, a/mx)
            g = round(165 - 139*lift)        # darker the more it is attended to
            add(f'<text x="{xs[i][0]:.1f}" y="{BASE}" fill="rgb({g},{g-4},{g-10})">{word}</text>')
    add('</g>')

    add(f'<text x="{M}" y="1528" font-family="{TITLE_STACK}" font-size="19" '
        f'letter-spacing="3.2" fill="{GREY}">WHAT THE ANSWER LOOKS BACK AT</text>')
    add(f'<text x="{M}" y="1698" font-family="{TITLE_STACK}" font-size="30" '
        f'letter-spacing="4.4" fill="{INK}">ATHANASIOS ARVANITIS</text>')

    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
            f'height="{H}">\n  ' + "\n  ".join(p) + "\n</svg>\n"), size, xs, top


if __name__ == "__main__":
    TTF = find_ttf()
    svg, size, xs, top = build()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "cover.svg"
    out.write_text(svg)
    print("wrote", out)
    print("font size", size, "| alpha sum", round(sum(ALPHA.values()), 6),
          "| arc springline", round(top))
