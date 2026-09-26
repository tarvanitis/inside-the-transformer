#!/usr/bin/env python3
"""Build a classic-technical-book PDF for *Inside the Transformer*.

Pipeline:
  concat Markdown
    → render ```mermaid``` fences to PNG via mermaid-cli (mmdc, themed)          [Chromium]
    → pandoc (--mathjax --wrap=none) to HTML
    → pre-render LaTeX display math to SVG via MathJax (tools/math/tex2svg.js)
    → WeasyPrint + assets/classic-book.css

Why each step: WeasyPrint can't run JS, so both math (MathJax→SVG) and diagrams
(mermaid→PNG) are pre-rendered at build time. Mermaid is rendered to PNG (not SVG)
because Chromium rasterises its HTML labels correctly, which SVG-text mode mangles.

Usage:
    python3 build-pdf.py OUT.pdf FILE1.md [FILE2.md ...]
"""
import hashlib
import html
import io
import json
import pathlib
import re
import subprocess
import sys
from html.parser import HTMLParser

import weasyprint

HERE = pathlib.Path(__file__).parent
CSS = HERE / "assets" / "classic-book.css"
COVER_CSS = HERE / "assets" / "cover-page.css"
TITLE = "Inside the Transformer: From Attention to Reasoning"
AUTHOR = "Athanasios Arvanitis"
SUBJECT = "How transformers and large language models work"
KEYWORDS = ("transformers, large language models, LLM, attention, deep learning, "
            "machine learning, neural networks, PyTorch")
TEX2SVG = HERE / "tools" / "math" / "tex2svg.js"
MMDC = HERE / "tools" / "math" / "node_modules" / ".bin" / "mmdc"
MERMAID_CFG = HERE / "assets" / "mermaid-config.json"
PUPPETEER_CFG = HERE / "tools" / "math" / "puppeteer-config.json"
DIAGRAMS = HERE / "dist" / "diagrams"

MERMAID_RE = re.compile(r"```mermaid\n(.*?)\n```", re.S)
DISPLAY_RE = re.compile(r'<span\s+class="math display">\s*\\\[(.*?)\\\]</span>', re.S)
INLINE_RE = re.compile(r'<span\s+class="math inline">\s*\\\((.*?)\\\)</span>', re.S)


def render_mermaid(md):
    """Replace each ```mermaid``` fence with a PNG rendered by mmdc (cached by content hash)."""
    DIAGRAMS.mkdir(parents=True, exist_ok=True)

    def repl(m):
        code = m.group(1)
        h = hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]
        png = DIAGRAMS / f"diag_{h}.png"
        if not png.exists():
            src = DIAGRAMS / f"diag_{h}.mmd"
            src.write_text(code, encoding="utf-8")
            subprocess.run(
                [str(MMDC), "-i", str(src), "-o", str(png), "-c", str(MERMAID_CFG),
                 "-p", str(PUPPETEER_CFG), "-b", "white", "-w", "1800"],
                check=True, capture_output=True, text=True,
            )
        return f'\n\n<div class="mermaid-fig"><img src="{png}" /></div>\n\n'

    return MERMAID_RE.sub(repl, md)


def md_to_html(md):
    return subprocess.run(
        ["pandoc", "-", "--mathjax", "--wrap=none", "-t", "html5"],
        input=md, capture_output=True, text=True, check=True,
    ).stdout


def tex_to_svg(items):
    if not items:
        return []
    proc = subprocess.run(
        ["node", str(TEX2SVG)], input=json.dumps(items),
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


def render_math(doc_html):
    disp = [{"tex": html.unescape(m), "display": True} for m in DISPLAY_RE.findall(doc_html)]
    inl = [{"tex": html.unescape(m), "display": False} for m in INLINE_RE.findall(doc_html)]
    dsvg, isvg = iter(tex_to_svg(disp)), iter(tex_to_svg(inl))
    doc_html = DISPLAY_RE.sub(lambda _: f'<div class="mathblock">{next(dsvg)}</div>', doc_html)
    doc_html = INLINE_RE.sub(lambda _: f'<span class="mathinline">{next(isvg)}</span>', doc_html)
    return doc_html


def extract_headings(doc_html):
    """Walk the pandoc-generated HTML and return h1/h2 entries for TOC generation.
    Each entry: {tag, id, text, frontmatter}.
    frontmatter=True when the heading is inside a div.frontmatter.
    Extraction happens before render_math so heading text is plain (no SVG).
    """
    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.entries = []
            self._div_stack = []      # (class_set, element_id) for open divs/sections
            self._h = None            # (tag, id, in_frontmatter)
            self._buf = []
            self._skip_span = False   # suppress math spans in headings

        @property
        def _in_fm(self):
            return any("frontmatter" in classes for classes, _ in self._div_stack)

        def _nearest_section_id(self):
            for classes, sec_id in reversed(self._div_stack):
                if sec_id:
                    return sec_id
            return ""

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag in ("div", "section"):
                self._div_stack.append((set(a.get("class", "").split()), a.get("id", "")))
            elif tag in ("h1", "h2") and self._h is None:
                h_id = a.get("id", "")
                # frontmatter h1s have no id on the element; pandoc puts it on <section>
                if not h_id and self._in_fm:
                    h_id = self._nearest_section_id()
                self._h = (tag, h_id, self._in_fm)
                self._buf = []
            elif tag == "span" and self._h:
                if set(a.get("class", "").split()) & {"math", "math-inline", "math-display",
                                                       "math-inline-left"}:
                    self._skip_span = True

        def handle_endtag(self, tag):
            if tag in ("div", "section") and self._div_stack:
                self._div_stack.pop()
            elif tag == "span":
                self._skip_span = False
            elif self._h and tag == self._h[0]:
                h_tag, h_id, is_fm = self._h
                text = "".join(self._buf).strip()
                if text:
                    self.entries.append(dict(tag=h_tag, id=h_id,
                                             text=text, frontmatter=is_fm))
                self._h = None
                self._buf = []

        def handle_data(self, data):
            if self._h and not self._skip_span:
                self._buf.append(data)

    p = _Parser()
    p.feed(doc_html)
    return p.entries


def build_toc(entries):
    """Return (toc_html, first_chapter_id).
    toc_html: a <div class="frontmatter toc-page"> ready to inject before Chapter 1.
    first_chapter_id: the pandoc-generated id of the first non-frontmatter h1.

    Three-phase layout:
      front  — frontmatter h1s before the first chapter (Preface, excl. title page)
      middle — non-frontmatter h1/h2 (Chapter 1-20 and their sections)
      back   — frontmatter h1s after the last chapter (Appendices, Bibliography)
    """
    # Locate the index boundaries for each phase.
    ch_indices = [i for i, e in enumerate(entries)
                  if not e["frontmatter"] and e["tag"] == "h1"]
    if not ch_indices:
        return "", None

    first_ch_idx = ch_indices[0]
    last_nonfm_idx = max(i for i, e in enumerate(entries) if not e["frontmatter"])
    first_ch_id = entries[first_ch_idx]["id"]

    front = [e for e in entries[:first_ch_idx]
             if e["tag"] == "h1" and e["text"] != "Inside the Transformer"]
    middle = entries[first_ch_idx: last_nonfm_idx + 1]
    back = [e for e in entries[last_nonfm_idx + 1:]
            if e["frontmatter"] and e["tag"] == "h1"]

    parts = [
        '<div class="frontmatter toc-page">',
        '<h1>Contents</h1>',
        '<nav class="toc">',
    ]

    # ── Front (Preface) ──────────────────────────────────────────────────────
    if front:
        parts.append('<div class="toc-section">')
        for e in front:
            href = f"#{e['id']}" if e["id"] else "#"
            parts.append(
                f'<div class="toc-entry toc-fm">'
                f'<a href="{href}">{html.escape(e["text"])}</a></div>'
            )
        parts.append('</div>')

    # ── Chapters + sections ───────────────────────────────────────────────────
    parts.append('<hr class="toc-sep">')
    parts.append('<div class="toc-section">')
    chapter_num = section_num = 0
    for e in middle:
        href = f"#{e['id']}" if e["id"] else "#"
        escaped = html.escape(e["text"])
        if e["tag"] == "h1":
            chapter_num += 1
            section_num = 0
            parts.append(
                f'<div class="toc-entry toc-ch">'
                f'<a href="{href}">'
                f'<span class="toc-ch-num">Chapter {chapter_num}</span>'
                f' {escaped}</a></div>'
            )
        elif e["tag"] == "h2":
            section_num += 1
            label = f"{chapter_num}.{section_num}"
            parts.append(
                f'<div class="toc-entry toc-sec">'
                f'<a href="{href}">'
                f'<span class="toc-sec-num">{label}</span>'
                f' {escaped}</a></div>'
            )
    parts.append('</div>')

    # ── Back matter (Appendices, Bibliography) ────────────────────────────────
    if back:
        parts.append('<hr class="toc-sep">')
        parts.append('<div class="toc-section">')
        for e in back:
            href = f"#{e['id']}" if e["id"] else "#"
            parts.append(
                f'<div class="toc-entry toc-fm">'
                f'<a href="{href}">{html.escape(e["text"])}</a></div>'
            )
        parts.append('</div>')

    parts += ["</nav>", "</div>"]
    return "\n".join(parts), first_ch_id


def build_cover_pdf(svg_src, base_url):
    """Render a cover SVG as a full-bleed A4 PDF page (zero margins, no header/footer).
    The SVG is embedded inline so we can inject preserveAspectRatio="none" and exact A4
    dimensions directly on the root element, preventing WeasyPrint from letterboxing it.
    Returns a BytesIO containing the one-page PDF.
    """
    import io
    svg_text = (pathlib.Path(base_url) / svg_src).read_text(encoding="utf-8")

    def _fix_svg_root(m):
        tag = m.group(0)
        # Remove any existing width/height/preserveAspectRatio so injected A4
        # values are the only ones present (XML uses the first occurrence).
        tag = re.sub(r'\s+(?:width|height|preserveAspectRatio)="[^"]*"', "", tag,
                     flags=re.IGNORECASE)
        return tag[:-1] + ' width="210mm" height="297mm" preserveAspectRatio="none">'

    svg_text = re.sub(r"<svg\b[^>]*>", _fix_svg_root, svg_text, count=1)
    html_str = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"</head><body>{svg_text}</body></html>"
    )
    buf = io.BytesIO()
    weasyprint.HTML(string=html_str, base_url=base_url).write_pdf(
        buf, stylesheets=[str(COVER_CSS)])
    buf.seek(0)
    return buf


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: build-pdf.py OUT.pdf FILE1.md [FILE2.md ...]")
    out, files = sys.argv[1], sys.argv[2:]
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)

    # If the first file is cover.md, extract its SVG path and handle it separately.
    # The cover.md is still fed into the main pipeline so that the page counter on
    # subsequent pages is correct (About the Cover = page 2, Preface = page 3, …).
    # After the main PDF is built, page 0 (the cover placeholder) is replaced with
    # a clean full-bleed cover rendered by build_cover_pdf().
    cover_pdf_buf = None
    if files and pathlib.Path(files[0]).name == "cover.md":
        cover_md = pathlib.Path(files[0]).read_text(encoding="utf-8")
        m = re.search(r'src="([^"]*\.svg)"', cover_md)
        if m:
            cover_pdf_buf = build_cover_pdf(m.group(1), str(HERE))

    md = "\n\n".join(pathlib.Path(f).read_text(encoding="utf-8") for f in files)
    md = render_mermaid(md)
    raw_html = md_to_html(md)
    # Extract headings before math SVG conversion so text is plain
    headings = extract_headings(raw_html)
    body = render_math(raw_html)
    # Bind each figure to the caption that follows it. They are sibling divs, so
    # `break-before: avoid` on the caption alone does not stop WeasyPrint splitting
    # them across a page; wrapping the pair gives it a single unbreakable box.
    body = re.sub(
        r'(<div class="mermaid-fig">.*?</div>)\s*(<div class="caption">.*?</div>)',
        r'<div class="figblock">\1\2</div>',
        body, flags=re.S)
    # Inject printed TOC before Chapter 1
    toc_html, first_ch_id = build_toc(headings)
    if first_ch_id:
        marker = f'<h1 id="{first_ch_id}"'
        pos = body.find(marker)
        if pos >= 0:
            body = body[:pos] + toc_html + "\n\n" + body[pos:]
    # WeasyPrint reads these into the PDF document-information dictionary:
    # <title> -> /Title, and the meta names below -> /Author, /Subject, /Keywords.
    # Without them a reader's Properties pane shows a blank author.
    page = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(TITLE)}</title>"
        f"<meta name='author' content='{html.escape(AUTHOR)}'>"
        f"<meta name='description' content='{html.escape(SUBJECT)}'>"
        f"<meta name='keywords' content='{html.escape(KEYWORDS)}'>"
        f"</head><body>"
        f"{body}</body></html>"
    )

    if cover_pdf_buf is not None:
        from pypdf import PdfWriter, PdfReader
        # Build the main PDF into memory
        main_buf = io.BytesIO()
        weasyprint.HTML(string=page, base_url=str(HERE)).write_pdf(
            main_buf, stylesheets=[str(CSS)])
        main_buf.seek(0)
        # Merge: cover PDF page 0 replaces main PDF page 0 (the placeholder).
        # Clone the whole document rather than copying pages one by one — the
        # bookmark outline lives in the catalog, not on the pages, so add_page()
        # in a loop would silently drop it. Outline destinations reference page
        # objects, so swapping page 0 leaves the rest of them intact.
        cover_reader = PdfReader(cover_pdf_buf)
        main_reader = PdfReader(main_buf)
        writer = PdfWriter(clone_from=main_reader)
        del writer.pages[0]                         # drop the placeholder
        writer.insert_page(cover_reader.pages[0], 0)   # full-bleed cover
        with open(out, "wb") as f:
            writer.write(f)
    else:
        weasyprint.HTML(string=page, base_url=str(HERE)).write_pdf(out, stylesheets=[str(CSS)])
    print("wrote", out)


if __name__ == "__main__":
    main()
