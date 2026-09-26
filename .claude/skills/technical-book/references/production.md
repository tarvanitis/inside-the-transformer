# Production: Assembly & Rendering

How to assemble the chapter files into one manuscript and render publishable deliverables, reusing
this project's existing tooling.

## Prerequisites (toolchain)

Verified present and working (2026-09-12): **pandoc 3.1.3**, **WeasyPrint 69.0**, **Node 18** +
**mathjax-full**, **PyMuPDF 1.28**, **TeX Gyre** fonts, **Noto Color Emoji**.

Two rendering paths, two math strategies:

- **PDF (professional review copy):** `build-pdf.py` renders ` ```mermaid ` fences to **PNG via
  mermaid-cli (Chromium, custom theme)**, runs `pandoc --mathjax --wrap=none`, pre-renders LaTeX
  `$$…$$` equations to **SVG via MathJax (Node)**, then **WeasyPrint** + `assets/classic-book.css`.
  Two things WeasyPrint can't do itself, both pre-rendered here: its native MathML doesn't stack
  fractions/radicals/matrices (and leaks raw LaTeX), and it can't run Mermaid's JS. Mermaid is
  rendered to **PNG, not SVG** — Chromium rasterises its HTML labels correctly, whereas SVG-text mode
  mangles multi-word labels. (`--wrap=none` matters: otherwise pandoc inserts newlines inside tags and
  the equation-extraction regex misses spans.)
- **Word (optional, not the released deliverable):** `pandoc … -o out.docx` converts `$$…$$`
  straight to native OMML Word equations, so MathJax is not needed on that path.

**Required**
- **pandoc** ≥ 3.
- **WeasyPrint** ≥ 66 (`pip install weasyprint`) — HTML+CSS → PDF.
- **Node** ≥ 18 + **mathjax-full** (`cd tools/math && npm install mathjax-full`) — LaTeX → SVG
  for the PDF path, via `tools/math/tex2svg.js`.
- **mermaid-cli** (`@mermaid-js/mermaid-cli`, gives `mmdc`) + a headless **Chromium** (auto-fetched by
  its puppeteer dependency) — renders ` ```mermaid ` fences to PNG with a themed config
  (`assets/mermaid-config.json`, which sets `flowchart.curve` etc.; keep mermaid's default
  `htmlLabels: true` since PNG rendering handles them). Installed in `tools/math`; needs
  `tools/math/puppeteer-config.json` =
  `{"args":["--no-sandbox","--disable-gpu","--disable-dev-shm-usage"]}` for Chromium to launch headless.
- A **serif book font** — `sudo apt-get install fonts-texgyre` (TeX Gyre Termes ≈ Times, Pagella ≈
  Palatino, plus the Courier-like *Cursor* mono); auto-selected by `classic-book.css`. DejaVu Serif
  is the fallback.

**Recommended**
- **PyMuPDF** (`pip install pymupdf`) — rasterize the PDF to PNG to eyeball/verify renders.
- **PyTorch** (`pip install torch`, CPU build is fine) — to *run* code-chapter listings (Parts III–IV),
  not just syntax-check them: extract the ` ```python ` fences, build the model on a tiny config, and
  smoke-test forward pass → one AdamW step (loss drops) → `generate()`. Code must execute, not just parse.
- **Noto Color Emoji** (`sudo apt-get install fonts-noto-color-emoji`) if the 🔑 / 🔬 markers stay;
  otherwise swap them for styled text labels (see `style-guide.md`).

PDF build: `python3 build-pdf.py OUT.pdf FILE1.md [FILE2.md …]`.

## How this book is assembled

There is no separate merge step, and no `combine.py`. The `Makefile` holds the file order and hands
it to `build-pdf.py`, which concatenates and renders in one pass:

```
make pdf  ->  python3 build-pdf.py dist/inside-the-transformer.pdf \
                frontmatter/*.md part-*/ch-*.md backmatter/*.md
```

Chapter H1s are preserved rather than demoted. The printed table of contents is generated inside
`build-pdf.py`: `extract_headings()` walks the pandoc HTML for h1 and h2 elements before the math is
converted to SVG, so heading text stays plain, and `build_toc()` lays it out in three phases, front
matter, chapters with sections, then back matter. It is injected before Chapter 1. The cover is
rendered separately by `build_cover_pdf()` and merged in as page 1, which is why `pypdf` is a build
dependency.

**The file order lives in the Makefile, and nothing discovers files automatically.** That is
deliberate: adding a chapter is a visible edit in one place, and a file that is not listed is
neither built nor gated. Globbing would hide both.

## Rendering deliverables

Draft everything in Markdown (the source of truth), then render:

- **HTML** (screen reading, review): render the Markdown to HTML (use a wide layout for
  the glossary/spec tables).
- **PDF** (the professional review copy): `python3 build-pdf.py OUT.pdf CH…md` — the classic
  technical-book theme (`classic-book.css`) with MathJax-SVG equations. Check that display equations
  render, page breaks fall sensibly at chapter boundaries, worked-example blocks don't overflow the
  page width, and (if kept) the 🔑/🔬 emoji render. Rasterize with PyMuPDF to verify visually.
- **Word** (.docx) — an optional deliverable, not the released one. The PDF is what ships.
    Use **Pandoc** if you need .docx, because the book's hybrid notation
  puts LaTeX in `$$…$$` display equations and only Pandoc turns those into native, editable Word
  equations (OMML). Baseline command:

  ```bash
  # Take the order from the Makefile. Globbing frontmatter/*.md is wrong: it sorts
  # cover.md last and swaps the preface with the cover note.
  pandoc frontmatter/cover.md frontmatter/00-copyright.md \
         frontmatter/02-about-cover.md frontmatter/01-preface.md \
         frontmatter/03-conceptual-track.md \
         part-*/ch-*.md backmatter/*.md \
         -o book.docx --toc --number-sections   # add: --reference-doc=styles.docx
  ```

  Two things to handle for a chapter that contains a Mermaid diagram (Part I chapters 1–2 have none,
  so this isn't blocking the sample): Pandoc doesn't render Mermaid, so **pre-render diagrams to
  PNG/SVG first** (a standalone `mmdc` call does it) and swap the
  ` ```mermaid ` fences for image links before running Pandoc. Then verify figure/table captions and
  numbering survive. Build a `--reference-doc` template once to lock fonts, heading styles, and
  equation styling for the whole book.

## Pre-publication checks

- All worked examples arithmetically verified in Python (CLAUDE.md quality bar).
- No LaTeX artifacts; all math is plain Unicode.
- Every Mermaid diagram validated with Mermaid CLI v11 before it goes in.
- Model-version callouts re-verified against vendor pages and re-dated.
- Cross-reference audit: no dangling `§`, no "this guide", every "see Chapter N"/"Figure N.k"
  resolves.
- TOC, List of Figures, List of Tables, and the index regenerated against the final chapter set.
- Front and back matter present; per-file footers and the bookmark dump gone.
- **Final whole-book humanise consistency pass** — per-chapter humanising already happened during
  drafting (see `chapter-workflow.md`); this last pass catches cross-chapter repetition and voice
  drift. Run the scanner over the assembled manuscript at `--min-count 2`; don't over-sand.

## Note

Do not hand-edit derived build artifacts. The rendered HTML, the PDF and any .docx are outputs, not
sources. Edit the canonical chapter files and rebuild. An edit made in an artifact is lost at the
next build, and until then the two disagree.
