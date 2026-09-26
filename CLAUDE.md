# Inside the Transformer: Project Guide

## Purpose

*Inside the Transformer: From Attention to Reasoning* is a free book on how transformers and large
language models work. This repository is the manuscript, its build toolchain, and the skill used to
write and maintain it.

## Source of Truth

The Markdown files are canonical:

| Path | Scope |
|---|---|
| `frontmatter/` | Cover, copyright, preface, conceptual track, about the cover |
| `part-1/` … `part-5/` | The 20 chapters, one file each |
| `backmatter/` | Glossary, model specifications appendix, bibliography |

`dist/` holds build artifacts. Never edit them, and never commit them. The released PDF is attached
to a GitHub Release.

## Content Conventions

### Math

Hybrid notation. The primary deliverable is the PDF, with display equations pre-rendered to SVG by
MathJax.

- **Inline notation: plain Unicode.** `√d_k`, `Σᵢ`, `π_θ`, `‖x‖`, `∝`, `≈`. This keeps prose
  readable and diffs clean.
- **Display equations: LaTeX in `$$…$$`, only where Unicode is genuinely worse.** Stacked
  fractions, bracketed matrices, multi-line derivations. Example:
  `$$\mathrm{softmax}(z_i)=\dfrac{e^{z_i}}{\sum_j e^{z_j}}$$`
- **Worked numeric examples stay in fenced code blocks.** Monospace, step by step. Do not convert
  these to LaTeX. The alignment is the point.

See `.claude/skills/technical-book/references/style-guide.md` for the full rule set.

### Figures

Every diagram is a static SVG, hand-authored through a generator script so that coordinates and
plotted values stay exact and match the worked examples. This covers both geometry (vectors, angles,
curves, bar charts) and flowcharts (architecture stacks, pipelines, loops).

- Generators: `tools/figures/gen_ch0N_figures.py`, one per chapter, all sharing the drawing toolkit
  in `tools/figures/svgkit.py`. Output: `assets/figures/<chapter>/*.svg` with content-based
  filenames, so the `Figure N.k` number lives only in the caption and never drifts.
- **Edit the generator, never the SVG by hand**, then regenerate.
- Palette matches `assets/classic-book.css`: dark-red `#A80036` accent, cream `#FEFECE` and
  warm-paper fills, serif labels, monospace for numbers and vectors. Dark-red marks the thing to
  look at. Gray is context.
- Prerequisites: Python 3.10+ (stdlib only to write the SVGs). `cairosvg` is optional, for `--png`
  previews. WeasyPrint renders the SVGs into the PDF.
- Embed as a `::: {.figure}` block holding the image, followed by a `::: {.caption}` block carrying
  `**Figure N.k.**`.
- The cover is generated the same way, by `tools/figures/gen_cover.py`, which writes
  `assets/cover/cover.svg`. It needs `Pillow` and a Times-metric serif font.

### Headings

One H1 per file, the chapter title. Sections are H2, sub-sections H3. Avoid H4 and deeper unless
genuinely necessary.

### Model Version Callouts

Use a dated blockquote:

```
> **Current model families (as of YYYY-MM-DD):** ...
```

Verify version numbers from vendor release pages. Do not rely on memory. Update the date when
versions change.

## Quality Standards

- All numerical worked examples must be arithmetically verified. Check in Python before committing.
- No dead widget stubs (interactive elements flattened to disconnected numbers or labels).
- Clarity target: technically accurate and accessible without a strong math background. Prefer plain
  English analogies for non-obvious concepts, especially before formal definitions.
- Plain language by default. Write for a reader new to mathematics or a junior CS student. Prefer
  the everyday word wherever it is still accurate. This is a register rule for the surrounding
  prose, not a ban on technical terms. When a concept genuinely needs a technical term
  (perpendicular, gradient, softmax), use the correct term and explain what it means. Never swap in
  a simpler but less accurate word. Figure and table captions follow the same rule.
- Do not add error handling, abstractions, or content beyond what the task requires.
- **Preserve the author's terminology.** Change a term only when it is factually wrong in context.
  To aid clarity, add a gloss in parentheses rather than swapping the original term for a synonym.

## Scope Boundaries

- Term definitions belong in `backmatter/appendix-a-glossary.md`. Chapters use terms but should not
  redefine them.
- The mathematics building blocks belong in Chapter 2. Later chapters reference them, they do not
  re-derive them.

## Language and Spelling

**American English throughout.** `-ize`/`-ization`, `-or`, `-er`, single `l` before a suffix:
normalization, optimizer, tokenizer, quantization, behavior, color, center, modeling, labeled,
analyze, license, defense, toward, gray.

Two exceptions, both enforced by the gate:

- **Published titles and proper nouns keep their printed spelling.** The paper *Layer
  Normalization* is quoted, not written. In `backmatter/bibliography.md` a title is a line that is
  nothing but a bold span, which is how the checker recognizes it.
- **Code is out of scope**: identifiers, APIs, string literals (`normalize()`, `chunk_size`).

Beware naive `-ise` to `-ize` substitution. It mangles words spelled the same in both dialects:
*advertising*, *surprising*, *unsupervised*, *pairwise*, *expertise*. Match whole words, never
prefixes.

## Style Notes

- En-dashes (`–`) in numeric ranges (`2–3`, `10–20`) are deliberate. Do not replace them.
- Em-dashes (`—`) are used *sparingly*: at most one pair per paragraph, and not in every paragraph.
  High em-dash density is one of the clearest signals of AI-generated prose, and the gate enforces a
  budget of one per 500 words.
- Semicolons joining two independent clauses, where a period would work, should be periods. The only
  legitimate prose use of a semicolon is as a higher-level separator in comma-lists
  (`Paris, France; London, UK`).

## Build Commands

```bash
make pdf        # both gates, then dist/inside-the-transformer.pdf
make check      # gates only
python3 tools/check-prose.py <files>
python3 tools/check-structure.py <files>
```

## The prose gate is mechanical — do not self-assess

`make` runs two gates before it will build a PDF.

**One implementation, one place.** Every prose rule lives in the prose-linter skill's
`scripts/scan.py`. `tools/check-prose.py` is a thin wrapper that calls `scan.py --gate` and supplies
project policy only: which files, the em-dash budget, and the exemption list. **Never reimplement a
check in the wrapper.** Two implementations drift, and the drift is silent. That is how a green tick
once hid 460 em-dashes.

- `scan.py --gate` enforces B9 (em-dash budget), B18 (clause-join semicolons), B17b (spelling
  dialect), and the render-only defects invisible in Markdown: lists missing a preceding blank line,
  hyphenated words split across lines. Word-choice findings stay advisory.
- `tools/check-structure.py` checks figure and table numbering, uncaptioned figures,
  cross-reference resolution, stale dated claims, and bibliography hard breaks. It is a
  byte-identical copy of the checker published as `book-auditor` in the same repository, vendored
  on purpose. Copying rather than wrapping keeps this gate dependency-free, so a fresh clone is
  still checked. The cost is two copies, so **a fix to either one has to be applied to both.**

**Rules for this gate, learned the hard way:**

- **The only exemption is fenced code, mermaid, and LaTeX math.** There is no "structural"
  category. Glossary entries, headings, bullet labels and table cells all count.
- **Every other exemption must be written into `tools/prose-allow.txt`** as `path:line  # reason`,
  one line at a time, each individually read and justified. An exemption that is not enumerated does
  not exist.
- **Do not write a new measurement script and trust it over the gate.** If a count looks wrong, read
  the lines the gate prints.
- **Book-specific conventions go in `book-audit.toml`, never in the script.** The structure checker
  infers conventions from the manuscript and reports what deviates. Where this book deliberately
  differs, declare it. `xref_styles = ["Chapter N", "Ch N"]` records that the abbreviated form in
  glossary entries and narrow table cells is intentional. Declaring keeps the check alive for
  genuine drift. Editing the script to silence it does not.

**The scanner is an optional dependency and is not bundled with this repository.** It lives in the
prose-linter skill (https://github.com/tarvanitis/ai-prose-tools). Without it,
`tools/check-prose.py` reports `PROSE GATE SKIPPED` and the build
continues, so that cloning and building the book does not require installing a skill. The structure
gate always runs.
