# Style & Consistency Guide

The source files were authored separately, so they disagree on voice, self-labeling, and numbering.
A book needs one editorial voice. These rules resolve the conflicts; where this guide is silent,
`CLAUDE.md` governs.

## Editorial voice

The target voice already exists in the strongest source chapters (file 02 fundamentals, file 05
training): **warm, precise, and pedagogical** — plain English first, formalism second, concrete
worked numbers throughout. Bring the other chapters up to this register:

- Prefer the direct second person ("you'll see", "notice that") and the inclusive first person plural
  for shared reasoning ("we can now…"). Pick one and keep it consistent within a chapter.
- Explain before you formalise: a plain-English sentence or analogy, then the definition, then the
  formula, then a worked example. This is the source's own order — enforce it everywhere.
- Drop self-labels ("Student Guide", "Complete Terminology Guide"). The book has one title.
- Fix source-quality issues carried over from the HTML originals: typos ("architecures",
  "geneally"), the emoji sign-off in file 06, and the informal register drift in file 04.

## Recurring devices (define once in the preface, use consistently)

- **The three lenses** — Geometric ("the shape of computation"), Representation ("the meaning"),
  Mechanistic ("the machinery"). Keep the exact three names and their meanings uniform. Don't
  introduce a fourth lens or rename them mid-book.
- **Typed callouts (no raw emoji).** Callouts are pandoc fenced divs from a controlled vocabulary,
  each rendered with a small (12pt), vertically-centred monochrome icon (accent red), no enclosing
  badge — see
  `assets/classic-book.css` and `assets/icons/`. Emoji are *not* used in the manuscript;
  they clash with the classic serif theme.
  - `::: {.callout .idea}` — **Key idea / insight** (light bulb).
  - `::: {.callout .plain}` — **Plain English** intuition (speech bubble).
  - `::: {.callout .lens}` — the three lenses; the bold label says which (Geometric / Representation /
    Mechanistic).
  - `::: {.callout .deepdive}` — **technical deep-dive** (magnifier).
  - `::: {.callout .note}` — neutral aside / info (info "i").
  - `::: {.callout .caution}` — pitfall / gotcha (warning triangle).
  **Labels:** drop the redundant lead-in where the icon + preface legend already convey it — `.idea`
  and `.plain` carry **no** label. `.lens` keeps a one-word qualifier (**Geometric.** /
  **Representation.** / **Mechanistic.**), since one icon can't distinguish the three. `.note`,
  `.caution`, and `.deepdive` keep their short descriptive lead (informative, not redundant). Keep the
  set small; incidental one-off asides may stay plain blockquotes (cream box, no icon). The chapter
  bridge ("Coming up:") stays a plain blockquote. The **icon legend lives in the preface**.

## Notation (inherits CLAUDE.md — restated because it's the top consistency risk)

- **Hybrid notation.** The released deliverable is the PDF, where display equations are pre-rendered
  to SVG by MathJax. The same LaTeX also converts to native OMML equations if a .docx is ever needed,
  which is why it is worth writing display math as LaTeX rather than Unicode.
  - *Inline* notation is plain Unicode: `√d_k`, `Σᵢ`, `π_θ`, `‖x‖`, `∝`, `≈`. Keeps prose readable
    and diffs clean.
  - *Display* equations use LaTeX in `$$…$$` **only where Unicode is genuinely worse**: stacked
    fractions (softmax), bracketed matrices, multi-line/aligned derivations. Don't LaTeX-ify things
    Unicode already renders cleanly.
  - *Worked numeric examples* stay in fenced code blocks — never convert step-by-step arithmetic to
    LaTeX; monospace alignment is the point.
  - In LaTeX, prefer `\mathrm{}` for operator names (`\mathrm{softmax}`), `\top` for transpose,
    `\lVert x\rVert` for norms, `\begin{bmatrix}…\end{bmatrix}` for matrices.
- Keep symbol usage identical across chapters: one symbol per concept (e.g. `d_model` vs
  `d_k` used consistently; don't let one chapter write `h` for what another calls `d_model`).
- Real worked numbers should stay internally consistent — the source often uses Llama 3 8B shapes
  (`E ∈ ℝ^{128000×4096}`); keep one running reference model per Part where possible.

## Numbering & captioning

- **Headings:** one H1 per chapter file (the chapter title). Sections H2 (`N.M`), sub-sections H3.
  Avoid H4+ (CLAUDE.md). Parts are inserted at assembly, not as chapter H1s.
- **Figures:** number `Figure N.k` per chapter; every Mermaid diagram and ASCII diagram that a
  reader would refer to gets a caption. Collect into a List of Figures if the edition warrants it.
- **Tables:** `Table N.k` with a caption; collect into a List of Tables if used.
- **Equations:** number displayed equations `(N.k)` only if they're referenced later; don't number
  every formula.
- **Cross-references** use these labels ("as shown in Figure 5.2", "see Chapter 2"), never file
  names or `§`-from-the-old-numbering.

## Model-version callouts

Use the dated blockquote form from CLAUDE.md:

```
> **Current model families (as of YYYY-MM-DD):** …
```

Verify every version number from vendor release pages before publishing — never from memory or
training data. Update the date when refreshed. Keep the "durable shape vs. dated snapshot" framing
the source uses, so the book ages gracefully.

## Consistency passes (run across the whole manuscript, not per chapter)

1. **Terminology:** every term defined once (Appendix A); first chapter mention links to it; no term
   redefined in a chapter.
2. **Voice:** one person/tense per chapter; self-labels gone.
3. **Notation:** one symbol per concept book-wide; hybrid math per the Notation section above
   (Unicode inline, `$$...$$` display only where Unicode is genuinely worse). The zero-LaTeX rule

4. **Numbering:** figures/tables/equations sequential and captioned; all cross-refs resolve.
5. **Dedup:** no foundation taught twice (see `book-architecture.md` dedup plan).
6. **Prose:** `python3 ~/.claude/skills/prose-linter/scripts/scan.py FILE.md --min-count 2`.
   ~36 content tells across 10,600 lines was deemed acceptable for the reference; hold roughly that
   density — don't over-sand the prose.

## Style choices to preserve (do not "fix")

- En-dashes (–) in numeric ranges (`2–3`, `10–20`) are deliberate (CLAUDE.md). Em-dashes (—) are
  **not** exempt: CLAUDE.md asks for them sparingly (at most one pair per paragraph, not in every
  paragraph), and the Stage 4 humanization pass reduces them. Do not treat em-dash density as
  protected style.
- ASCII diagrams and hand-worked arithmetic — they're a feature, keep them.
- The analogies ("keyhole vs. whole page", "GPS coordinates for words", "matrix as a recipe") — keep
  and, where helpful, reuse consistently rather than inventing new ones for the same idea.
