# Contributing to *Inside the Transformer*

This book is free to read and free to improve. Corrections, clarifications, better explanations,
and translations are all welcome.

Before you spend real time on something, please read the short section on contribution terms below.
It exists to keep the book's future options open, and it is the one part of this file that has
consequences.

---

## The easiest way to help

**Open an issue.** If a formula is wrong, a number does not add up, a cross-reference points at the
wrong chapter, or an explanation lost you, say so. You do not need to propose a fix.

This is also the path with no legal strings attached at all. A correction is a fact, and facts are
not subject to copyright. Telling me that the softmax denominator in Chapter 5 is wrong hands me
information, not text, so nothing in the next section applies to you.

For anything larger than a paragraph, open an issue before writing. It saves you from building
something that does not fit the book's plan.

---

## Contribution terms

By submitting a pull request that contains text, figures, or code, you agree to two things:

1. **You license your contribution to everyone** under the same terms as the rest of the book.
   That is CC BY-NC-SA 4.0 for prose and figures, and the MIT License for code. See `LICENSE.md`.
2. **You grant the author** a perpetual, worldwide, non-exclusive, irrevocable, royalty-free right
   to use, reproduce, modify, translate, publish, distribute, relicense, and sublicense your
   contribution under any terms, including commercial terms.

**You keep your copyright.** This is not an assignment, and you remain free to use your own work
anywhere else, for anything, forever.

Here is why the second point is necessary. The book is under a NonCommercial license, and I am
bound by that license with respect to *your* contribution just as anyone else is. Without the
grant, a single contributed paragraph would permanently prevent a future edition on different
terms, such as a printed edition sold at cost or through a publisher. One person's good-faith
improvement would quietly close that door for the whole book. The grant keeps the option open
without taking anything away from you.

If you would rather not agree to this, open an issue describing the change instead. Ideas and
corrections create no rights question, and a well-written issue is often more useful to me than a
patch.

### What you are confirming

When you submit a contribution, you are confirming that it is your own work and that you have the
right to grant the terms above. In particular:

- It is not copied from a book, paper, blog post, or repository without attribution and a
  compatible license.
- It is not covered by an agreement with an employer or client that would give them a claim to it.
- **If you used an AI tool to generate the text, say so in the pull request.** This is not a
  prohibition. Parts of this book were themselves drafted with AI assistance. The reason it matters
  is narrow and practical: in several jurisdictions, purely AI-generated text has no copyright
  owner, which means nobody can grant the license in point 1. I need to know what I am accepting.

---

## House style

The book has a consistent voice and a few hard rules. Matching them makes review fast.

**Language.** American English throughout: normalization, optimizer, behavior, gray, modeling,
analyze. Published titles and proper nouns keep their printed spelling.

**Register.** Write for a reader who is new to the mathematics or is a junior CS student. Prefer
the everyday word wherever it is still accurate. This is a rule about the surrounding prose, not a
ban on technical terms. When a concept genuinely needs the correct term, use it and explain what it
means. Never swap in a simpler word that is less accurate.

**Math notation.** Inline math is plain Unicode: `√d_k`, `Σᵢ`, `π_θ`, `‖x‖`. Use LaTeX in `$$…$$`
only for display equations where Unicode is genuinely worse, meaning stacked fractions, bracketed
matrices, and multi-line derivations. Worked numeric examples stay in fenced code blocks, because
the alignment is the point.

**Figures.** Every diagram is a static SVG produced by a generator script. Edit
`tools/figures/gen_ch0N_figures.py` and regenerate. **Never hand-edit a file in
`assets/figures/`.** The generated coordinates are what keeps the pictures consistent with the
worked examples.

**Punctuation.** Em-dashes are used sparingly and the build enforces a budget. Semicolons joining
two independent clauses should be periods instead. En-dashes in numeric ranges (`2–3`, `10–20`) are
deliberate, so leave them alone.

**Headings.** One H1 per file, which is the chapter title. Sections are H2, sub-sections H3. Avoid
H4 and deeper.

**Numbers.** Every worked example must be arithmetically correct. Check it in Python before you
submit, and say in the pull request that you did.

**Do not edit build artifacts.** Anything under `dist/` is generated. So is anything in
`assets/figures/`.

---

## Before you open a pull request

Run the gates. They are fast, and a failing gate blocks the build:

```bash
make check                                  # prose and structure gates, all files
python3 tools/check-prose.py <your-files>    # just the files you touched
```

If you changed a chapter, rebuild and look at the result:

```bash
make pdf
```

The prose gate enforces the punctuation and spelling rules above, along with a few defects that are
invisible in Markdown but show up in the PDF. The structure gate checks figure and table numbering,
captions, and cross-references. Word-choice suggestions are advisory and need your judgment.

The prose gate needs a scanner that is not bundled with this repository. Without it the gate
reports `PROSE GATE SKIPPED` and the build continues, so you can clone and build the book without
installing anything. To run the style checks locally:

```bash
git clone https://github.com/tarvanitis/ai-prose-tools
ln -s "$PWD/ai-prose-tools/prose-linter" ~/.claude/skills/prose-linter
```

The structure gate has no dependencies and always runs.

---

## Translations

Translations are explicitly allowed by the license and very welcome. Please open an issue first so
that two people do not translate the same Part in parallel.

A translation is a substantial piece of original expression, so the contribution terms above matter
more here than anywhere else. If you plan to translate a significant portion of the book, let us
agree the terms in the issue before you start rather than at the end.

---

## Out of scope

- Rewriting the book for a different audience, such as a research-level or executive-level version.
- Converting the notation style, for example moving everything to LaTeX.
- Adding coverage of every new model or framework. Chapter 1 carries a dated snapshot on purpose,
  and the rest of the book deliberately teaches the durable mechanics instead.
- Expanding the scope beyond transformers and language models.

---

## Credit

Contributors are acknowledged in the book. If you would prefer not to be named, say so in your pull
request and I will leave you out.
