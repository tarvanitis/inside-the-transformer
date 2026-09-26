# Per-Chapter Workflow

The repeatable procedure for turning one source section into one book chapter. Apply it chapter by
chapter; tick the checklist at the end before moving on.

## Procedure

1. **Extract** the relevant section(s) from the source file into the chapter file per the
   source→chapter mapping (`book-architecture.md`).
2. **Strip modular tells:**
   - the per-file provenance footer ("Reviewed and updated …", source-HTML credits);
   - file 06's version/hardware block and emoji sign-off;
   - the standalone H1 and any "this guide is about…" opening that only makes sense in isolation;
   - self-labels in titles ("Student Guide", "Complete … Guide").
3. **Add a chapter opener** (template below).
4. **Rework the first paragraph** into a book paragraph that assumes the reader arrived from the
   previous chapter — not a cold standalone intro.
5. **Deduplicate:** if this chapter re-teaches a foundation owned by Ch 2 or Ch 3, replace the full
   treatment with a one-line reminder + cross-reference.
6. **Convert cross-references** (table below).
7. **Renumber** figures, tables, and displayed equations to the chapter scheme (`style-guide.md`).
8. **Add a chapter closer**: a short summary + a one- or two-sentence bridge to the next chapter.
9. **Verify** every worked example arithmetically (Python) and normalise any LaTeX to plain Unicode.
10. **Prose pass (per-chapter, the primary one).** Run the prose-linter scanner on the chapter,
    focused on the *new* connective prose (opener, bridges, summary, transitions) — carried-over
    source content is already human. Fix real tells; don't over-sand. This is a chapter
    "definition of done." A final light *whole-book* consistency pass (cross-chapter repetition,
    voice drift) happens once at the end — see `production.md`.

## Chapter opener template

```
# Chapter N. <Title>

<One or two sentences: what this chapter is about and why it matters now — written as a
continuation of the book, not a standalone abstract.>

**In this chapter**
- <objective 1>
- <objective 2>
- <objective 3>

<Optional: "Before this chapter" — one line pointing back to prerequisite chapters,
e.g. "This chapter assumes the dot product and softmax from Chapter 2.">
```

Keep the openers short. The source's own rich intros (especially file 02) can supply the framing;
compress them, don't stack a second abstract on top.

## Chapter closer template

```
## Summary

<3–6 bullets or a short paragraph capturing the chapter's load-bearing ideas — not a restatement
of every heading.>

> **Coming up:** <one or two sentences bridging to the next chapter — name the open question this
> chapter leaves that the next one answers.>
```

The bridge is the single most important thing a reference lacks. Every chapter except the last gets
one; make it a genuine hand-off ("We can now embed and position tokens — but nothing yet lets them
*interact*. That is attention, and it is the subject of Chapter 5.").

## Cross-reference conversion

| Source form | Book form |
|---|---|
| `see §15`, `(see §2)` (file 02's internal numbering) | `see Chapter <N>` / `see §<N.M>` in book numbering |
| `*Related:* Term · Term` (glossary chains in 01/05) | Keep inside Appendix A as see-alsos; in a chapter, link the term's **first mention** to its glossary entry |
| "in this guide" / "this document" | "in this book" / "in this chapter" / "in Part <N>" |
| "the section above/below" (within a former file) | resolve to the actual chapter/section, or "earlier/later in this chapter" |
| A concept re-taught from scratch | "Recall from Chapter <N> that …" + reference |

Build a global anchor scheme first (chapter numbers, then `N.M` sections) so links resolve while you
work. Do a final link-audit pass once all chapters exist — dangling `§` references are the most
likely breakage.

## Optional additions (ask before adding)

- **Exercises / "Try it" boxes** at chapter ends — natural for Ch 2 (math), Ch 10–11 (code), Ch 17
  (evaluation). Only add if the author wants them; don't invent filler.
- **Key-terms list** per chapter, each linking to Appendix A.

## Per-chapter checklist

- [ ] Footer, version block, and standalone H1/self-label removed
- [ ] Opener added (title, framing sentence, objectives)
- [ ] First paragraph reads as a continuation, not a cold start
- [ ] Duplicated foundations replaced with reminders + cross-refs
- [ ] All `§` / "this guide" / `*Related:*` references converted
- [ ] Figures, tables, equations renumbered to chapter scheme
- [ ] Closer added (summary + bridge to next chapter)
- [ ] Worked examples arithmetically verified. Figures regenerated from their generator
- [ ] (code chapters) every `python` listing **executed** (build on a tiny config → forward → one
  training step → generate), not just parsed; code comments cross-reference the relevant chapters and
  the Chapter 2 math (chapter-level refs, not section numbers, so they don't go stale)
- [ ] prose-linter scan run on the chapter
