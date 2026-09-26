# Chapter 2 figures

Eleven static SVG figures for the mathematics chapter, in the book's classic-book palette
(dark-red `#A80036` accent, cream / warm-paper panels, serif labels, monospace for numbers and
vectors). They are **wired into `part-1/ch-02-mathematics.md`** as `Figure 2.1`–`Figure 2.11`.

Source of truth is the generator, not the SVGs: **`tools/figures/gen_ch02_figures.py`**. Every
coordinate and plotted value is computed there, so the figures stay consistent with the worked
examples. Edit the generator, never the SVG by hand, then regenerate.

## Regenerate

```bash
# from the repo root
python3 tools/figures/gen_ch02_figures.py          # writes the 11 SVGs
python3 tools/figures/gen_ch02_figures.py --png     # also writes PNG previews to _preview/
```

## Prerequisites

| Tool | Needed for | Notes |
|---|---|---|
| **Python 3.10+** | writing the SVGs | Standard library only. No third-party packages required. |
| **cairosvg** (`pip install cairosvg`) | the optional `--png` preview | Preview only; not needed for the book build. |
| **WeasyPrint** | the PDF build | Already a book dependency. Renders the SVGs natively — no new build step. |

Nothing new is added to the book build itself.

## Filenames → in-book figure number

Filenames describe content; the `Figure 2.x` number lives only in the caption (so inserting or
reordering a figure never forces a file rename). Numbers below are the current order of appearance.

| File | Figure | Concept |
|---|---|---|
| `fig-vectors-in-space.svg` | 2.1 | words as points in space (positioning) |
| `fig-matmul-shapes.svg` | 2.2 | matrix multiply: row·column → cell, and the shape rule |
| `fig-dot-product-alignment.svg` | 2.3 | dot product = alignment (angles, perpendicularity) |
| `fig-cosine-length-invariance.svg` | 2.4 | cosine reads the angle, not the length |
| `fig-transpose-diagonal.svg` | 2.5 | transpose = flip across the diagonal |
| `fig-rotation-rope.svg` | 2.6 | rotation, and how RoPE turns position into an angle |
| `fig-softmax-bars.svg` | 2.7 | softmax: scores → probabilities |
| `fig-temperature.svg` | 2.8 | temperature reshapes the distribution |
| `fig-sigmoid.svg` | 2.9 | the sigmoid S-curve |
| `fig-gradient-descent.svg` | 2.10 | gradient descent and the step size |
| `fig-quantization.svg` | 2.11 | quantization snaps to a coarse grid |

## Embedding pattern (already applied)

Each figure is a `.figure` fenced div (centered by `.figure` in `classic-book.css`) with an
**empty alt** — an alt string makes pandoc emit a duplicate figcaption — followed by a `.caption`
block:

```markdown
::: {.figure}
![](assets/figures/ch02/fig-vectors-in-space.svg)
:::

::: {.caption}
**Figure 2.1.** Words as points in space. Similar words (cat, dog) sit close together, while
unrelated words (car, king) sit far apart.
:::
```

## Notes

- Verified end to end: `make check` passes (prose + structure gates), and a scoped
  `build-pdf.py` render of ch-02 shows all eleven figures centered with their captions.
- The same generator pattern extends cleanly to the other figure-less chapters the review flagged:
  Chapter 4 (the clock/rotation view of position) and Chapter 3 (the embedding scatter and the
  `king − man + woman ≈ queen` parallelogram).
- `_preview/` PNGs are throwaway; regenerate with `--png`. The SVGs are the deliverable.
