PYTHON  := python3
SCRIPT  := build-pdf.py
OUT     := dist/inside-the-transformer.pdf

FRONTMATTER := \
	frontmatter/cover.md \
	frontmatter/00-copyright.md \
	frontmatter/02-about-cover.md \
	frontmatter/01-preface.md \
	frontmatter/03-conceptual-track.md

CHAPTERS := \
	part-1/ch-01-what-is-a-transformer.md \
	part-1/ch-02-mathematics.md \
	part-2/ch-03-tokenization-embeddings.md \
	part-2/ch-04-positional-encoding.md \
	part-2/ch-05-attention.md \
	part-2/ch-06-transformer-block.md \
	part-2/ch-07-interpretability.md \
	part-2/ch-08-inference.md \
	part-2/ch-09-architecture-trends.md \
	part-3/ch-10-build-gpt.md \
	part-3/ch-11-annotated.md \
	part-4/ch-12-training-pipeline.md \
	part-4/ch-13-pretraining.md \
	part-4/ch-14-sft.md \
	part-4/ch-15-preferences.md \
	part-4/ch-16-reasoning-rl.md \
	part-4/ch-17-evaluation.md \
	part-5/ch-18-rag.md \
	part-5/ch-19-agents.md \
	part-5/ch-20-offline.md

BACKMATTER := \
	backmatter/appendix-a-glossary.md \
	backmatter/appendix-b-specs.md \
	backmatter/acknowledgments.md \
	backmatter/bibliography.md

# Every manuscript file is gated. Do not narrow this list — an ungated file is
# an invisible exemption, which is how 460 em-dashes hid once already.
PROSE := $(FRONTMATTER) $(CHAPTERS) $(BACKMATTER)

# Public prose that is not manuscript. Gated for style only. The structure gate
# is manuscript-specific (it resolves "Chapter N" against the files in the same
# run, and expects captioned tables), so these are deliberately excluded from it.
DOCS := README.md CONTRIBUTING.md CLAUDE.md

.PHONY: pdf check
# The prose gate runs first. A build that violates B9/B18 or the render-only
# checks should not produce a PDF at all.
pdf: check
	$(PYTHON) $(SCRIPT) $(OUT) $(FRONTMATTER) $(CHAPTERS) $(BACKMATTER)

check:
	@$(PYTHON) tools/check-prose.py $(PROSE)
	@$(PYTHON) tools/check-structure.py $(PROSE)
	@$(PYTHON) tools/check-prose.py $(DOCS)
