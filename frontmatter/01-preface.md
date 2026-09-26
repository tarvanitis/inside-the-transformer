::: {.frontmatter}

# Preface

Most explanations of language models sit at one of two extremes. On one side are the breezy
analogies ("it's just autocomplete on steroids") that feel satisfying for about a minute and then
leave you unable to answer the next question. On the other are the research papers, which tend to
assume you already know what a key and a value are, and why anyone would think to divide by √d_k.
There isn't much in between: a single path that starts from nothing and doesn't stop until you can
see the actual machinery.

That path is what this book tries to be.

I came to it as a software engineer of thirty years, not as a researcher. That shaped what follows.
The book treats a Transformer as a machine to take apart rather than a result to prove, and it
assumes you want to know what each part does and why it is there.

## Who this is for

If you have used a chatbot and come away curious (not satisfied by "it predicts the next word," but
wanting to know *how*) you are the reader I had in mind. You don't need a mathematics degree, and
you don't need to have trained a neural network. Chapter 2 builds every mathematical idea Parts I
to III rely on, starting from what a vector is, and each concept arrives with a worked example you
can follow by hand. The reinforcement-learning and evaluation maths of Part IV arrives where it is
used, on the same plain-English-first terms.

The book also doesn't talk down. It ends somewhere real: the circuits that form inside a trained
model, the training pipeline that turns raw next-word prediction into a helpful assistant, and the
applied systems (retrieval, agents, on-device deployment) built on top. If you already know the
basics, skip ahead. The later chapters are written to reward you.

## What you'll be able to do by the end

- Follow text as it turns into numbers, and numbers as they turn into a prediction.
- Explain attention: what it actually computes, and why it broke so sharply from what came before.
- Read the architecture of a modern model and know what each block is there to do.
- Understand how a model learns, from pre-training through the alignment stages that shape how it
  behaves.
- Say what it means for a reasoning model to "think," and what that thinking costs.
- Reason about applied systems (RAG, agents, offline models) well enough to build one.

## The three lenses, and the icons in the margin

The same idea often looks different depending on where you stand. Three recurring **lenses** run
through the book:

- **Geometric.** What is happening in vector space: angles, distances, directions, rotations. The
  *shape* of the computation.
- **Representation.** What information is being encoded, compressed, moved, or routed. The *meaning*
  of the computation.
- **Mechanistic.** What the model is *literally* doing, at the level of matrices and operations. The
  *machinery* of the computation.

Small icons in the margin of the callout boxes mark what each box is for:

| Icon | What it marks |
|:--:|:--|
| ![](assets/icons/idea.svg){width=13px} | **Key idea.** A load-bearing insight worth pausing on. |
| ![](assets/icons/plain.svg){width=13px} | **Plain English.** The intuition, in everyday terms. |
| ![](assets/icons/lens.svg){width=13px} | **A lens.** One of the three views above (the box says which). |
| ![](assets/icons/deepdive.svg){width=13px} | **Deep dive.** Extra detail you can skip without getting lost. |
| ![](assets/icons/note.svg){width=13px} | **Note.** A useful aside. |
| ![](assets/icons/caution.svg){width=13px} | **Caution.** A common pitfall. |

## How to read this book

You can read it straight through, chapters ordered so each one earns the next, or you can treat it
as a reference and jump to what you need. Two things are built to be returned to at any time:
**Chapter 2** (the mathematics) and **Appendix A**, the glossary at the back. When a symbol or a
term stops making sense, that's where to look. **Appendix B** collects the model specifications and
dimension tables the chapters quote.

If you already know some of this, three routes through:

- **You know what attention does, and want the modern stack.** Start at Chapter 6, then 8 and 9.
- **You are here to build something.** Chapter 2 for notation, then straight to Part III.
- **You are here for training and alignment.** Chapter 12 opens that story and assumes only Part II.

The book is in five parts:

- **Part I: Foundations.** What a transformer is, and the mathematics it runs on.
- **Part II: How Transformers Work.** Tokenization and embeddings, positions, attention, the block,
  what forms inside a trained model, how it generates text, and where the frontier architectures
  have moved since.
- **Part III: Building One.** A working transformer in PyTorch, then every component annotated.
- **Part IV: Training and Alignment.** Pre-training through the post-training pipeline that makes a
  model useful and safe, and how any of it gets evaluated.
- **Part V: Applications.** Retrieval-augmented generation, agents, and running models offline.

## Running the code

Listings throughout the book run as written. The two full build-it chapters, 10 and 11, also ship as
standalone scripts in the companion `code/` directory, with chapter cross-references intact. The
shorter listings elsewhere are meant to be read in place or pasted into a session.

**What you need:**

```bash
# Python 3.10+ and PyTorch (CPU build is fine — no GPU required)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Chapter scripts:**

```bash
# Chapter 10 — Building a GPT from Scratch
python3 code/ch10_gpt.py
# Trains a small decoder-only GPT. Prints loss per step and generated token IDs.

# Chapter 11 — Every Component, Annotated (encoder–decoder transformer)
python3 code/ch11_transformer.py
# Builds and trains an encoder-decoder transformer. Confirms backprop works.
```

Both scripts seed the random number generator before anything is initialized, and print their output
so you can see exactly what each listing computes. If a number in the book says "Step 0 loss:
10.5752," running the script produces the same number. The worked examples are the executed code.

If you want to go one level deeper (building the autograd engine itself, not just the model)
Andrej Karpathy's **microGPT** (2026) assembles a complete GPT in around 200 lines of pure Python
with no external libraries: a from-scratch autograd engine, tokenizer, transformer, Adam, and both
training and inference loops.
It is the natural next step after finishing Part III.

## A note on dates

The mechanics in this book are stable. Attention, embeddings, the transformer block, the training
objectives. These have barely changed in years and are unlikely to change soon. Model *names*,
sizes, and the shape of the frontier, on the other hand, move fast. Where the text names specific
models, it says "as of" a date and treats them as a dated snapshot. Trust the shapes. Check the
version numbers.

## A note on sources and attribution

This book started the same way most learning does: with curiosity and a lot of browser tabs open.
I wanted to understand how language models actually work, down to the machinery underneath. Getting there required pulling material together from a wide range of sources:
research papers, tutorials, documentation, blog posts, and code from the people who built these
systems.

What I have tried to do here is organize that collected knowledge into a single coherent path, and
add explanations calibrated for readers who may not have a strong background in mathematics or
machine learning. The goal was to make the subject genuinely accessible without sacrificing accuracy.

I wrote this book with AI assistance. That would be a strange thing to leave unsaid in a book about
language models, so: the structure, the order in which ideas arrive, the choice of worked examples
and the checking of their arithmetic are mine, as are the figures, which are drawn by scripts from
values I verified. The prose was drafted with a model, then read, cut, corrected and sent back, over
many passes. A book about transformers, written with one.

The ideas, techniques, and results described in this book belong to their original authors. The
annotated bibliography at the back credits the key papers and resources this book draws from.
Copyright in those works remains with their respective owners. This book is a synthesis and a
study guide, not an original research contribution.

:::
