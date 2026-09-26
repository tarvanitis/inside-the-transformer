::: {.frontmatter .reading-guide}

# The Conceptual Track

*A math-light path through the book.*

## Who this track is for

You want to understand how large language models really work, the whole story from text going in to a
reasoned answer coming out, but you do not want to (or are not yet ready to) work through the
mathematics first. Maybe you are a student meeting this before calculus and linear algebra, a teacher
planning a lesson, a writer or manager who needs the real mechanics rather than a slogan, or simply
curious and short on time.

This is a path through the same book, not a different book. It leaves every formula exactly where it
is. It just tells you what to read closely, what to skim, and what to save for a second pass, so you
can follow the full arc on ideas alone.

## The one habit that makes this work

Every chapter is built the same way. It explains an idea in plain English first, then writes that same
idea in mathematics. The plain-English part is usually a short **Plain English** callout (the
speech-bubble icon) or the opening paragraphs of a section. Read those closely.

When you reach a formula or a block of code, do not stop to decode it. Read the sentence right before
it, which says what it is about, and the sentence right after it, which says what just happened. Then
keep moving. You will lose none of the story.

The math is never gone, only deferred. Chapter 2 builds every mathematical idea from scratch, starting
from what a number in a list is, and Appendix A (the glossary) defines every term. Those two are your
safety net, not your starting line.

## The path, chapter by chapter

"Read" means read the prose closely. "Skim" means read the plain-English parts and glide past the
formulas. "Later" means come back on a second pass, or when you decide to pick up the math or the code.

### Part I: Foundations

| Chapter | On this track | What you get |
|---|---|---|
| 1. What Is a Transformer? | Read | The whole picture in plain language. The best half hour you can spend here. |
| 2. The Mathematics You'll Need | Skim / reference | Read each idea's plain intro and its "why it matters" line. Skip the formal parts until a later chapter sends you back. Treat it as a dictionary, not a chapter. |

### Part II: How Transformers Work

| Chapter | On this track | What you get |
|---|---|---|
| 3. From Text to Numbers | Read | How words become numbers, told through everyday analogies. |
| 4. Position: How a Model Knows Word Order | Skim | The clock and rotation analogies carry the idea. Glide past the sine and cosine formula. |
| 5. Attention: The Core Mechanism | Read | The heart of the book. The search-engine analogy and the query/key/value table give you the whole mechanism. Skim the five-step arithmetic. |
| 6. The Transformer Block | Skim | What each part of a layer is for. Skip the normalization math. |
| 7. Inside a Trained Model | Read | Mostly conceptual, and genuinely fascinating: the circuits a trained model grows. Very little math. |
| 8. Generating Text: Inference | Read | The generation loop, sampling, and why speed tricks matter. |
| 9. The Full Architecture and Modern Trends | Read | The current landscape: Mixture-of-Experts, reasoning models, where the frontier is going. The diagrams do the heavy lifting. |

### Part III: Building One

| Chapter | On this track | What you get |
|---|---|---|
| 10. Building a GPT from Scratch in PyTorch | Later | Code-heavy. Read the prose to watch the pieces assemble. Save the code for when you want to build. |
| 11. Every Component, Annotated | Later | The same, in more detail. Return here if and when you program. |

### Part IV: Training and Alignment

| Chapter | On this track | What you get |
|---|---|---|
| 12. The Training Pipeline at a Glance | Read | The clearest plain-English map of how a model is actually made. High payoff, low math. |
| 13. Pre-training | Skim | What "learning from the whole internet" really means. Skim the loss formula and the scaling tables. |
| 14. Supervised Fine-Tuning | Skim | How a raw model becomes a helpful assistant. Read the LoRA on-ramp. Skip the code. |
| 15. Learning from Human Preferences | Skim | Read the plain on-ramps: the reward model, RLHF, and the idea that a model can be its own judge. Skip the derivations. |
| 16. Training Reasoning Models | Read | How a model learns to "think." Read the "grading on a curve" on-ramp. Skim the objective. |
| 17. Evaluating Language Models | Read | How we know whether any of it worked: benchmarks, human ratings, and the traps. Almost all prose. |

### Part V: Applications

| Chapter | On this track | What you get |
|---|---|---|
| 18. Retrieval-Augmented Generation | Read | Giving a model access to outside knowledge. Very approachable. |
| 19. AI Agents | Read | Letting a model take actions and use tools. Approachable. |
| 20. Running Models Offline | Read | The practical payoff: running a model on your own machine. |

### Back matter

| Section | On this track | What you get |
|---|---|---|
| Appendix A: Glossary | Companion | Keep it open. When a term stops making sense, look it up here. |
| Appendix B: Model Specifications | Reference | Dip in when you want the actual numbers behind a model family. |

## If you only have an afternoon

Read **Chapter 1**, the **Plain English** callouts in **Chapters 3 and 5**, and all of **Chapter 12**.
That is the shortest honest path to "I understand the gist," and it touches no mathematics at all.

## When you are ready for the math

Start again at **Chapter 2**, this time reading it in full. Each entry ends with a "where it shows up"
pointer to the chapter that puts it to work, so you can learn a tool and immediately see it used. Then
reread Part II with the formulas switched on. The ideas you already have from this track are exactly
the scaffolding the math hangs on, so the second pass is far easier than starting cold.

:::
