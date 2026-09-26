# What Is a Transformer?

Nearly every large language model in use today (the one that drafts your email, the one that writes
code, the one that answers a question with a paragraph that reads as though a person wrote it) is
built on a single architecture called the *transformer*. Before we take it apart, this chapter
answers the first question plainly: what is this thing, where did it come from, and why did it win?

**In this chapter**

- What a transformer is, in plain terms, and the one idea that makes it work.
- A short history: how we got from 2017 to today's reasoning models.
- The problem transformers solved, and why the models that came before them hit a wall.

No prior knowledge is assumed here. The mathematics begins in Chapter 2.

## The one-sentence version

A **transformer** is a type of artificial neural network, a mathematical system loosely inspired by
how neurons in the brain connect and communicate. Transformers are built to process sequences (like
sentences), and they have become the foundation of modern AI language models.

::: {.callout .plain}
Think of a transformer as a very sophisticated pattern-matching machine. You
give it some text (say, the start of a sentence) and it predicts what comes next by recognizing
patterns it learned from reading billions of documents during training.
:::

That prediction is the whole game, and it is worth sitting with how strange it is that so much
follows from it. A system whose only trained ability is "guess the next chunk of text" turns out to
be able to translate, summarize, write code, and hold a conversation. Most of this book is an answer
to *why*.

The engine underneath is one idea:

::: {.callout .idea}
The transformer's superpower is **attention**: the ability to look at all the words
in a passage at once and work out which ones matter for understanding each other. Before
transformers, models had to read one word at a time, like peering at a page through a keyhole. A
transformer sees the whole page at once.
:::

::: {.callout .lens}
**Geometric.** In the older models, information flowed along a *line*, token to token. In a
transformer it flows across a *complete graph*, where every token can connect directly to every other
token in a single step. The geometry changes from a queue to a web.
:::

## A quick history

- **2017.** Google researchers publish *Attention Is All You Need*, introducing the transformer.
- **2018–2020.** GPT, BERT, GPT-2, and GPT-3 show that transformers learn language remarkably well.
- **2022–2023.** ChatGPT (GPT-3.5/4), Claude, and Llama show they can hold human-like conversations.
- **2024.** Mixture-of-Experts models (MoE: each token is routed through only a few of many
  specialized sub-networks, Chapter 9) and long-context (100K+) models mature. OpenAI's o1 introduces
  inference-time ("thinking") compute scaling.
- **2025.** Reasoning models go mainstream (DeepSeek-R1, OpenAI o3, GPT-5's unified fast/thinking
  router, Claude with extended thinking, Gemini 2.5). Mixture-of-Experts becomes the frontier default
  (DeepSeek-V3, Llama 4, Qwen3), and hybrid Mamba–transformer models ship (IBM Granite 4, NVIDIA
  Nemotron-H).
- **2026.** The *frontier* (the current most-capable, state-of-the-art models) is a crowded field of reasoning-capable, mostly Mixture-of-Experts models
  competing on price-to-performance.

> **Current model families (as of 2026-09-14).** Version numbers churn fast, so treat this as a
> dated snapshot pulled from vendor model pages on 2026-09-14. The durable facts are the *shape*
> (reasoning-capable, increasingly Mixture-of-Experts, multimodal, long-context). Current flagships:
>
> - **Proprietary:** OpenAI **GPT-6 Astra** (flagship since September 2026), above the **GPT-5.6**
>   family (`gpt-5.6-sol`, plus Terra/Luna tiers; ~1M-token context); Anthropic's **Claude 5 family**:
>   **Fable 5.1** (highest capability), **Opus 5**, **Sonnet 5**, plus **Haiku 4.5**, with adaptive
>   thinking and a 1M-token context (200K on Haiku); Google **Gemini 3** (latest **Gemini 3.8
>   Flash**, with **Gemini 3.1 Pro** heading the Pro line; the 2.5 series remains available); xAI
>   **Grok 4.6** (500K context).
> - **Open-weight:** DeepSeek **V4** (`deepseek-v4-pro` / `-flash`, MoE, 1M context, thinking modes);
>   Alibaba **Qwen3.8** (incl. the Qwen3.8-2.4T-A95B MoE); Z.ai **GLM-5.2** (MoE, MIT-licensed, 1M
>   context), currently at the top of the open-weight rankings; Moonshot **Kimi K3** (2.8T-parameter
>   MoE, 1M context); Meta **Llama 4** (Scout / Maverick, MoE; still the current Llama generation,
>   though Meta's frontier work has moved to the proprietary **Muse** line); **Mistral Large 3** /
>   **Medium 3.5** (research license, not fully open); Google **Gemma 4**.

## Why transformers? The problem with reading one word at a time

To see why the transformer mattered, it helps to know what it replaced. Before 2017, the dominant
architecture for language was the **recurrent neural network (RNN)** and its more capable variant,
the **long short-term memory (LSTM)** network. These models processed tokens one at a time, left to
right, carrying a fixed-size "hidden state" (a single list of numbers) meant to summarize
everything seen so far.

::: {.callout .plain}
A *token* is the unit a model actually reads. Usually a word, sometimes a piece of one ("un-" +
"believable"), sometimes just a space or a comma. A model never sees letters or sentences, only a
stream of these pieces. Chapter 3 shows exactly how text gets chopped up. For now, read "token" as
"roughly a word." Every term set in bold like this also has a short entry in Appendix A, the
glossary at the back.
:::

That worked reasonably well, but it had three problems baked in:

- **A sequential bottleneck.** The token at position *t* can only be computed after position *t − 1*.
  No parallelism. Training on long sequences was slow.
- **Vanishing gradients.** During training, the learning signal has to travel backward through every
  step. Over long sequences it shrinks toward zero before it reaches the early tokens, so the model
  effectively forgets what happened far back.
- **Fixed-width memory.** All the past is squeezed into one vector of fixed size. Whether the
  sentence is 10 tokens or 10,000, the "memory" is the same bottleneck.

LSTMs eased the vanishing-gradient problem with gating, but they didn't solve the parallelism or the
fixed-memory problems. Those were structural.

::: {.callout .idea}
The transformer (Vaswani et al., 2017, *Attention Is All You Need*) discards
recurrence entirely. Every token attends *directly* to every other token in a single step. The
dependency between any two tokens is one operation away — not *distance* operations away.
:::

That single change is what removed the bottleneck. Because every position is computed in parallel,
transformers train efficiently on enormous corpora. Every token can reach every other token
directly, so nothing has to be remembered through a narrow channel. Scale became possible, and scale is
most of what has happened since.

## Summary

- A transformer is a neural network for processing sequences. Modern language models are
  transformers, and their one trained skill is predicting the next chunk of text.
- Its defining mechanism is **attention**: comparing every token with every other token at once,
  rather than reading left to right through a keyhole.
- It replaced RNNs and LSTMs, which were held back by a sequential bottleneck, vanishing gradients,
  and a fixed-size memory. Removing recurrence removed all three at once and made scaling possible.

> **Coming up:** "Compare," "combine," and "weigh" are convenient words, but underneath they are all
> arithmetic: multiplying lists of numbers and adding up the results. Chapter 2 assembles that
> arithmetic from the ground up, so that when attention appears in Chapter 5 it reads as a formula
> you can compute by hand, not a black box.
