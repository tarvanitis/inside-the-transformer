# Inside a Trained Model: Mechanistic Interpretability

We've now built the whole stack: embeddings, attention, the block, depth. But building the machine
and understanding what it *does* once trained are two different things. A trained transformer is often
called a "statistical pattern matcher," which isn't wrong but undersells what's inside.
**Mechanistic interpretability**, reverse-engineering the computations a trained model actually
performs, has found something more specific: identifiable *circuits*, subnetworks that implement
recognizable algorithms. This chapter looks at a few of the best-understood ones.

**In this chapter**

- What a "circuit" is.
- Induction heads, a two-layer circuit that copies patterns, and the engine behind in-context
  learning.
- How the feed-forward layers store facts as key-value memories.
- The residual stream as the model's working memory.

This chapter builds on attention and its heads (Chapter 5) and the feed-forward network and residual
stream (Chapter 6).

## What is a circuit?

::: {.callout .lens}
**Mechanistic.** A *circuit* is a group of components (attention heads and FFN neurons) that work
together across layers to implement a specific computation. Much as a CPU has identifiable functional
units (an arithmetic unit, a cache controller, a branch predictor), a trained transformer has
identifiable computational circuits.
:::

The claim here is stronger than "the model has tendencies." These are concrete, testable mechanisms:
you can locate the heads involved, ablate them, and watch the specific behavior disappear.

## Induction heads

The best-studied circuit is the **induction head**. Its behavior is easy to state:

```text
Input:   … A B … A            (A occurred earlier, and was followed by B)
Output:  the model predicts B  (whatever followed A last time)
```

Concretely: feed the model *"Mr Dursley was the director of a firm called Grunnings. Mr"* and it
predicts *" D"* → *"urs"* → *"ley"*. It has not memorized that name. It found the earlier "Mr", looked
at what followed it, and copied that forward.

This is *pattern copying* (a general-purpose "what came next last time?" retrieval), and it needs at
least two layers working together:

::: {.figure}
![](assets/figures/ch07/fig-induction-circuit.svg)
:::

::: {.caption}
**Figure 7.1.** The two-layer induction-head circuit: previous-token head, then induction head.
:::

- **Layer ℓ: a "previous-token head"** attends to the token just before the current one and carries
  that information forward, so position *t* now holds a trace of token *t − 1*.
- **Layer ℓ + 1: the "induction head"** uses that trace to find an earlier place where the current
  token appeared, then attends to *the token that followed it*, copying what came next last time.

::: {.figure}
![](assets/figures/ch07/fig-induction-trace.svg)
:::

::: {.caption}
**Figure 7.2.** An induction head in action. Seeing "Mr" again, the circuit finds the earlier "Mr", looks at the token that followed it, and copies that token forward as its prediction.
:::

::: {.callout .idea}
**Why induction heads matter.** This is not memorization — it is an *algorithm* for in-context learning,
and a key mechanism behind few-shot prompting (show the model a pattern and it continues it). Notably,
induction heads tend to appear *suddenly* during training, in a sharp phase transition rather than a
gradual improvement.
:::

## FFN neurons as key-value memories

If attention *routes* information, the feed-forward layers (Chapter 6) *store* it. Geva et al. (2021)
showed that an FFN behaves like a key-value memory: the *columns* of its first matrix `W₁` act as
**keys** (each column is a `d_model`-long pattern, and a token vector that aligns with it switches
that neuron on), and the *rows* of its second matrix `W₂` act as **values** that write information
back into the residual stream.

When a token's vector matches a stored key pattern, the corresponding neuron fires and adds its value
to the representation. This is how factual knowledge is held: an input about the Eiffel Tower
activates neurons whose values encode "Paris", "France", "tall structure", and so on.

::: {.figure}
![](assets/figures/ch07/fig-ffn-memory.svg)
:::

::: {.caption}
**Figure 7.3.** A feed-forward layer as key-value memory. A token that matches a stored key (a column of W₁) switches on that neuron, which writes its value (a row of W₂) back into the residual stream.
:::

## The residual stream as working memory

Chapters 5 and 6 gave us the two verbs of a transformer. Interpretability ties them together.

::: {.callout .lens}
**Representation.** The residual stream is the model's working memory. Attention *reads* from it
selectively and *writes back* context-integrated information. FFN neurons *fire* on pattern matches
and *write back* factual and relational knowledge. The representation at the final layer is the
accumulated result of all those reads and writes.
:::

Seen this way, a forward pass is less one monolithic function than a sequence of small, legible
operations on a shared workspace, which is exactly what makes reverse-engineering possible at all.

## Summary

- **Mechanistic interpretability** reverse-engineers the computations inside a trained model, finding
  **circuits**: groups of components that implement identifiable algorithms.
- **Induction heads** are a two-layer circuit that copies patterns ("what followed this token last
  time?") and underpin in-context learning. They emerge in a sharp training phase transition.
- The **feed-forward layers** act as **key-value memories** that store factual knowledge, where keys select
  neurons, values write facts into the residual stream.
- The **residual stream** is the shared working memory that attention and the FFN read from and write
  to across every layer.

> **Coming up:** We've built the model and looked at what forms inside it. Chapter 8 turns to what it
> was built for, *generation*: how a trained transformer produces text one token at a time, and the
> tricks (the KV-cache, sampling controls) that make it fast.
