# Evaluating Language Models

Evaluation is not a step at the end of training. It is the feedback loop that runs through
every stage. Without it you cannot tell whether a training decision improved the model, degraded
it, or simply shifted one capability while eroding another. RLHF routinely improves helpfulness
while slightly degrading factual accuracy. Only continuous measurement catches this before
the model ships.

This chapter covers what good evaluation looks like across the full training pipeline: the
metrics appropriate to each stage, the benchmark landscape and its failure modes, and the human
evaluation methods that anchor everything when automated metrics fall short.

**In this chapter**

- Perplexity: what it measures and how to read it.
- Stage-specific evaluation, from perplexity through benchmark accuracy to human win-rates.
- Classic benchmarks and why they have saturated.
- The current frontier evaluation suite (GPQA Diamond, SWE-bench, AIME 2025, ARC-AGI).
- LLM-as-judge and Chatbot Arena.
- The three traps: contamination, Goodhart's Law, and the metric–capability gap.

## Perplexity: the primary pre-training metric

Perplexity is the standard metric for tracking pre-training progress. It converts the raw
cross-entropy loss (Chapter 2, Chapter 13) into a more intuitive number: on average, how many
equally likely choices did the model think it had at each token position?

$$\text{Perplexity} = \exp(L_{\text{CE}}) = \exp\!\left(-\frac{1}{T}\sum_t \log P(y_t \mid y_{<t})\right)$$

Lower is better. A perplexity of 10 means the model was as uncertain as choosing uniformly
among 10 words at each step. A perplexity of 1 would be perfect prediction with no uncertainty.

### Worked example: one sentence

Evaluate a trained model on "The French Revolution began in 1789." The model conditions on each
prefix and assigns a probability to the next token:

```
// Model predicts each token given its prefix
P(French     | The)                       = 0.04  →  L = 3.22
P(Revolution | The French)                = 0.31  →  L = 1.17
P(began      | The French Revolution)     = 0.45  →  L = 0.80
P(in         | The French Revolution began) = 0.72 →  L = 0.33
P(1789       | The French Revolution began in) = 0.26 → L = 1.35

Avg loss   = (3.22 + 1.17 + 0.80 + 0.33 + 1.35) / 5 = 1.37
Perplexity = e^1.37 = 3.9   ← very good (model knows this fact)
```

The model was most uncertain about "French" (only 4% probability, since there are many things that
can follow "The") and "1789" (26%, because many years were possible). Function words like "in" (72%)
were nearly free. A perplexity of 3.9 means the model was, on average, choosing among fewer
than 4 plausible next words, a strong signal that it has encoded this historical fact.

### Perplexity across the training pipeline

Perplexity on held-out text changes dramatically across training stages:

| Stage | Approximate PPL | Interpretation |
|---|---|---|
| Random initialization | ~30,000–50,000 | Uniform over vocabulary, pure noise |
| After 10B tokens | ~25 | Learns basic language structure |
| After 1T tokens | ~8 | Solid world knowledge and fluency |
| After full pre-training | ~5–8 | Frontier base model quality |
| After SFT | ~6–7* | Slight increase, distribution shifts toward instruction style |
| Human-written text | ~2–4 | Empirical estimate of the entropy of well-formed English, not a hard floor |

::: {.caption}
**Table 17.1.** Perplexity at each stage of the training pipeline.
:::

\*Perplexity typically rises slightly after SFT, because the model's distribution shifts to
match instruction-following style, which differs from the general pre-training corpus.

::: {.callout .caution}
**Perplexity is not a capability measure.** A model can have low perplexity on held-out web
text while failing basic arithmetic, exhibiting hallucinations, or being unable to follow
instructions. Perplexity measures language modeling quality on a fixed distribution, so it says
nothing about whether the model can *do* things. Use perplexity to track training progress.
Use benchmarks and human evaluations to measure capability.
:::

## Stage-specific evaluation

Each training stage requires different measurement methods:

| Stage | What to measure | How |
|---|---|---|
| Pre-training | Language modeling quality | Perplexity on held-out text, loss curve monitoring |
| SFT | Instruction-following accuracy | Held-out prompts with automated format checkers, human comparison with base model |
| Reward model | Ranking accuracy | Agreement with held-out human preferences, preference pair accuracy |
| RLHF / DPO | Alignment quality | Reward model scores (imperfect proxy), human preference win-rate vs. previous model |
| Reasoning RL | Correctness on hard tasks | Benchmark accuracy on math / code / logic, pass@k on code |
| All stages | Regressions | Spot-check a fixed held-out set after every major change |

::: {.caption}
**Table 17.2.** What to measure at each training stage, and how.
:::

The last row is the most important in practice. Training stages regularly trade one capability
for another. An RLHF run that improves helpfulness might quietly degrade factual accuracy.
Only a fixed held-out evaluation suite, run after every significant change, catches these
regressions before they reach users.

## Classic capability benchmarks

Standardized benchmarks allow comparisons across models and training runs. The most widely used:

| Benchmark | Tests | Metric | Human baseline |
|---|---|---|---|
| MMLU | 57 academic subjects (law, medicine, physics…) | % correct | ~89% |
| HumanEval | Python function completion from docstring | pass@1 | — (none published) |
| GSM8K | Multi-step grade-school math word problems | % correct | ~95% |
| MATH | Competition mathematics (AMC/AIME difficulty) | % correct | ~40% |
| TruthfulQA | Questions where models commonly hallucinate | % truthful | 94% |
| HellaSwag | Commonsense sentence completion | % correct | ~95% |
| BIG-Bench Hard | Hard reasoning tasks (23 sub-tasks) | % correct | Varies |

::: {.caption}
**Table 17.3.** Classic capability benchmarks and their metrics.
:::

::: {.callout .note}
**pass@k.** For code benchmarks, "pass@1" means: does a single attempt pass the test suite?
"pass@10" means: does *at least one* of 10 attempts pass? Pass@k increases with k because
diverse sampling improves the chance of finding a correct solution. It is averaged over the
benchmark's problems, and is normally computed with an unbiased estimator from n > k samples rather
than from exactly k.

One caveat: pass@k credits a problem as solved if *any* of the k attempts passes, which quietly
assumes something can tell you which attempt was the good one. When you have a verifier (unit
tests, a math checker) that assumption holds and pass@k is a real capability (this is best-of-N,
Chapter 16). When you do not, pass@10 is an upper bound on what you could actually ship.
:::

## Benchmark saturation and the frontier suite

Most of the classic benchmarks above are now **saturated**: frontier models score 90+ on MMLU,
HumanEval, and GSM8K, so they no longer discriminate between top systems. Worse, several are
suspected of **training data contamination**: test questions may appear verbatim in the
pre-training corpus, inflating apparent capability.

::: {.figure}
![](assets/figures/ch17/fig-benchmark-saturation.svg)
:::

::: {.caption}
**Figure 17.1.** Benchmark saturation. As models improve, classic benchmarks climb toward the ceiling and stop telling top systems apart, so the field keeps replacing them with harder ones that still discriminate.
:::

The current frontier evaluation suite is harder, more contamination-resistant, and
built to reward the reasoning models of Chapter 16:

| Benchmark | Tests | Why it resists saturation |
|---|---|---|
| **GPQA Diamond** | 198 PhD-level "Google-proof" science questions | Requires genuine expert reasoning, hard to find online |
| **MMLU-Pro** | Harder, cleaned MMLU with more options | Removes easy questions, reduces guessing advantage |
| **AIME (current year)** | Competition mathematics (AMC/AIME level) | A fresh problem set every year, so contamination resets annually, though each year's set saturates within months (frontier models reached ~100% on AIME 2025) |
| **SWE-bench Verified** | 500 human-vetted real GitHub issues (resolve + pass tests) | Full software engineering, hard to fake with pattern-matching |
| **ARC-AGI** | Abstract visual reasoning patterns | Resists memorization, tests novel generalization |

::: {.caption}
**Table 17.4.** The frontier benchmark suite, and why each resists saturation.
:::

::: {.callout .deepdive}
**SWE-bench Verified as the coding standard.** SWE-bench presents a model with a real GitHub
issue, the repository codebase, and a test suite. The model must generate a patch that fixes
the issue and passes the tests: end-to-end software engineering, not just code completion.
It was human-verified (the "Verified" suffix) by 93 professional Python developers, who screened a
random 1,699-sample slice of the original SWE-bench. They flagged 38% of tasks as having an
underspecified issue description and 61% as having unit tests that could reject a valid fix. SWE-bench
Verified is the resulting human-screened set of 500 tasks. As of 2026, frontier
models score 40–70% on it. This is considered the most credible single-number test of real-world coding ability.
:::

## LLM-as-judge evaluation

For tasks without a single correct answer (open-ended writing, explanation quality, multi-step
reasoning) automated evaluation needs a richer signal than a benchmark accuracy number.
**LLM-as-judge** uses a capable model (GPT-4, Claude, Gemini) to rate or compare responses,
optionally emitting a chain-of-thought critique before a verdict.

Typical setup: give the judge model a response (or two responses to compare), a rubric, and
instructions for how to score. The judge's verdicts serve as the evaluation signal.

Advantages:

- Scales cheaply to any task, with no bespoke test suite required.
- Can evaluate subjective qualities (depth of explanation, appropriate hedging, creativity).
- Can generate *explanations* of why one response is better, useful for debugging.

Limitations:

- Inherits the judge model's biases. A GPT-4 judge tends to prefer GPT-4-style responses.
- "Verbosity bias": longer responses are systematically rated higher, independent of quality.
- Cannot reliably evaluate correctness on expert domains where the judge itself is fallible.

::: {.callout .caution}
**Positional bias.** When shown two responses (A and B), LLM judges rate whichever appears
first higher about 55–60% of the time, a strong positional bias. In practice, good
LLM-as-judge evaluations always swap the presentation order and average the results.
:::

## Human evaluation: Chatbot Arena

The most credible real-world evaluation is crowdsourced human preference: **Chatbot Arena**
(formerly LMSYS, now LMArena.ai). Users chat with two anonymous models simultaneously, then
vote for whichever gave the better response. With millions of battles, Elo-style scores emerge
that reflect genuine conversational quality. (Elo is the rating system chess uses: you gain more
points for beating a strong opponent than a weak one, so the numbers track real skill rather than
raw win counts.)

Advantages of the Arena:

- Avoids benchmark contamination entirely, because it uses live user queries.
- The massive sample size (millions of votes) makes the rankings statistically robust.
- Captures qualities that automated metrics miss: genuine helpfulness, conversational flow,
  appropriate tone.

Limitations:

- Slow to update: getting enough votes on a new model takes weeks.
- Swayed by stylistic signals: longer, more formatted responses score higher even when the
  content is no better (the "verbosity effect").
- User population is self-selected, skewed toward technically literate users, which may not
  represent all use cases.
- Cannot be the sole arbiter for safety: users won't systematically probe for harmful outputs.

In practice, Arena Elo and verifiable reasoning benchmarks (AIME, SWE-bench) are read
together: Arena captures conversational quality, benchmarks capture task correctness, and
neither alone is sufficient.

## The three evaluation traps

::: {.callout .caution}
**Contamination.** A model that has seen benchmark test questions in its pre-training data will
score higher than its genuine capability warrants. Mitigation: use freshly created benchmarks
(AIME changes every year), use benchmarks with private test sets, or test on problems generated
after the model's training cutoff. Treat any benchmark where the training data cannot be
audited with suspicion.
:::

::: {.callout .caution}
**Goodhart's Law.** "When a measure becomes a target, it ceases to be a good measure." Once
teams optimize training specifically for a published benchmark, that benchmark stops reflecting
genuine capability. MMLU scores rose sharply in 2023–2024 as labs trained on MMLU-style data,
but real-world performance gains were much smaller. The solution: rotate benchmark suites,
weight recent and private evaluations more heavily, and watch for unusual score jumps that are
not accompanied by capability gains on other benchmarks.
:::

::: {.callout .caution}
**The metric–capability gap.** Low perplexity does not mean good instruction following. High
benchmark accuracy does not mean the model is safe to deploy. A model can score 90% on MMLU
while confidently hallucinating facts it was never tested on. Every metric measures a proxy.
The question to ask for any evaluation is: does this metric actually track the thing I care
about in deployment, and what does it miss?
:::

## Summary

- **Perplexity** is the standard pre-training metric: PPL = e^(avg cross-entropy). It tracks
  training progress and detects regressions, but does not measure capability.
- **Stage-specific evaluation** is essential: perplexity for pre-training, instruction-following
  accuracy for SFT, human win-rates for alignment, benchmark accuracy for reasoning RL.
- **Classic benchmarks** (MMLU, HumanEval, GSM8K) are saturated. The current frontier suite
  (GPQA Diamond, SWE-bench Verified, AIME 2025, ARC-AGI) is built to remain discriminating.
- **LLM-as-judge** scales cheaply but inherits judge biases. Always swap presentation order
  to counteract positional bias.
- **Chatbot Arena** gives the most credible real-world quality signal but is slow to update
  and dominated by stylistic preferences.
- The three traps (contamination, Goodhart's Law, and the metric–capability gap) apply to
  every evaluation approach. Use multiple methods together. Trust none alone.

> **Coming up:** Part IV is complete. You now understand how models are built and trained, from
> random weights through pre-training, SFT, preference alignment, and reasoning RL. Part V turns
> to deployment: Chapter 18 covers retrieval-augmented generation, the technique that grounds
> language models in up-to-date external knowledge.
