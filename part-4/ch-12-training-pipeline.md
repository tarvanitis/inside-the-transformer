# The Training Pipeline at a Glance

Part III built a transformer that works. This chapter asks the harder question: how do you make
it *useful*? The answer is a sequence of training stages, each with its own data, objective
function and purpose, that turns a randomly initialized network into a helpful assistant.
Part IV covers every stage in detail. This chapter is the map.

**In this chapter**

- The seven training stages and what each one does.
- The compute budget: where time and money actually go.
- Why stage ordering is a toolbox, not a fixed recipe.
- The objective functions, in plain English. The mathematics is in Chapter 2.

## From random weights to a reasoning model

A freshly initialized transformer is random noise: every parameter is a Gaussian draw, every
prediction is uniform over the vocabulary. The training pipeline is the process of converting
those random weights into something that can hold a conversation, write working code, and solve
competition mathematics.

No single training run does all of this at once. Modern frontier models are produced in distinct
stages that build on each other, each adding a different layer of capability.

::: {.figure}
![](assets/figures/ch12/fig-pipeline.svg)
:::

::: {.caption}
**Figure 12.1.** The seven training stages, from random weights to a deployed reasoning model.
:::

The diagram shows one common ordering. Section 12.4 explains why teams now reorder and combine
these stages freely.

## The seven stages

| Stage | Name | Data | Goal |
|---|---|---|---|
| ① | Pre-training | Trillions of tokens from the internet | World knowledge, language fluency |
| ② | Supervised Fine-Tuning (SFT) | (Instruction, ideal response) pairs | Instruction-following format and tone |
| ③ | Reward Model | Pairwise human preference comparisons | A differentiable proxy for human taste |
| ④ | RLHF / PPO | Rollouts scored by the reward model | Quality alignment via policy gradient RL |
| ⑤ | DPO | Same preference pairs as ③ | Alignment without a separate reward model |
| ⑥ | Reasoning RL (GRPO / RLVR) | Problems with checkable answers | Step-by-step reasoning, longer chains of thought |
| ⑦ | Evaluation | Held-out benchmarks and human judgments | Measure whether training is working |

::: {.caption}
**Table 12.1.** The seven stages: data, goal, and what each one changes.
:::

Each is covered in its own chapter (Chapters 13–17). Here is a one-paragraph preview of each.

**① Pre-training** is where the model develops language understanding and world knowledge. The
objective is simple: predict the next token. Trained on 1–15 trillion tokens of web text, code,
books, and scientific papers, the model implicitly learns grammar, facts, reasoning patterns, and
common sense from statistical patterns alone. Pre-training consumes the overwhelming majority of
the compute budget, often 90–97% of total training cost for a frontier model. Everything that
follows is fine-tuning a pre-trained base.

**② Supervised Fine-Tuning (SFT)** converts a base model into an assistant. The loss function is
the same cross-entropy as pre-training (Chapter 2), but applied only to *response tokens*:
the instruction is supplied as context, but its prediction error is discarded. This "response
masking" forces the model to learn how to *answer*, not how to continue arbitrary text. SFT is
fast: days of GPU time versus months for pre-training.

**③ The Reward Model (RM)** is trained on human *comparisons*, not demonstrations. An annotator
sees two responses to the same prompt and picks the better one. A separate transformer, the
reward model, learns to assign a scalar quality score to any (prompt, response) pair such that
it agrees with the annotator's choices. The reward model captures preferences that are easy to
compare but hard to demonstrate, and is used as a trainable signal in Stage ④.

**④ RLHF / PPO** is Reinforcement Learning from Human Feedback. The language model, now called
the *policy*, generates responses, the reward model scores them, and PPO (Proximal Policy
Optimization) updates the policy weights to maximize the reward. A KL divergence penalty
prevents the model from drifting too far from the SFT starting point. PPO is powerful but
expensive: it requires four models in memory simultaneously and complex engineering to stabilize.

**⑤ DPO (Direct Preference Optimization)** achieves RLHF's goal with a single supervised loss,
no reward model, and no RL loop. It uses the same pairwise preference data as Stage ③ but shows
that the optimal RLHF policy has a closed-form relationship to the SFT model, making alignment
a one-pass training problem. DPO dominates open-source alignment because it requires only
SFT-level infrastructure.

**⑥ Reasoning RL (GRPO / RLVR)** is the most recent stage to become widespread. Instead of
learning from human preference, the model is rewarded for *getting the answer right* on
verifiable tasks: math problems, code that passes tests, logic puzzles. No reward model is
trained. A deterministic checker provides the signal. Under this pressure, models spontaneously
learn to think step by step, backtrack, and self-verify. This is what produces the "reasoning
model" behavior of systems like o1, DeepSeek-R1, and Claude's extended thinking mode.

**⑦ Evaluation** is not a final step. It runs throughout the entire pipeline. Perplexity on
held-out text tracks pre-training progress. Human preference win-rates track alignment. Capability
benchmarks (MMLU, HumanEval, MATH) measure specific skills. Evaluation is also how teams catch
regressions: a stage that improves one capability often degrades another, and only continuous
measurement catches this.

## Where the compute goes

The stages are not equal in cost. Pre-training is the primary investment:

::: {.callout .note}
**Compute proportions (approximate, frontier model).** Pre-training: 90–97% of total GPU
hours, trillions of tokens, months of training. SFT: days. Reward model training: days. RLHF
or DPO: weeks. Reasoning RL: weeks to months for full GRPO runs, with long rollout sequences
(8k–32k tokens each). Evaluation: continuous but small overhead. The compute budget determines
the quality ceiling. Fine-tuning stages shape the behavior within that ceiling.
:::

::: {.figure}
![](assets/figures/ch12/fig-compute-budget.svg)
:::

::: {.caption}
**Figure 12.2.** Where the compute goes. Pre-training consumes almost all of a frontier model's GPU-hours. Every post-training stage combined is a thin slice on top (expanded below), which is why the pre-trained base sets the quality ceiling.
:::

This asymmetry is why the quality of a model is largely determined by its pre-training data and
scale. You cannot fine-tune a weak base into a frontier model.

## A toolbox, not a pipeline

The numbered stages are a teaching sequence, not a fixed recipe. Modern post-training pipelines
mix and reorder them freely:

- Some labs skip a separate reward model entirely (DPO or GRPO sidestep Stage ③).
- Some skip SFT before reasoning RL entirely. DeepSeek-R1-Zero ran pure GRPO on a base model
  with no SFT and developed coherent reasoning from scratch.
- Most frontier models now combine multiple passes of DPO or RLHF with at least one round of
  reasoning RL.
- Evaluation runs continuously throughout, not once at the end.

::: {.callout .idea}
Think of the seven stages as a toolkit. Each tool teaches the model a different thing:
pre-training installs knowledge, SFT installs format, preference optimization installs taste, and
reasoning RL installs the ability to think harder on hard problems. Which tools you use, and in
what order, depends on the task.
:::

## The objective functions, in one place

All the training losses in this part come back to two building blocks from Chapter 2: the
**cross-entropy loss** and the **KL divergence**. For quick reference:

| Stage | Objective (informal) | Loss |
|---|---|---|
| Pre-training (①) | Minimize surprise at the next token | Cross-entropy averaged over all positions |
| SFT (②) | Minimize surprise at response tokens only | Cross-entropy masked to response tokens |
| Reward model (③) | Score winner above loser | Bradley-Terry pairwise loss: −log σ(r_w − r_l) |
| RLHF/PPO (④) | Maximize reward − KL from reference | PPO policy gradient + KL penalty |
| DPO (⑤) | Widen preferred-vs-rejected gap | −log σ(β·Δlog ratio_w − β·Δlog ratio_l) |
| GRPO (⑥) | Reinforce correct reasoning paths | Clipped PPO gradient + KL, group-relative baseline |
| Evaluation (⑦) | *Not a loss, measures the result* | Perplexity, benchmark accuracy, win-rate |

::: {.caption}
**Table 12.2.** The objective function at each stage of the pipeline.
:::

Cross-entropy: Chapter 2. KL divergence: Chapter 2. Each loss is derived in its chapter.

## Summary

- Modern LLMs are produced by a **multi-stage training pipeline**, not a single run. Pre-training
  provides knowledge and language fluency. Subsequent stages shape behavior, alignment, and
  reasoning capability.
- **Pre-training dominates the compute budget.** Everything else is fine-tuning within the
  quality ceiling the pre-trained base sets.
- The stages are a **toolbox**: labs combine, skip, and reorder them based on their data,
  infrastructure, and objectives. The numbered sequence here is a teaching order, not a law.
- All the loss functions in Part IV are built from **cross-entropy** and **KL divergence**, both
  already covered in Chapter 2.

> **Coming up:** Chapter 13 goes inside Stage ①, the most computationally intensive and arguably
> the most important stage: pre-training. The data, the scaling laws, the learning rate schedules,
> and what exactly the model learns from trillions of next-token predictions.
