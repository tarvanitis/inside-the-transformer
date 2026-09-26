# Learning from Human Preferences

SFT teaches format by example: here is a good response, learn to produce it. But the space of
possible responses is vast, and the qualities that separate a good response from a great one
(appropriate hedging, genuine helpfulness, the right level of detail) are hard to demonstrate
explicitly. It is far easier to show a human two responses and ask which is better.

This chapter covers the three stages built on that comparison signal: the reward model, which turns
those comparisons into a single quality score the model can train on; RLHF/PPO, which uses that score
to reshape the language model through reinforcement learning; and DPO, which reaches the same
destination with one ordinary training pass, no reinforcement loop, skipping the reward model
entirely.

**In this chapter**

- Stage ③, the reward model: Bradley-Terry loss, types of reward signal, and reward hacking.
- Stage ④, RLHF/PPO: the training loop, the KL leash, and four models in memory.
- Stage ⑤, DPO: the closed-form shortcut and its variants.
- When to use each: a practical decision guide.

This chapter assumes KL divergence, sigmoid, and cross-entropy from Chapter 2. Chapter 13 shows how cross-entropy is applied in pre-training.

## Stage ③ (The Reward Model)

### Why we need a reward model

Collecting enough (instruction, ideal response) demonstration pairs to teach every quality
dimension through SFT is prohibitively expensive. But annotators find it straightforward to
*compare* two responses: given the same prompt and two candidate answers, which is better? This
preference signal is cheap to collect, scalable to millions of comparisons, and can encode
subtle quality dimensions (insight, calibration, appropriate tone) that are hard to
demonstrate but easy to recognize.

The reward model (RM) distills those comparison judgments into a single-number quality score for any
(prompt, response) pair. The score is *differentiable* (training can follow its gradient), so once
trained the RM can score responses on demand inside the training loop.

### Architecture

The reward model begins as a copy of the SFT model, so it has the same transformer weights and
sees the same token sequences. One change: the language-model head (`vocab_size` outputs) is
replaced with a **scalar head** (a single linear layer outputting one number per sequence). The
model reads the full (prompt, response) and produces a score r(x, y) ∈ ℝ.

### The Bradley-Terry loss

Training data is pairwise: for each prompt x, an annotator has chosen y_w ("won") over y_l
("lost"). The **Bradley-Terry model** is a classical ranking model that frames preference as
probability: the chance that a human prefers y_w over y_l is the sigmoid of the gap between their
scores, σ(r_w − r_l). A gap of zero means a coin flip, and a large positive gap means near-certainty.
(Bradley & Terry, 1952.) The training loss pushes the winner's score above the loser's:

$$L_{\text{RM}} = -\log\, \sigma\!\left(r(x, y_w) - r(x, y_l)\right)$$

Every symbol:

- **r(x, y).** The reward model's scalar quality score for response y to prompt x.
- **y_w / y_l.** The preferred ("won") and rejected ("lost") responses.
- **r(x, y_w) − r(x, y_l).** The score gap. We want this large and positive.
- **σ (sigmoid).** Squashes any real number to (0, 1). Here σ(gap) is the model's predicted
  probability that y_w is genuinely the better response.
- **−log σ(gap).** Near 0 when the gap is large and positive (clear correct ordering), grows
  large when the model has it backwards.

### Worked example: haiku preferences

Prompt: "Write a haiku about autumn." Two responses compared:

```
// Before training: RM has the pair backwards
r(x, y_w) = 0.3   ← preferred (more evocative)
r(x, y_l) = 0.7   ← rejected  (generic) — WRONG ordering!

diff     = 0.3 − 0.7 = −0.4      ← negative: preferred scored LOWER
σ(−0.4) = 1/(1 + e^0.4) = 0.401  ← only 40% confidence in correct ordering
L        = −log(0.401) = 0.913    ← high loss → large gradient

// After training: RM correctly ranks the responses
r(x, y_w) = 1.8   ← preferred, now higher ✓
r(x, y_l) = 0.4   ← rejected, now lower ✓

diff     = 1.8 − 0.4 = 1.4
σ(1.4)  = 1/(1 + e^{−1.4}) = 0.802   ← 80% confidence in correct ordering
L        = −log(0.802) = 0.220         ← low loss ✓
```

Before training the model had the pair backwards, with a negative gap and a loss of 0.91. After training
the gap is +1.4, the model is 80% confident it has the ordering right, and the loss has fallen
to 0.22.

::: {.figure}
![](assets/figures/ch15/fig-bradley-terry.svg)
:::

::: {.caption}
**Figure 15.1.** The Bradley-Terry model. The reward model's score gap between the winning and losing response is passed through a sigmoid to give the probability a human prefers the winner. Training widens the gap for correctly ordered pairs.
:::

### Types of reward model

The single-scalar preference RM is only one option. The modern landscape is richer:

| Type | Reward signal | Strength | Weakness |
|---|---|---|---|
| **Outcome RM (ORM)** | Scalar score on the final response | Simple, general | Noisy for long reasoning |
| **Process RM (PRM)** | Score per intermediate reasoning step | Dense feedback, better for multi-step math | Expensive: needs step-level labels |
| **LLM-as-judge** | Strong LLM prompted or fine-tuned to rate responses | Scales cheaply, handles nuanced criteria | Inherits the judge model's biases |
| **Verifier (rule-based)** | Deterministic check, correct/incorrect | No reward hacking, perfectly calibrated | Requires a checkable answer |

::: {.caption}
**Table 15.1.** Types of reward model, and the failure mode of each.
:::

::: {.callout .caution}
**Reward hacking.** Reward hacking means the policy finds responses that score high on the
reward model without being genuinely good, like a student who learns to please the marker
rather than understand the material. The RM is an imperfect proxy for human preferences, and
once a language model is trained to maximize its score (Stage ④), it exploits RM weaknesses:
generating very long, flattering, or structurally unusual responses that fool the scorer. This
is Goodhart's Law in action. The KL penalty in RLHF (next section) is the primary defense.
Verifiable reward models (Stage ⑥) sidestep the problem by replacing the learned proxy with a
hard correctness check.
:::

::: {.callout .note}
**RLAIF and Constitutional AI.** Collecting hundreds of thousands of human comparisons is slow
and expensive. **RLAIF (RL from AI Feedback)** replaces most human labels with judgments from
a capable LLM. Anthropic's **Constitutional AI** is the best-known instance: a written set of
principles guides a model to critique and revise its own outputs, generating the preference
pairs used for training with no human annotators in the loop. Studies find RLAIF can match
RLHF on tasks like summarization and dialogue at far lower cost. Most 2025-era labs use a
majority fraction of AI-generated preference data.
:::

---

## Stage ④ (RLHF / PPO)

### The training loop

With a trained reward model in hand, the language model is fine-tuned to produce responses that
score highly on it. The training uses **Proximal Policy Optimization (PPO)** (Schulman et al.,
OpenAI 2017), a reinforcement learning algorithm. PPO treats the language model as a *policy*, a probability distribution
over next tokens, and updates it to maximize expected reward.

::: {.figure}
![](assets/figures/ch15/fig-rlhf-loop.svg)
:::

::: {.caption}
**Figure 15.2.** The RLHF loop: policy, reward model, and the KL leash to the frozen reference.
:::

Each training step:

1. Sample a prompt x from the training distribution.
2. Run the policy (the LM) to generate a response y.
3. Score (x, y) with the frozen reward model: get r(x, y).
4. Compute how far the policy has drifted from the frozen reference model π_ref: the KL divergence.
5. Apply the PPO update: nudge the policy toward higher-reward responses, constrained by the KL term.

### The objective

::: {.callout .plain}
The 𝔼[…] below is the word "average" in mathematical clothing. It means: draw a prompt x from your
dataset, let the model write a response y, score it, then repeat thousands of times and take the
average score. You can never compute that average exactly, because there are far too many possible
responses, so training estimates it from the batch it actually sampled. Every objective in this
chapter and the next has that shape: maximize an average you can only ever estimate.
:::

$$\max_\theta\; \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot\mid x)}\!\left[\, r_\phi(x,y) \;-\; \beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \,\right]$$

Every term:

- **π_θ.** The policy: the language model being trained.
- **π_ref.** A *frozen* copy of the SFT model, the anchor the policy is not allowed to drift far
  from. (The SFT model of Stage ② is what fills this slot, and the book writes it π_ref throughout.)
- **r_φ(x, y).** The reward model's score for response y to prompt x.
- **log(π_θ / π_ref).** The per-sample form of the KL divergence: how far the policy's token
  distributions have moved from the reference's. Zero means identical, larger means more drift.
- **β.** The KL coefficient (typically 0.02–0.1), controlling how hard the leash pulls.
- **max_θ.** Adjust the weights to maximize this whole expression.

::: {.callout .idea}
The KL penalty is the leash that prevents reward hacking. Without it, the policy learns to
exploit RM weaknesses, generating sycophantic, padded, or structurally unusual text that
scores highly but is not genuinely good. The leash forces the model to stay close to the SFT
model's behavioral distribution while pushing toward higher-reward responses within that
neighborhood.
:::

::: {.figure}
![](assets/figures/ch15/fig-kl-leash.svg)
:::

::: {.caption}
**Figure 15.3.** The KL leash. RLHF pulls the policy toward higher reward but tethers it to the frozen reference with a KL penalty, keeping it from drifting into reward-hacked behavior.
:::

### Worked example: one PPO step

```
// Step 1: generate a rollout — one complete response, sampled from the policy
prompt x = "Explain black holes to a 10-year-old"
y = π_θ(x) → "Black holes are regions where gravity is so strong..."

// Step 2: score with the frozen reward model
r = r_φ(x, y) = 2.3   ← good response

// Step 3: KL penalty — how far has the policy drifted from the SFT reference?
// KL here is the per-token average over T positions of the response.
KL = (1/T) Σₜ log[ π_θ(yₜ|y<t) / π_ref(yₜ|y<t) ] = 0.18   ← slight drift

// Step 4: total reward signal (reward minus the leash)
total_reward = r − β·KL = 2.3 − 0.05 × 0.18 = 2.291

// Step 5: PPO policy gradient update
// For each token t in the response:
ratio_t  = π_θ(yₜ) / π_θ_old(yₜ)           ← how much did this token's prob change?
A_t      = return from t onward − V_ψ(s_t)   ← better or worse than the critic predicted?
L_clip_t = min(ratio_t · A_t, clip(ratio_t, 1−ε, 1+ε) · A_t)
// The clip prevents any single step from changing the policy too drastically.
```

::: {.callout .plain}
The value function is the model's own weather forecast. Before the policy has finished writing a
response, the critic looks at the half-finished text and guesses what score the reward model will
eventually give it. That guess is the baseline. If the finished response beats the forecast, the
tokens that got it there were a pleasant surprise, and the policy is nudged to produce them more
often. If it falls short, they are nudged down. The gap between what actually happened and what the
critic predicted is the **advantage**, and it is the advantage, not the raw reward, that drives
every update.
:::

Two symbols in that block are worth naming. **π_θ_old** is a snapshot of the policy taken before the
current batch of rollouts was generated. The ratio measures how far the live policy has moved from
the version that actually produced this data. **ε** is the clip range, conventionally 0.2, so a
single update can change any token's probability by at most ±20%.

The raw reward was 2.3. The policy paid a 0.009 KL toll for drifting from the reference, leaving
2.291 as the effective learning signal. The **advantage** Aₜ answers "was this particular token
better or worse than the average expected return at this step?" Positive advantages make a
token more likely next time, negative ones less likely. The PPO *clip* caps how large any single
update can be, which is what tames this notoriously unstable training regime.

### Four models in memory

PPO requires four neural networks running simultaneously:

| Model | Frozen? | Role |
|---|---|---|
| Policy π_θ | No, updated | The LM being aligned |
| Reference π_ref | Yes | KL anchor, prevents drift |
| Reward model r_φ | Yes | Scores each rollout |
| Value function V_ψ | No, updated | Estimates future reward (critic) |

::: {.caption}
**Table 15.2.** The four models PPO holds in memory simultaneously.
:::

All four must fit in GPU memory during training. For large models this requires careful
parallelism across many accelerators, a significant engineering burden that motivated the search
for simpler alternatives.

---

## Stage ⑤ (Direct Preference Optimization (DPO))

### The key insight

::: {.callout .plain}
Here is the whole idea before the algebra. RLHF trains a separate judge, the reward model, and then
runs a fragile loop with four models to teach the policy to please it. DPO asks a sharper question:
what if the model can be its own judge? A response is good to exactly the degree the model being
trained makes it *more* likely than the frozen starting model did. The "reward," in other words, is
already sitting inside the policy, in the gap between those two probabilities. Once you see that, the
separate reward model is redundant, and you can train straight from the preference pairs. That is what
*closed-form* means here: instead of hunting for the best policy step by painful step, the way PPO
does, you can write the answer down directly.
:::

DPO (Rafailov et al., Stanford 2023) shows that the optimal policy under the RLHF objective has
a *closed-form* relationship to the SFT reference model. This means the reward model and the PPO
loop are not necessary. The LM can be trained directly on preference pairs with a single
supervised loss.

The insight: the implicit reward encoded in the RLHF objective can be written as:

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \text{const}$$

::: {.callout .deepdive}
**Where the constant goes.** That "const" is really β log Z(x), a normalizing term that depends on
the prompt x but not on the response y. That is the whole trick. When both responses answer the same
prompt, the Bradley-Terry loss only ever uses the difference r(x, y_w) − r(x, y_l), and two identical
constants cancel exactly. The intractable term disappears, and what is left is the DPO loss,
computable from four log-probabilities and nothing else.
:::

Note that this β plays the same mathematical role as the KL coefficient in the RLHF objective above,
falling out of the same derivation rather than being a genuinely new knob. In practice it is retuned
to a larger effective range (DPO is typically 0.1–0.5, against RLHF's 0.02–0.1).

The reward model of Stage ③ is *already implicit in the language model itself*. Substituting
this into the RLHF objective and simplifying yields the DPO loss:

$$L_{\text{DPO}} = -\log\, \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

Every term:

- **π_θ(y|x) / π_ref(y|x).** How much *more* (or less) likely the model being trained makes
  response y compared with the frozen SFT reference. This log-ratio *is* the implicit reward,
  the reward model of Stage ③, folded into the language model itself.
- **y_w / y_l.** Preferred and rejected responses, same as reward model training.
- **β.** How far the model is allowed to move from the reference per unit of preference (large
  β = stay close to SFT, small β lets it move further). Typically 0.1–0.5.
- **σ.** Sigmoid. The −log σ(gap) form shrinks as the preferred-vs-rejected gap grows.

**What DPO does in one sentence:** increase the probability of y_w relative to the reference,
decrease the probability of y_l relative to the reference, simultaneously, via one
differentiable loss, with no reward model, no PPO, no value function.

### Worked example: ethics preference pair

Prompt: "Is it ethical to lie?", two responses, β = 0.2.

```
// Log probability ratios: how does the current policy compare to the reference?
// Preferred response y_w: "That's a complex question with context-dependent answers..."
log π_θ(y_w|x)   = −12.3
log π_ref(y_w|x)  = −13.1
log ratio_w = −12.3 − (−13.1) = +0.8   ← policy likes y_w MORE than reference does

// Rejected response y_l: "Yes, lying is always fine."
log π_θ(y_l|x)   = −14.9
log π_ref(y_l|x)  = −14.2
log ratio_l = −14.9 − (−14.2) = −0.7   ← policy likes y_l LESS than reference does

// DPO gap (want large and positive)
gap   = β · log_ratio_w − β · log_ratio_l
      = 0.2 × (+0.8) − 0.2 × (−0.7)
      = 0.16 + 0.14 = +0.30   ← pointing the right direction ✓

σ(0.30) = 1/(1 + e^{−0.30}) = 0.574
L_DPO   = −log(0.574) = 0.554   ← moderate loss — correct ordering but gap could be wider
// At convergence: gap → large positive → L → 0
```

Both signals point the right way: the policy already makes the good answer more likely than the
reference (+0.8) and the bad answer less likely (−0.7). Training widens that gap further,
driving the loss toward 0.

### DPO vs RLHF

| Property | RLHF (PPO) | DPO |
|---|---|---|
| Reward model needed? | Yes, trained separately (Stage ③) | No, implicit in the loss |
| Models during training | 4 (policy, reference, RM, value function) | 2 (policy + frozen reference) |
| Online rollouts? | Yes, generates text during training | No, uses fixed offline preference data |
| Implementation complexity | Very high (custom RL infrastructure) | Low (identical infrastructure to SFT) |
| Compute cost | High | ~Same as SFT |
| Alignment quality | Often better on hard, subjective tasks | More conservative, closer to the reference |
| Representative users | GPT-4, Claude 2 | Zephyr-7B-β, Tulu 2/3, Qwen2.5-Instruct, Llama 3 Instruct |

::: {.caption}
**Table 15.3.** DPO against RLHF, property by property.
:::

::: {.callout .deepdive}
**DPO variants.** Several limitations of the basic DPO loss have spawned targeted fixes, each
now used in production:

- **IPO** (Identity Preference Optimization) replaces the log-sigmoid with a squared loss,
  preventing the unbounded-reward growth that causes DPO to overfit when preferences are
  *deterministic* (the same response always wins), which drives the implicit reward gap toward
  infinity and the KL regularization toward zero.
- **KTO** (Kahneman-Tversky Optimization) drops the requirement for pairs and uses single
  labeled good/bad responses with prospect-theory loss weighting. Useful when you only have
  thumbs-up/thumbs-down labels rather than comparison pairs.
- **SimPO.** Removes the reference model entirely, using the *length-normalized* average
  log-probability as the implicit reward plus a target margin. Reduces the tendency of standard
  DPO to inflate response length.
- **ORPO** (Odds-Ratio Preference Optimization) merges SFT and preference alignment into
  *one* training pass: standard SFT cross-entropy on the preferred response plus an odds-ratio
  penalty on the rejected one, with no reference model and no separate SFT stage. Roughly halves
  training memory.
:::

---

## Which to use: a decision guide

::: {.callout .plain}
There is no single correct choice. The method follows the data and the infrastructure.

**DPO (or a variant).** When you have a fixed preference dataset, want simple single-GPU
infrastructure, and do not need to generate new rollouts during training. Dominant for
open-source and mid-scale alignment.

**PPO (online RLHF).** When online rollouts against a learned reward model squeeze out quality
on hard, subjective tasks, justified when you have the engineering capacity to run four models
simultaneously. Still used at some frontier labs for the final alignment stage.

**GRPO / RLVR** (Chapter 16) is for when rewards are *verifiable* (math, code, agentic tasks). This
is now the default for training reasoning models. It drops the value network, uses a group of
sampled answers as the baseline, and is far more sample-efficient than PPO when correctness is
checkable.
:::

## Summary

- **The reward model** converts pairwise human comparisons into a differentiable scalar quality
  score, trained with the Bradley-Terry loss: −log σ(r_w − r_l). The RM can be a scalar head, a
  process model scoring each reasoning step, an LLM-as-judge, or a deterministic verifier.
- **RLHF/PPO** trains the language model to maximize RM score subject to a KL penalty that
  prevents reward hacking. It needs four models in memory and complex RL infrastructure.
- **DPO** reaches the RLHF goal via a closed-form supervised loss on the same preference pairs,
  needing only two models (policy + frozen reference). The implicit reward is the log-ratio of
  the policy's and reference's probabilities.
- **Variants** (IPO, KTO, SimPO, ORPO) fix specific failure modes of DPO: overfitting,
  missing pair labels, length inflation, and the need for a separate SFT pass.

> **Coming up:** Chapter 16 covers the most recent shift in training: Stage ⑥, where the
> model is rewarded not for matching human preferences but for *getting the answer right*.
> Verifiable rewards, GRPO, and the emergence of reasoning models.
