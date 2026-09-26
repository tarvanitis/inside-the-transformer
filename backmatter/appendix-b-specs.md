::: {.frontmatter}

# Appendix B: Model Specifications and Architecture Evolution

## Frontier model specifications

Architecture dimensions for selected models. Closed models (GPT-4, Claude) are estimated from
published benchmarks and community reverse-engineering, so treat them as approximate.

### Dense models

| Model | Layers | d_model | Q heads | KV heads | Vocab | Context |
|---|---|---|---|---|---|---|
| GPT-2 (2019, 124M) | 12 | 768 | 12 | 12 | 50K | 1K |
| GPT-3 (2020, 175B) | 96 | 12,288 | 96 | 96 | 50K | 2K |
| Llama 2 7B (2023) | 32 | 4,096 | 32 | 32 | 32K | 4K |
| Mistral 7B (2023) | 32 | 4,096 | 32 | 8 | 32K | 32K |
| Llama 3.1 8B (2024) | 32 | 4,096 | 32 | 8 | 128K | 128K |
| Llama 3.1 70B (2024) | 80 | 8,192 | 64 | 8 | 128K | 128K |
| GPT-4 (est., 2023) | ~120 | ~12K | ~96 | GQA | 100K | 128K |
| Claude 3 Opus (est., 2024) | ~100+ | ~8K+ | ~64+ | GQA | ~100K | 200K |

::: {.caption}
**Table B.1.** Dense model specifications, GPT-2 through GPT-4.
:::

### Mixture of Experts (MoE) models

MoE models have *total* and *active* parameter counts. Only active parameters run per token,
making inference cheaper than the total count suggests.

| Model | Total params | Active params | Experts | Context | Released |
|---|---|---|---|---|---|
| Mixtral 8×7B (2023) | 47B | 13B | 8 (top-2) | 32K | Dec 2023 |
| DeepSeek-V3 (2024) | 671B | 37B | 256 (top-8) | 128K | Dec 2024 |
| DeepSeek-R1 (2025) | 671B | 37B | 256 (top-8) | 128K | Jan 2025 |
| Llama 4 Scout (2025) | 109B | 17B | 16 | 10M | Apr 2025 |
| Llama 4 Maverick (2025) | ~400B | 17B | 128 | 1M | Apr 2025 |
| Qwen3-235B-A22B (2025) | 235B | 22B | 128 | 128K+ | May 2025 |
| DeepSeek-V4 (2026) | 1.6T | 49B | fine-grained (top-8) | 1M | Apr 2026 |
| GLM-5.2 (2026) | ~744B | ~40B | fine-grained | 1M | Jun 2026 |
| Kimi K3 (2026) | 2.8T | 104B | fine-grained | 1M | Jul 2026 |
| Qwen3.8-2.4T-A95B (2026) | 2.4T | 95B | fine-grained | 262K | Aug 2026 |

::: {.caption}
**Table B.2.** Mixture-of-Experts model specifications.
:::

This table is restricted to models whose architecture details have been published.

### GPT-2-scale reference model

The running reference model for **Part II**'s worked examples and shape annotations:

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Vocabulary size | V | 50,000 | Total number of possible tokens |
| Model dimension | d_model | 768 | Embedding and hidden state size |
| Attention heads | h | 12 | Parallel attention mechanisms |
| Head dimension | d_k | 64 | d_model / h |
| FFN dimension | d_ff | 3,072 | 4 × d_model |
| Layers | N | 12 | Transformer block depth |
| Parameters |  | ~124M | Total (excluding tied LM head) |
| Sequence length | T | varies | Number of tokens in the current sequence |

::: {.caption}
**Table B.3.** The Part II reference model: GPT-2-scale dimensions.
:::

Part III's two builds use smaller configurations, chosen so they train in seconds on a laptop:

| | Chapter 10 (GPT) | Chapter 11 (encoder-decoder) |
|---|---|---|
| Vocabulary | 50,257 | 32,000 |
| d_model | 384 | 64 |
| Heads | 6 | 4 |
| Layers | 6 | 2 |
| Context | 256 | 512 |
| Parameters | ~10.8M | ~6.4M |

::: {.caption}
**Table B.4.** The two Part III build configurations.
:::

## Architecture evolution: 2017 to 2026

| Component | Original Transformer (2017) | Modern LLM (2026) |
|---|---|---|
| Architecture | Encoder-decoder | Decoder-only |
| Position encoding | Sinusoidal (fixed) | RoPE (+ YaRN/LongRoPE for extension) |
| Normalization | LayerNorm, post-norm | RMSNorm, pre-norm |
| Attention | Multi-head (full KV per head) | Grouped-query (MLA in DeepSeek) |
| FFN activation | ReLU | SwiGLU |
| FFN width | 4× d_model | ~2.67× d_model (SwiGLU parameter match) |
| Attention implementation | Standard (O(n²) memory) | FlashAttention-2/3 |
| Sparsity | Dense (all parameters active) | MoE (fraction active per token) |
| Context length | ~512 tokens | 128K–1M+ tokens (10M on Llama 4 Scout) |
| Inference compute | Fixed per token | Variable: reasoning / test-time scaling |
| Smallest capable model | ~125M | ~1–3B |
| Largest model | 213M (Transformer-big; the 65M base model is the one usually quoted) | 1T+ total (MoE) |

::: {.caption}
**Table B.5.** Architecture evolution, 2017 to 2026.
:::

## Frontier model families (as of 2026-09-14)

A snapshot of publicly known model families, verified against vendor pages on 2026-09-14. Chapter 1
carries the same snapshot in prose. Version numbers change frequently, so check vendor pages for
current releases.

| Vendor | Model family | Notable capabilities |
|---|---|---|
| Anthropic | Claude 5: Fable 5.1, Opus 5, Sonnet 5, Haiku 4.5 | Adaptive thinking; 1M context (200K on Haiku) |
| OpenAI | GPT-6 Astra; GPT-5.6 (Sol / Terra / Luna) | Reasoning modes; ~1M context |
| Google | Gemini 3 (3.8 Flash latest; 3.1 Pro heads the Pro line) | Multi-modal; 1M context |
| Google | Gemma 4 | Open weights; E2B / E4B / 26B MoE / 31B dense |
| Meta | Llama 4 Scout / Maverick | Open weights; MoE; 10M context (Scout). Meta's frontier work has moved to the proprietary Muse line |
| DeepSeek | DeepSeek-V4; R1 | Open weights; reasoning model; MoE; 1M context |
| Alibaba | Qwen3.8; Qwen3-Coder | Open weights; strong coding |
| Z.ai | GLM-5.2 | Open weights (MIT); tops the open-weight rankings |
| Moonshot | Kimi K3 | Open weights; 2.8T-parameter MoE; 1M context |
| xAI | Grok 4.6 | 500K context; strong coding and agentic work |
| Mistral | Mistral Large 3 / Medium 3.5 | 675B MoE (41B active); research license, not fully open weights |

::: {.caption}
**Table B.6.** Frontier model families as of 2026-09-14.
:::

## Key architectural identifiers

When reading a model card or paper, these fields identify the core architectural decisions:

- **Decoder-only vs encoder-decoder.** Decoder-only is the default for generative LLMs.
- **GQA / MQA / MLA.** KV head count. Fewer KV heads = lower inference memory.
- **RMSNorm + pre-norm.** The modern default. Older checkpoints may use LayerNorm + post-norm.
- **SwiGLU vs ReLU.** SwiGLU is standard since 2023. ReLU indicates an older model.
- **RoPE base theta.** The frequency parameter. A larger base = better long-context generalization.
- **FlashAttention.** Present in essentially all models from 2023 onward.
- **Total / active parameters.** If two numbers are given, the model is MoE.

:::
