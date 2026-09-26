# Running Models Offline

Every chapter in this book has assumed an API call, a prompt sent over a network to a remote
model, a response streamed back. That works well for most applications, but it is not the only
option. You can run capable language models on your own hardware (a laptop, a workstation, a
private server) with no internet connection, no API key, and no data leaving the machine.

This matters for several reasons: data privacy (code, documents, and conversations stay local),
cost (no per-token billing after setup), offline capability (reliable in environments without
network access), and transparency (you know exactly which model you are running).

This chapter covers how: what quantization is and why it is the key enabler for CPU inference,
the GGUF file format and the Ollama runtime that makes local deployment practical, how to select
a model for local use, and the tooling options for integrating a local model into a daily
workflow.

**In this chapter**

- Why local inference is now practical, and why it wasn't before.
- Quantization: reducing weight precision without destroying quality.
- GGUF and llama.cpp: the CPU-inference stack.
- Ollama: pulling, managing, and serving models with one command.
- Choosing the right model and quantization level.
- Integration: CLI, IDE assistant, browser-based chat, agentic tools.
- Realistic performance expectations and what to tune.

## Why local inference is now practical

As recently as 2024, running a useful language model locally required specialized hardware. The smallest
models that could follow instructions reliably had 7 billion or more parameters, storing each
as a 16-bit floating-point number: 7B × 2 bytes = 14 GB of RAM just for the weights, before the
context window, activations, or operating system overhead. A typical laptop with 8–16 GB RAM
could not fit one.

Three developments changed this:

**Smaller models became capable.** The "Phi" series (Microsoft), Gemma (Google), Qwen (Alibaba),
and Llama (Meta) demonstrated that 3B–7B parameter models, trained carefully on high-quality
data, can follow instructions, write code, and reason: tasks that previously required models
ten times larger.

**Quantization matured.** Reducing each weight from 16 bits to 4 bits shrinks a 7B model from
14 GB to a little over 4 GB, a bit more than 3× smaller on disk, with quality that is hard to
distinguish from the original on most tasks. The key insight
from the GGUF/llama.cpp ecosystem is that not all weights are equally important: K-quant
techniques preserve precision on the layers that matter most and compress the rest aggressively.

**CPU inference became fast enough.** The bottleneck in generating a token is not arithmetic, it is
moving the weights. Every weight has to travel from RAM into the CPU once per token, so a model's
speed is roughly its memory bandwidth divided by its size on disk. Shrinking weights from 16 bits to
4 bits cuts that traffic by about 3.4× and so roughly triples the token rate. Modern vector instructions
(AVX2, and AVX-512 on server and AMD parts) and multi-core parallelism close the rest of the gap.
A 3B model at Q4_K_M achieves 25–30 tokens per second on a laptop CPU. At roughly 1.3 tokens per
English word that is over 1,000 words per minute, several times faster than you can read it. Not
GPU-fast, but well past the point where speed is the bottleneck.

::: {.callout .plain}
Here is the intuition that explains almost every number in this chapter. To produce one token, your
CPU has to read the entire model once. A 4 GB model on a laptop with 50 GB/s of memory bandwidth
therefore tops out around 12 tokens per second, no matter how fast the processor is. That is why
making the model smaller makes it faster, and why a 70B model is slow even on a machine with enough
RAM to hold it.
:::

::: {.figure}
![](assets/figures/ch20/fig-bandwidth-wall.svg)
:::

::: {.caption}
**Figure 20.1.** The memory-bandwidth wall. To generate one token the CPU reads the whole model from RAM, so the token rate is roughly memory bandwidth divided by model size. Shrinking the weights is what makes local inference fast.
:::

## Quantization: the key to fitting models in RAM

Quantization reduces the numerical precision of model weights. The original pre-trained weights
are stored in 16-bit floating point (FP16), two bytes per number. Quantization maps them to
lower-precision representations:

| Format | Nominal bits per weight | 7B model size | Practical quality | CPU speed (typical) |
|---|---|---|---|---|
| FP16 (original) | 16 | ~14 GB | reference | baseline |
| Q8_0 | 8 | ~7.5 GB | indistinguishable | slow |
| Q6_K | 6 | ~5.8 GB | indistinguishable | moderate |
| **Q5_K_M** | **5** | **~4.9 GB** | **near-reference** | **good** |
| **Q4_K_M** | **4** | **~4.1 GB** | **slight degradation, rarely noticeable in chat** | **fast** |
| Q3_K_M | 3 | ~3.3 GB | visible on code and factual recall | very fast |
| Q2_K | 2 | ~2.7 GB | clearly degraded | fastest |

::: {.caption}
**Table 20.1.** GGUF quantization formats: size, quality and speed for a 7B model.
:::

::: {.callout .note}
**Why the sizes don't divide evenly.** A Q4_K_M file is not exactly half the size of a Q8_0 one.
K-quants store weights in blocks, and each block carries a scale and an offset in higher precision
alongside the 4-bit values. Averaged out, Q4_K_M costs about 4.8 bits per weight rather than 4,
which is why a 7B model lands near 4.1 GB instead of the 3.5 GB the nominal arithmetic suggests. Use
the size column, not the bits column, when you are checking whether a model fits in RAM.
:::

The recommended sweet spot for most local use is **Q4_K_M**: fast CPU inference, comfortable RAM
usage on 8–16 GB machines, and a quality drop most people never notice outside code generation.

::: {.callout .idea}
The "K" in Q4_K_M stands for "k-quant", a mixed-precision quantization (scheme) that stores most weights at the
nominal bit width but keeps a few sensitive tensors (such as the attention values and the
feed-forward down-projection) at higher precision, using a fixed hand-designed allocation. Compared
to the older legacy formats (Q4_0, Q4_1), k-quants give the same file size with noticeably
better quality. Always prefer k-quants unless you have a specific reason not to.
:::

::: {.figure}
![](assets/figures/ch20/fig-quant-tradeoff.svg)
:::

::: {.caption}
**Figure 20.2.** The quantization trade-off for a 7B model. Quality holds up well down to about 4 bits, then falls off quickly. Q4_K_M and Q5_K_M are the usual sweet spot: most of the size saving, little of the quality loss.
:::

### What quantization loses

Quantization is lossy compression. The weight values are rounded to the nearest representable
value in the lower-precision grid. The loss is small at Q4 (~5%) but meaningful at Q2 (~25%).
On code generation and factual recall tasks, the degradation is more noticeable than on
open-ended chat. Prefer Q5_K_M or higher if coding quality matters.

::: {.callout .caution}
**Not all quantization formats are interchangeable.** AWQ and GPTQ are 4-bit formats designed
for GPU inference with libraries like vLLM. On a CPU-only machine they are not usable:
you need GGUF format models specifically. Always check the model's format before downloading.
:::

## GGUF and the CPU-inference stack

**GGUF** is the model file format for CPU inference. A GGUF file is
self-contained: it holds the quantized weights, the tokenizer, and all metadata needed to run
the model. You download one file and you have everything.

**llama.cpp** (Georgi Gerganov, 2023) is the C++ inference engine that reads GGUF files and
runs the transformer forward pass efficiently on CPU. It uses SIMD vectorization (AVX2/AVX-512, applying one instruction to many numbers at once),
exploits memory-mapped files to reduce RAM usage, and parallelizes across CPU cores. Ollama,
LM Studio, and most local deployment tools are built on llama.cpp.

**Ollama** wraps llama.cpp in a convenient CLI and a REST API that mimics the OpenAI chat
completions format. It handles model downloading, version management, server lifecycle, and
provides an endpoint at `http://localhost:11434` that any OpenAI-compatible client can speak to.

## Ollama: pull, serve, run

Installation is a single command:

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS: brew install ollama  (CLI only — the desktop app is a separate download)
# Windows: download installer from ollama.com
```

Once installed, Ollama starts a local server at `http://localhost:11434`. Model management
mirrors docker:

```bash
ollama pull qwen2.5-coder:7b   # download a model (GGUF, Q4_K_M by default)
ollama list                     # show downloaded models with sizes
ollama show qwen2.5-coder:7b   # details: quantization, context length, params
ollama rm qwen2.5-coder:7b     # delete model to free disk space
ollama ps                       # show currently loaded models and RAM usage
```

Run a model interactively:

```bash
ollama run qwen2.5-coder:7b
>>> Create a Python function that validates an email address
```

Or call it programmatically via the OpenAI-compatible REST API:

```python
from openai import OpenAI

# Point the OpenAI client at the local Ollama server
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="qwen2.5-coder:7b",
    messages=[{"role": "user", "content": "Write a Python class for a binary search tree"}],
)
print(response.choices[0].message.content)
```

Because Ollama speaks the OpenAI API, any library or tool that supports OpenAI (LangChain,
the Claude Agent SDK via `base_url` override, Open WebUI, Continue) can point at a local model
with a two-line configuration change.

## Choosing a model

The key parameters for local model selection are: parameter count (determines capability),
quantization level (determines memory and speed), and context length (determines how much text
the model can consider at once).

| Task | Recommended model | Size at Q4_K_M | CPU speed (typical) |
|---|---|---|---|
| Fast autocomplete | `qwen2.5-coder:3b` | 1.9 GB | 25–30 tok/s |
| Code generation | `qwen2.5-coder:7b` | 4.1 GB | 12–15 tok/s |
| Writing / general chat | `phi4-mini` | 2.2 GB | 25–30 tok/s |
| Fast chat | `gemma3:4b` | 2.5 GB | 25–30 tok/s |
| High quality, slow | `qwen3-coder:30b` | ~18 GB | 4–6 tok/s |

::: {.caption}
**Table 20.2.** Recommended local models by task, with measured CPU throughput.
:::

::: {.callout .note}
**The Mixture-of-Experts entry.** `qwen3-coder:30b` is an MoE model (~30B total, ~3B active per token,
Chapter 9). On CPU only the active experts stream for each token, so it runs faster than its 18 GB
on-disk size would suggest under the bandwidth rule above, though the whole file must still fit in RAM.
The other rows are dense models, where speed tracks size directly.
:::

::: {.callout .note}
**Model naming.** `ollama pull qwen2.5-coder:7b` pulls the 7B instruct model at the default
quantization (usually Q4_K_M). To request a specific quantization:
`ollama pull qwen2.5-coder:7b-instruct-q5_K_M`. Check `ollama show <model>` to confirm
what you got. Model names and default tags change between Ollama releases. Verify on
`ollama.com/library` for the current tag format.
:::

### Performance tuning

CPU inference speed scales with available cores, and Ollama already defaults to your physical core
count, which is the right answer for nearly everyone. (Hyperthreads rarely help, because the work is
memory-bound rather than compute-bound.) If you do need to override it, thread count is a *model*
parameter rather than an environment variable: `PARAMETER num_thread N` in a Modelfile, or
`/set parameter num_thread N` in an interactive session.

```bash
export OLLAMA_NUM_PARALLEL=1   # run one model at a time (prevents thrashing)
```

Memory is the binding constraint. Before running a model:

- Check free RAM: `free -h` (Linux), Activity Monitor (macOS), Task Manager (Windows)
- Close RAM-heavy applications: browsers with many tabs, Docker containers, Electron apps
- Target at least 8 GB free, covering the model weights plus context KV cache

**The KV cache and context length.** The KV cache (Chapter 8) stores intermediate attention
keys and values so the model does not recompute them for each new token. On CPU, this lives
in RAM and grows proportionally with context length. Doubling the context from 4k to 8k tokens
roughly doubles the KV cache size, on top of the already-large model weights. Raise the context
length only as far as your task actually requires.

Set the context window explicitly when you load a model (`OLLAMA_CONTEXT_LENGTH`, or
`/set parameter num_ctx 32768` interactively) and confirm what you got with `ollama ps`. Ollama will
not warn you when a request overflows it. It silently drops the oldest messages, and a long agent
session simply forgets what it already did.

**Working out the cost.** The KV cache is linear in context length, and you can compute it directly:

```
KV cache = 2 (keys and values)
         × layers × kv_heads × head_dim
         × bytes_per_value
         × context_length

Qwen2.5-Coder 7B: 28 layers, 4 KV heads (GQA), head_dim 128, 2 bytes (fp16)

  per token = 2 × 28 × 4 × 128 × 2 = 57,344 bytes ≈ 56 KB

  at  4k context:  56 KB × 4,096   = 0.22 GB
  at  8k context:  56 KB × 8,192   = 0.44 GB
  at 32k context:  56 KB × 32,768  = 1.75 GB

  So a 4.1 GB model at 32k context needs ~5.9 GB before the OS takes its share.
  On a 16 GB laptop that is comfortable; on 8 GB it is not.
```

This is also why grouped-query attention (Chapters 5 and 9) matters so much for local inference. An
older model with full multi-head attention at the same size would need 392 KB per token, seven times
more.

### Beyond CPU-only

Two things change the picture, and both are common on the machines readers actually own.

**Partial GPU offload.** Llama.cpp and Ollama will put as many layers as fit onto a GPU and run the
rest on the CPU. This is a spectrum, not a binary: even an 8 GB laptop GPU taking half the layers of
a 7B model gives a large speedup, because those layers stream from much faster memory. If you have
any discrete GPU, it is worth using.

**Apple Silicon unified memory.** On an M-series Mac the GPU addresses system RAM directly, so a
32 GB MacBook comfortably runs models that no x86 laptop with the same RAM can. The usual
"weights must fit in VRAM" rule does not apply.

If you have a real GPU and want to serve more than one user, **vLLM** is the better tool: it targets
GPUs, batches many requests continuously, and reads GPTQ and AWQ formats. llama.cpp and Ollama are
built for the single-user, CPU-or-mixed case, and read GGUF. That is the split the AWQ and GPTQ note
above is pointing at.

## Integration options

Four interfaces suit different workflows:

**Ollama CLI.** Simplest, no setup beyond Ollama itself. Good for quick questions, batch scripts,
and piped workflows. No conversation history, no GUI:

```bash
ollama run phi4-mini "Explain the difference between TCP and UDP in two sentences"

# Pipe-friendly for batch processing
cat requirements.txt | ollama run qwen2.5-coder:7b "Implement these features"
```

**VSCode Continue.** Free, open-source IDE extension that connects to Ollama. Provides inline
autocomplete, a chat panel, and `@codebase` context that searches your entire project. The best
daily-driver experience for developers:

```yaml
# ~/.continue/config.yaml (current format)
models:
  - name: Qwen2.5-Coder 7B
    provider: ollama
    model: qwen2.5-coder:7b
    roles: [chat, edit]
  - name: Qwen2.5-Coder 3B
    provider: ollama
    model: qwen2.5-coder:3b
    roles: [autocomplete]
  - name: nomic-embed-text
    provider: ollama
    model: nomic-embed-text
    roles: [embed]
```

**Open WebUI.** A browser-based ChatGPT-style interface for Ollama. Conversation history,
file uploads with RAG, multi-user support, and model switching. Good for non-technical users
or when you want a polished GUI without installing a desktop app:

```bash
pip install open-webui
open-webui serve   # then open http://localhost:8080
```

**Agentic coding tools.** Tools that use the OpenAI-compatible API (like Continue, the `openai`
Python SDK, or any tool with a configurable base URL) can point at `http://localhost:11434/v1`
directly. Tools that expect Anthropic's API format require a translation proxy such as LiteLLM
(`litellm proxy --model ollama/qwen2.5-coder:7b`), which re-exposes the model on a local port
with the correct schema. Note that agentic use requires large context windows (32k+ tokens) and
is significantly slower on CPU than cloud models.

::: {.callout .caution}
**Context length requirements for agents.** Agentic coding tasks accumulate tool outputs, file
contents, and reasoning traces. A task that touches three files and runs four tests can consume
20k–40k tokens of context. Models with smaller context windows (8k) will silently truncate
earlier context, causing the agent to "forget" what it already did. For agentic work, use a
model with at least 32k context and monitor the context fill via `ollama ps`.
:::

## What works well, and what to expect

**Works well:**

- Code explanation and documentation on functions and files
- Answering technical questions from provided context
- Code generation for well-specified, self-contained functions
- Simple refactoring and test generation
- Draft writing and editing

**Slower but feasible:**

- Multi-file code generation (expect 2–10 minutes per task on CPU)
- Long document analysis (requires large context, so check RAM headroom)
- Autonomous agentic editing sessions

**Not practical on CPU-only hardware:**

- 70B+ parameter models (40+ GB even at Q4, under 3 tokens/second)
- Real-time autocomplete with 7B models (latency too high to feel responsive)
- Bulk batch inference at scale

A 3B model at Q4_K_M generating 25 tok/s produces about 1,100 words per minute, so output arrives
faster than you read it. A 7B model at 12–15 tok/s is around 550–690 words per minute, still
comfortably ahead of reading speed. The threshold where CPU inference starts to *feel* slow is
roughly 10 tok/s (about 460 words per minute). Below that, prefer a smaller model or a lighter
quantization.

## Summary

- **Local inference** is now practical because small, capable models (3B–7B parameters)
  exist and quantization brings their memory requirements under 5 GB.
- **Quantization** reduces weight precision from FP16 (2 bytes) to Q4 (~0.5 bytes), shrinking
  a 7B model from 14 GB to ~4 GB, with a quality drop most people never notice outside code
  generation. Q4_K_M is the recommended sweet spot, with Q5_K_M for quality-sensitive tasks like coding.
- **GGUF** is the file format. **llama.cpp** is the inference engine. **Ollama** wraps both
  in a convenient CLI and an OpenAI-compatible REST API at `http://localhost:11434`.
- **Model selection:** 3B models for speed (autocomplete, quick queries); 7B for quality
  (code generation, analysis). Match context length to your task. Larger contexts cost RAM.
- **Integration options:** Ollama CLI for scripts, VSCode Continue for daily coding, Open WebUI
  for a browser-based GUI, agentic CLI tools for autonomous editing.
- **Realistic expectations:** 10–30 tok/s for the 3B–7B models this chapter recommends. Larger
  models fall well below that. Close memory-hungry applications, offload to a GPU if you have one,
  and prefer smaller models over slower larger ones.

> **What comes next:** The appendices collect the reference material the chapters pointed to —
> a full glossary of terms, a lineage of model families and parameter counts, and an annotated
> bibliography of the papers and resources this book draws from.
