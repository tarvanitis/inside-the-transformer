::: {.frontmatter}

# Appendix A: Glossary

Concise definitions of the terms used throughout this book, grouped by topic. For the full
worked examples and intuitions, see the chapter references in parentheses.

## Math Notation

**∈ (element of).** "x ∈ ℝ" means x is a real number; "W ∈ ℝ^{m×n}" means W is a real-valued matrix with m rows and n columns. Used in papers to specify the type and shape of every variable.

**ℝ, ℝⁿ, ℝ^{m×n}.** ℝ is the set of all real numbers (scalars); ℝⁿ is n-dimensional vector space; ℝ^{m×n} is the set of all m×n matrices of reals.

**Σ (summation).** Σᵢ₌₁ⁿ xᵢ = x₁ + x₂ + … + xₙ. Used in loss functions, softmax, and attention score normalization.

**∂ (partial derivative).** The rate of change of a function with respect to one variable while holding all others fixed. The building block of backpropagation.

**∇ (gradient / nabla).** A vector of all partial derivatives of a scalar function with respect to its inputs. Points in the direction of steepest increase. Gradient descent moves opposite to it.

**‖x‖ (vector norm).** The length of vector x. The L2 norm ‖x‖₂ = √(x₁² + x₂² + … + xₙ²). Used in attention scaling and normalization layers.

**argmax.** The input value that produces the maximum output: argmax_x f(x) is the x at which f is largest. Greedy decoding selects the token with the highest logit via argmax.

**θ (theta, model parameters).** Conventionally denotes all learnable weights in a model. "Minimize L(θ)" means adjust all weights to reduce the loss.

**π (policy).** In reinforcement learning, the probability distribution over actions given a state. In RLHF and GRPO, the policy is the language model: π_θ(y|x) = probability of generating response y given prompt x.

**𝔼[·] (expectation).** The probability-weighted average of a random variable: 𝔼[X] = Σ x · P(X = x). The RLHF and GRPO objectives maximize expected reward under the policy distribution.

**KL divergence.** D_KL(P ‖ Q) = Σ P(x) log(P(x)/Q(x)). Measures how much distribution P differs from reference Q, and is always non-negative. Used as a regularizer in RLHF to prevent the policy from drifting too far from the reference model. (Ch 2, 15, 16)

## Math Foundations

**Softmax.** Converts a vector of real numbers (logits) into a probability distribution: softmax(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ. Used in attention score normalization and the LM head. (Ch 2)

**Sigmoid.** σ(x) = 1 / (1 + e⁻ˣ). Squashes any real number to (0, 1). Used in gating mechanisms (SwiGLU) and reward model scoring. (Ch 2)

**Cross-entropy loss.** L = −(1/T) Σₜ log P(yₜ | y<t). The standard training objective for language models. It measures surprise averaged over all token positions. (Ch 2, 13)

**Perplexity.** PPL = e^L where L is average cross-entropy. Measures how many equally-likely choices the model had at each token. Lower is better. A random model has PPL ≈ vocabulary size (~50,000), while frontier models reach PPL ≈ 4–8. (Ch 2, 13, 17)

**Cosine similarity.** (a · b) / (‖a‖ ‖b‖). Measures the angle between two vectors. 1 = identical direction, 0 = orthogonal, −1 = opposite direction. Used in retrieval (Ch 18) and the dot-product attention score (Ch 5).

**Backpropagation.** The algorithm that computes gradients of the loss with respect to every weight by applying the chain rule backwards through the computation graph. (Ch 2)

**Adam / AdamW.** Adaptive gradient optimizer that maintains a running mean (momentum) and running variance (second moment) for each parameter, giving per-parameter adaptive learning rates. AdamW adds decoupled weight decay. The standard optimizer for transformer training. (Ch 2, 13)

**Gradient clipping.** Scales the entire gradient vector down if its norm exceeds a threshold (typically 1.0). Prevents catastrophic weight updates from anomalously large gradients. (Ch 13)

## Architecture

**Transformer.** A neural network architecture based entirely on attention mechanisms (no recurrence). Modern generative LLMs are decoder-only. Encoder-only transformers remain standard for embedding models (Ch 18), and hybrid SSM-transformer designs are now a production class (Ch 9). (Ch 1, 5–9)

**Frontier model.** One of the current most capable, state-of-the-art LLMs: the largest, best-performing systems at the leading edge, as opposed to smaller or older models. Which models count as frontier changes constantly. Chapter 1 and Appendix B (Table B.6) give the current snapshot. (Ch 1)

**Token.** The atomic unit that a language model reads and writes. Typically a subword piece: "unbelievable" might be 3 tokens. (Ch 3)

**Tokenizer.** Converts text to token IDs and back. BPE (Byte Pair Encoding) is the most common algorithm. Vocabulary sizes range from ~32K to 200K tokens. (Ch 3)

**Embedding.** A learnable dense vector representation of a token. The embedding table maps each token ID to a d_model-dimensional vector. (Ch 3)

**Positional encoding.** Adds position information to token embeddings, since attention is order-agnostic. Modern models use RoPE (rotary position embeddings). (Ch 4)

**RoPE (Rotary Position Embedding).** Encodes position by rotating the query and key vectors by a position-dependent angle before the dot product. Enables relative position generalization and is the standard in modern LLMs. (Ch 4)

**Attention (Q/K/V).** The mechanism by which each token gathers information from other tokens. Queries (Q) ask questions; Keys (K) are matchable identities; Values (V) are the information retrieved. Score = softmax(QKᵀ / √d_k) · V. (Ch 5)

**Multi-head attention.** Runs h independent attention heads in parallel, each projecting Q/K/V to a lower dimension d_k = d_model / h. Outputs are concatenated and re-projected. Lets the model attend to multiple relationship types simultaneously. (Ch 5)

**Grouped-query attention (GQA).** Uses fewer K/V heads than Q heads (e.g. 8 K/V heads for 32 Q heads). Reduces KV-cache memory and bandwidth at inference with minimal quality loss. Used in Llama 3 and most modern models. (Ch 5)

**Causal mask.** Prevents a position from attending to any later position: position t can attend to positions 0 through t, its own position included. Implemented either as a lower-triangular matrix of 1s marking what to keep (as in Ch 11's `torch.tril`) or, equivalently, as an upper-triangular matrix of −∞ added to the scores before softmax. Required for autoregressive generation. (Ch 5)

**Feed-forward network (FFN).** Two linear projections with a non-linearity between them, applied position-wise after attention in each transformer block. Typically 4× wider than d_model. SwiGLU replaces ReLU in modern models. (Ch 6)

**SwiGLU.** FFN activation used in Llama and most modern models. Two parallel projections: one gated by a swish activation. Outperforms ReLU at the same parameter budget, but requires 2/3 × 4 = ~2.67× width to match parameter count. (Ch 6)

**Layer normalization (LayerNorm).** Normalizes each hidden state vector to zero mean and unit variance, then applies learnable scale γ and shift β. (Ch 6)

**RMSNorm.** A simpler, faster alternative to LayerNorm that normalizes by the root-mean-square of the vector (no mean subtraction). Used in Llama, Mistral, and most current models. (Ch 6)

**Residual connection (skip connection).** Adds the input of a sub-layer directly to its output: x + Sublayer(x). Allows gradients to flow unchanged through deep networks. All transformers use this around both attention and FFN sub-layers. (Ch 6)

**Pre-norm vs post-norm.** Whether layer normalization is applied before (pre-norm) or after (post-norm) each sub-layer. Pre-norm is the modern default. It stabilizes training and enables deeper models. (Ch 6)

**Logits.** The raw, unnormalized scores output by the final linear projection (LM head) before softmax. One logit per vocabulary token. Argmax gives greedy decoding. (Ch 8)

**d_model.** The dimensionality of token embeddings and hidden states throughout the model. Typical values: GPT-2 768, Llama 3 8B 4096, frontier models 8192+.

**Context window.** The maximum number of tokens a model can attend to in a single forward pass. Frontier models support 128K–1M+ tokens. (Ch 8)

**Mixture of Experts (MoE).** Architecture where each FFN is replaced by N "expert" FFNs and a router that selects the top-k for each token. Scales total parameters cheaply because only a fraction are active per token. (Ch 9)

**Encoder-decoder.** Architecture with a separate encoder (bidirectional attention) and decoder (causal attention + cross-attention). Original transformer architecture, now mainly used in translation and seq2seq tasks. (Ch 9, 11)

**Mechanistic interpretability.** The project of reverse-engineering the specific algorithms a trained model has learned, rather than treating it as a black box and only measuring its outputs. (Ch 7)

**Circuit.** A small subgraph of attention heads and FFN neurons that together implement one identifiable behavior, such as copying a repeated name. Circuits are found, not designed. They emerge from training. (Ch 7)

**Induction head.** The best-studied circuit: a two-layer mechanism that spots an earlier occurrence of the current token and predicts whatever followed it last time. A key mechanism behind in-context learning, and it tends to appear suddenly during training. (Ch 7)

**Residual stream.** The running vector that each transformer block reads from and adds back into, rather than replacing. Attention and FFN sub-layers communicate by writing into this shared channel. (Ch 6, 7)

## Training

**Pre-training.** Training a model from random initialization on trillions of tokens with the causal language modeling (next-token prediction) objective. Produces a base model with world knowledge but no instruction-following behavior. (Ch 13)

**Supervised Fine-Tuning (SFT).** Training on (instruction, response) pairs with the loss masked to response tokens only. Converts a base model into an instruction-following assistant. (Ch 14)

**Loss masking.** Setting instruction token positions to `ignore_index=-100` so `F.cross_entropy` skips them. Ensures gradients flow only through response tokens. (Ch 14)

**Teacher forcing.** At each training step, feeding the model the correct previous tokens as context rather than its own predictions. Makes training stable, but introduces "exposure bias" at inference. (Ch 14)

**Chat template.** A structured format with special tokens marking system prompt, user, and assistant turns. Different model families use different templates. Mixing them silently degrades quality. (Ch 14)

**LoRA (Low-Rank Adaptation).** Freezes original weights W and adds trainable matrices A (r × d_in) and B (d_out × r) with r ≪ d, so the update is B @ A. Reduces trainable parameters by 100–1000× for SFT. The LoRA adapter (50–200 MB) can be shipped separately from the full model. (Ch 14)

**Scaling laws.** Empirical relationships between model size, training tokens, compute, and loss. The Chinchilla law (Hoffmann et al. 2022) shows the compute-optimal ratio is ~20 training tokens per parameter. In practice, "overtrained" smaller models are preferred because inference is cheaper. (Ch 13)

**Learning rate schedule.** Warm-up for the first 1,000–2,000 steps (linear ramp from near-zero to peak LR), then cosine decay to ~10% of peak. Prevents chaotic early updates and enables fine-grained convergence. Pre-norm blocks (Ch 6) reduce how critical warm-up is, but production runs still use it. (Ch 13)

**Weight tying.** Sharing weights between the input embedding table and the output LM head projection. Reduces parameters by V × d_model (~525M for Llama 3 8B) and often improves performance. (Ch 8)

## Alignment

**Reward model (RM).** A model trained on human preference pairs (y_w preferred over y_l given prompt x) to predict a scalar reward. Uses the Bradley-Terry loss: L = −log σ(r(x, y_w) − r(x, y_l)). (Ch 15)

**RLHF (Reinforcement Learning from Human Feedback).** Training the LM policy to maximize reward model scores subject to a KL divergence constraint keeping the policy near the SFT model. PPO is the standard RL algorithm. (Ch 15)

**PPO (Proximal Policy Optimization).** Policy gradient algorithm that clips the importance ratio ρ = π_θ / π_old to (1−ε, 1+ε), preventing destructively large policy updates. Requires four models in memory: policy, reference, reward model, value network. (Ch 15)

**DPO (Direct Preference Optimization).** Closed-form alternative to RLHF that optimizes preference pairs directly without a learned reward model or RL loop. The implicit reward is the log-ratio of policy and reference probabilities. (Ch 15)

**KL leash.** The KL divergence penalty β · D_KL(π_θ ‖ π_ref) added to the RL objective. Prevents the policy from drifting so far from the reference SFT model that it forgets general knowledge. (Ch 15)

**Reward hacking.** Exploiting flaws in a proxy reward model to score highly without actually improving in the intended way. A known failure mode of RLHF when the reward model is imperfect. (Ch 15)

**RLVR (RL with Verifiable Rewards).** Training on tasks with checkable answers (math, code) using a deterministic verifier instead of a learned reward model. Sharply reduces reward hacking, though a verifier can still be gamed. (Ch 16)

**GRPO (Group Relative Policy Optimization).** RLVR algorithm that replaces PPO's learned value network with group-relative advantage: Â = (r − mean) / std within a group of G rollouts per prompt. Requires only 2–3 models in memory vs PPO's 4. (Ch 16)

**Constitutional AI (CAI).** Training approach (Anthropic) where a model is trained on its own self-critiques and revisions guided by a set of principles, reducing reliance on human labeling for harmlessness. (Ch 15)

**Sycophancy.** A failure mode where a model agrees with the user's stated position regardless of correctness. A known side-effect of optimizing for human approval. (Ch 15)

**Hallucination / confabulation.** Generating fluent, confident text that is factually incorrect. A fundamental failure mode arising from the model predicting plausible continuations rather than verified facts. (Ch 17)

## Inference

**Inference.** Running a trained model to produce output: the generation process itself, as opposed to training (which sets the weights). The model turns a prompt into output tokens one at a time (autoregressive generation). That output may take any form (prose, code, a list, JSON), so "inference" is the *how* and the generated text is the *what*. (Ch 8)

**Autoregressive generation.** Generating text one token at a time, feeding each generated token back as input for the next step. The standard decoding procedure. (Ch 8)

**KV-cache.** Caching the key and value tensors for all processed tokens so they need not be recomputed on each new token generation. Reduces per-step compute from O(n²) to O(n). (Ch 8)

**Temperature.** A scaling factor applied to logits before softmax: logits / τ. τ < 1 sharpens the distribution (more deterministic); τ > 1 flattens it (more random); τ = 0 gives greedy decoding. Written τ rather than T because T is the sequence length elsewhere in the book. (Ch 2, 8)

**Top-p sampling (nucleus sampling).** Sample only from the smallest set of tokens whose cumulative probability exceeds p (e.g. p = 0.9). Adapts the number of candidates to the local probability distribution. (Ch 8)

**Top-k sampling.** Sample from only the k highest-probability tokens (e.g. k = 50), discarding the rest. Simpler than top-p but uses a fixed candidate count. (Ch 8)

**Greedy decoding.** At each step, select the token with the highest probability. Deterministic and fast, but tends to produce repetitive text for long sequences. (Ch 8)

**Speculative decoding.** A small "draft" model generates k tokens, and a large "target" model verifies them in parallel. Accepted tokens are kept. The first rejected token triggers a correction. Achieves 2–3× latency improvement with identical output distribution. (Ch 9)

**FlashAttention.** Memory-efficient attention implementation that computes attention in tiles, keeping intermediate results in fast SRAM rather than writing to HBM. Makes long-context training and inference practical. (Ch 8)

**Prefill vs. decode phase.** Prefill: process all prompt tokens in one parallel forward pass (fast). Decode: generate one token at a time using the KV-cache (slower, memory-bandwidth-bound). (Ch 8)

**Test-time compute scaling.** Allocating more inference tokens (longer chain of thought, best-of-N sampling) to improve accuracy on hard tasks. The second scaling axis after parameter count and training tokens. (Ch 16)

**Prompt injection.** An attack in which instructions are smuggled into content the model reads (a web page, a document, a tool result) and the model follows them as if they came from the operator. There is no prompt that reliably prevents it. The mitigation is to keep untrusted-content-reading and privileged action in separate agents. (Ch 19)

**ANN (approximate nearest neighbor).** Index structures (flat, IVF, HNSW) that find *almost* the closest vectors by searching a fraction of the candidates. The fraction of genuine matches they miss is traded against latency. (Ch 18)

## Efficiency & Deployment

**Quantization.** Reducing weight precision from FP16 (2 bytes) to lower bit widths (e.g. 4-bit). A 7B model shrinks from ~14 GB to ~4 GB at Q4_K_M, with a quality drop most people never notice outside code generation and factual recall. Enables local CPU inference. (Ch 20)

**GGUF.** The file format for CPU-quantized models. Self-contained (weights + tokenizer + metadata). The native format for llama.cpp and Ollama. (Ch 20)

**GPTQ / AWQ.** GPU-oriented 4-bit quantization formats, read by GPU inference libraries such as vLLM. They target CUDA rather than CPU and are not interchangeable with GGUF, the format llama.cpp and Ollama use. (Ch 20)

**vLLM / PagedAttention.** High-throughput GPU inference server that manages the KV cache in fixed-size memory pages, enabling continuous batching and near-optimal GPU utilization. (reference entry: mentioned in Ch 8)

**Continuous batching.** Serving new requests without waiting for existing requests to complete by inserting them mid-generation. Improves GPU utilization in production inference. (reference entry: mentioned in Ch 8)

**GPU / VRAM.** Graphics Processing Units and their onboard memory. Model weights, activations, KV cache, and optimizer states all compete for VRAM. The primary resource constraint in LLM training. (Ch 2, 20)

**Mixed precision training.** Training with FP16/BF16 weights and activations but FP32 master weights and optimizer states. Halves memory for the forward/backward pass while retaining numerical stability. (Ch 10)

**Q4_K_M and the k-quants.** GGUF quantization formats that store weights in blocks, each carrying its own scale and offset in higher precision. This metadata is why a nominally 4-bit format costs closer to 4.8 bits per weight in practice. Q4_K_M is the usual default for local inference. (Ch 20)

**Ollama.** A tool that wraps llama.cpp with a model registry, a CLI modeled on docker, and an OpenAI-compatible REST API on `localhost:11434`. (Ch 20)

**llama.cpp.** The C/C++ inference engine behind most local LLM tooling, built for CPU and mixed CPU/GPU execution of GGUF models. (Ch 20)

## Fine-Tuning Methods

**LoRA.** See Training section above.

**QLoRA.** Quantized LoRA: base model weights are frozen at 4-bit precision while LoRA adapters are trained in 16-bit. Reduces VRAM for SFT of large models by another 3–4×.

**PEFT (Parameter-Efficient Fine-Tuning).** Umbrella term for methods that update only a small subset of parameters during fine-tuning: LoRA, prefix tuning, prompt tuning, adapter tuning.

**Full fine-tuning.** Updating all model parameters during SFT. Achieves the best quality but requires as much VRAM as pre-training.

## Evaluation

**Benchmark contamination.** A model scores artificially high on a benchmark because the test questions appeared verbatim in the pre-training corpus. The primary validity threat for most published benchmarks. (Ch 17)

**Goodhart's Law.** "When a measure becomes a target, it ceases to be a good measure." Once labs train specifically for a benchmark, it stops reflecting genuine capability. (Ch 17)

**MMLU.** Massive Multitask Language Understanding. 57 academic subject areas, multiple-choice. Now saturated, with frontier models scoring 90+, and superseded by MMLU-Pro and GPQA Diamond. (Ch 17)

**GSM8K.** Grade School Math, ~8,800 problems (7,473 train + 1,319 test). Word problems requiring multi-step arithmetic. Saturated for frontier models (95%+). (Ch 17)

**HumanEval.** 164 Python function completion tasks from docstrings, evaluated by pass@1. Saturated. SWE-bench Verified is the current coding standard. (Ch 17)

**LLM-as-judge.** Using a capable model to rate or compare responses. Scales cheaply but inherits the judge's biases. Always swap presentation order to counteract positional bias. (Ch 17)

**Chatbot Arena.** Crowdsourced human preference evaluation (LMArena.ai) using paired anonymous model battles and Elo scoring. The most credible real-world quality signal. (Ch 17)

**Pass@k.** The probability that at least one of k independent code generation attempts passes the test suite. Grows with k, and separates average quality from best-of-N quality. (Ch 17)

**Process Reward Model (PRM).** A reward model that scores individual reasoning steps rather than only the final answer. Enables denser feedback for long chains of thought and is used in best-of-N with RLVR. (Ch 15, 16)

**Verifier.** A deterministic checker (a test suite, a math evaluator, a parser) that returns ground-truth reward for a model's output. What RLVR substitutes for a learned reward model. (Ch 16)

**Value network (critic).** The model PPO trains alongside the policy to predict the return from a given partial response. Its prediction is the baseline the advantage is measured against. GRPO removes it and uses the group mean instead. (Ch 15, 16)

**Best-of-N.** Sample N responses, score each with a verifier or PRM, and return the highest scorer. The simplest form of test-time compute scaling. (Ch 16)

**Exposure bias.** The mismatch created by teacher forcing: during training the model always sees ground-truth history, but at inference it sees its own, possibly wrong, previous tokens. (Ch 14)

**SWE-bench / SWE-bench Verified.** A coding benchmark built from real GitHub issues, where a model must produce a patch that passes the repository's tests. "Verified" is the 500-task subset left after human screening removed underspecified and unfairly-tested items. (Ch 17)

**GPQA Diamond.** 198 PhD-level, deliberately "Google-proof" science questions, used as a frontier reasoning benchmark. (Ch 17)

**MMLU-Pro.** A harder, cleaned rebuild of MMLU with more answer options, reducing the advantage of guessing. (Ch 17)

**Elo.** A relative rating computed from pairwise win/loss records, borrowed from chess and used to rank models on Chatbot Arena. (Ch 17)

**Positional bias.** The tendency of an LLM judge to prefer whichever response it is shown first, independent of quality. Mitigated by scoring both orderings. (Ch 17)

## Agentic & RAG

**RAG (Retrieval-Augmented Generation).** Grounding LLM responses in external knowledge by retrieving relevant documents at query time and injecting them into the prompt. Makes knowledge updatable and auditable without retraining. (Ch 18)

**Vector database.** A database that stores dense embedding vectors and supports approximate nearest-neighbor (ANN) search. Examples: PGVector, Chroma, Pinecone, Qdrant. (Ch 18)

**Chunking.** Splitting documents into smaller passages before embedding. chunk_size and chunk_overlap are the primary design parameters. Smaller chunks are more precise, larger chunks carry more context. (Ch 18)

**Function calling / tool use.** The mechanism by which a language model requests execution of a specific function. The model emits a structured tool call, and the host application runs the function and returns the result. (Ch 19)

**MCP (Model Context Protocol).** Anthropic's open client-server protocol (2024) for standardizing how agents discover and invoke tools and resources. Decouples the agent framework from the tool implementation. (Ch 19)

**A2A (Agent-to-Agent Protocol).** Google's open protocol (2025, donated to the Linux Foundation) for agent-to-agent communication, including task delegation, discovery via Agent Cards, and stateful long-running tasks. (Ch 19)

**AI agent.** An autonomous system using an LLM as its reasoning engine, extended with memory, tools, and an iterative action loop. Governs its own control flow until a goal is met, unlike a single-shot LLM call. (Ch 19)

**ReAct.** Reason + Act. The standard agent pattern: Thought (natural language reasoning) → Action (tool call) → Observation (result) → repeat. The scratchpad THOUGHT block substantially improves tool selection accuracy. (Ch 19)

**In-context learning.** The ability of a language model to learn a new task from examples placed in the prompt, without any weight updates. Scales with model size, and few-shot prompting exploits this. (Ch 7)

**Few-shot / zero-shot prompting.** Zero-shot: ask the model to perform a task with only an instruction. Few-shot: include 1–5 examples in the prompt demonstrating the desired format or reasoning style.

**Chain-of-thought (CoT).** Prompting the model to produce a step-by-step reasoning trace before the final answer. Substantially improves accuracy on multi-step tasks. (Ch 16)

**Context overflow / truncation.** When the input to a model exceeds the context window length. Older tokens are silently dropped, causing the model to "forget" earlier content. (Ch 19, 20)

**Embedding model.** A separate, usually small encoder-only transformer that turns a whole chunk of text into one vector for retrieval. Distinct from the token embeddings inside a generative model, and typically of unrelated dimensionality (768, 1,536). (Ch 18)

**Hybrid search.** Running vector search and BM25 keyword search together and fusing the two ranked lists, so that exact strings (error codes, proper names) are not lost to semantic matching. (Ch 18)

**BM25.** The classical TF-IDF-family keyword ranking function, still the standard lexical half of hybrid search. (Ch 18)

**Reciprocal Rank Fusion (RRF).** Merges ranked lists by summing `1 / (60 + rank)` across them, using only ranks and never raw scores, so no score calibration is needed. (Ch 18)

**Cross-encoder / bi-encoder.** A bi-encoder embeds query and document separately and compares the vectors (fast, used for retrieval). A cross-encoder reads both together and judges joint relevance (accurate, too slow for a whole corpus, used for reranking). (Ch 18)

**GraphRAG.** A retrieval variant that extracts entities and relationships into a knowledge graph at indexing time, enabling corpus-level and multi-hop questions that isolated-chunk retrieval cannot answer. (Ch 18)

**Agent Card.** The JSON descriptor an A2A agent publishes at `/.well-known/agent-card.json`, declaring its identity, skills, and endpoints. (Ch 19)

:::
