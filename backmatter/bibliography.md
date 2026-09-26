::: {.frontmatter}

# Bibliography

An annotated selection of the papers, tutorials, and tools this book draws from. Organized by
topic. Entries within each section are roughly chronological.

## Foundational papers

**Attention Is All You Need**  
Vaswani, A. et al. (2017). arXiv:1706.03762.  
The original transformer paper. Introduced the encoder-decoder architecture with multi-head
self-attention and positional encoding. Every modern LLM descends from this design.

**Language Models are Few-Shot Learners (GPT-3)**  
Brown, T. et al. (2020). arXiv:2005.14165.  
Demonstrated that scaling language models to 175B parameters produces emergent few-shot
learning ability. Established the pre-train-then-prompt paradigm.

**Training Compute-Optimal Large Language Models (Chinchilla)**  
Hoffmann, J. et al. (2022). arXiv:2203.15556.  
Showed that for a fixed compute budget, model size and training tokens should scale together
in a ~1:20 ratio. Established the scaling law used to plan pre-training runs.

**RoFormer: Enhanced Transformer with Rotary Position Embedding**  
Su, J. et al. (2021). arXiv:2104.09864.  
Introduced RoPE, rotary position embeddings applied to Q/K before the attention dot product.
The positional encoding used in Llama, Mistral, and most current models.

**GLU Variants Improve Transformer (SwiGLU)**  
Shazeer, N. (2020). arXiv:2002.05202.  
Introduced gated linear unit variants for FFN activations. SwiGLU (swish × linear gate) is
the standard FFN activation in all modern LLMs.

**Root Mean Square Layer Normalization**  
Zhang, B. and Sennrich, R. (2019). arXiv:1910.07467.  
Introduced RMSNorm, normalization by root-mean-square without mean subtraction. Faster and
simpler than LayerNorm, and the standard normalization in Llama, Mistral, and successors.

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**  
Dao, T. et al. (2022). arXiv:2205.14135.  
Reordered the attention computation to stay within fast SRAM rather than writing to HBM.
Makes long-context training and inference practical.

**GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**  
Ainslie, J. et al. (2023). arXiv:2305.13245.  
Introduced Grouped-Query Attention (GQA) as a middle ground between Multi-Head Attention
and Multi-Query Attention. Now the standard KV-sharing strategy.

**Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
Gu, A. and Dao, T. (2023). arXiv:2312.00752.  
Introduced Mamba, a selective state-space model (SSM) that achieves linear-time inference.
The leading non-attention architecture alternative.

**Language Models are Unsupervised Multitask Learners (GPT-2)**  
Radford, A. et al. (2019). OpenAI technical report.  
The 124M–1.5B decoder-only models whose dimensions are this book's Part II reference scale.

**Neural Machine Translation of Rare Words with Subword Units (BPE)**  
Sennrich, R. et al. (2015). arXiv:1508.07909.  
Introduced byte-pair encoding for NLP, the tokenization scheme of Chapter 3.

**Layer Normalization**  
Ba, J. et al. (2016). arXiv:1607.06450.  
The normalization used in the original transformer, and the predecessor RMSNorm simplifies.

**A Mathematical Framework for Transformer Circuits**  
Elhage, N. et al. (2021). Anthropic / Transformer Circuits Thread.  
The residual-stream view and the circuit vocabulary underpinning Chapter 7.

**In-context Learning and Induction Heads**  
Olsson, C. et al. (2022). Anthropic / Transformer Circuits Thread.  
Identified induction heads and their sudden formation during training.

**Switch Transformers: Scaling to Trillion Parameter Models**  
Fedus, W. et al. (2021). arXiv:2101.03961.  
Simplified MoE routing to a single expert per token, making sparse scaling practical.

**YaRN: Efficient Context Window Extension of Large Language Models**  
Peng, B. et al. (2023). arXiv:2309.00071.  
The RoPE-scaling method used to extend trained context windows, referenced in Chapters 4, 8 and 13.

## Training and alignment papers

**Training language models to follow instructions with human feedback (InstructGPT)**  
Ouyang, L. et al. (2022). arXiv:2203.02155.  
The paper that established RLHF for instruction-following. Showed that a 1.3B model trained
with RLHF was preferred over a 175B base model by human evaluators.

**LoRA: Low-Rank Adaptation of Large Language Models**  
Hu, E. et al. (2021). arXiv:2106.09685.  
Introduced LoRA, fine-tuning via small rank-r updates B@A added to frozen weight matrices.
The standard PEFT method, enabling SFT of large models on consumer hardware.

**LIMA: Less Is More for Alignment**  
Zhou, C. et al. (2023). arXiv:2305.11206.  
Showed that 1,000 carefully curated SFT examples produce a model competitive with those
trained on 52,000 synthetic examples. Established the quality-over-quantity principle for SFT.

**Direct Preference Optimization: Your Language Model is Secretly a Reward Model**  
Rafailov, R. et al. (2023). arXiv:2305.18290.  
Derived a closed-form RLHF objective that optimizes preference pairs directly without a
separate reward model or RL loop. The log-ratio of policy and reference model is the implicit
reward.

**DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**  
Shao, Z. et al. (2024). arXiv:2402.03300.  
Introduced GRPO (Group Relative Policy Optimization), replacing PPO's value network with
group-relative advantage estimation. The training algorithm behind DeepSeek-R1.

**DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**  
DeepSeek-AI (2025). arXiv:2501.12948.  
Demonstrated that RLVR training (RL with verifiable rewards on math/code) produces emergent
chain-of-thought reasoning, backtracking, and self-verification without human demonstrations.
Also showed that distilling a large reasoning model into a small one via SFT outperforms
running RL on the small model directly.

**DeepSeek-V3 Technical Report**  
DeepSeek-AI (2024). arXiv:2412.19437.  
MoE at scale with MLA (Multi-Head Latent Attention), 671B total / 37B active parameters.
Established the current state of the art for open-weight models.

**Proximal Policy Optimization Algorithms**  
Schulman, J. et al. (2017). arXiv:1707.06347.  
Introduced PPO and the clipped surrogate objective. The RL algorithm behind InstructGPT-style
RLHF, and the baseline GRPO is defined against.

**Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**  
Wei, J. et al. (2022). arXiv:2201.11903.  
Showed that prompting a model to write intermediate reasoning steps sharply improves performance
on arithmetic and commonsense tasks. The origin of the chain-of-thought idea Part IV builds on.

**Constitutional AI: Harmlessness from AI Feedback**  
Bai, Y. et al. (2022). arXiv:2212.08073.  
Replaced human harmlessness labels with model-generated critiques against a written constitution
(RLAIF).

**QLoRA: Efficient Finetuning of Quantized LLMs**  
Dettmers, T. et al. (2023). arXiv:2305.14314.  
LoRA on a 4-bit quantized base model, bringing 65B fine-tuning onto a single GPU.

**Adam: A Method for Stochastic Optimization**  
Kingma, D. & Ba, J. (2014). arXiv:1412.6980.  
The adaptive optimizer used, in its AdamW form, for essentially all LLM training.

**Decoupled Weight Decay Regularization (AdamW)**  
Loshchilov, I. & Hutter, F. (2017). arXiv:1711.05101.  
Separated weight decay from the gradient update, fixing Adam's regularization behavior.

**Scaling Laws for Neural Language Models**  
Kaplan, J. et al. (2020). arXiv:2001.08361.  
Established the power-law relationship between loss, parameters, data and compute, and the
parameter-heavy allocation that Chinchilla later corrected.

## Architecture and inference efficiency

**Llama 2: Open Foundation and Fine-Tuned Chat Models**  
Touvron, H. et al. (2023). arXiv:2307.09288.  
Open-weight models establishing GQA (in the larger sizes), RMSNorm, SwiGLU, and RoPE as the default decoder-only
architecture. The foundation for most open fine-tuned models through 2024.

**Efficient Memory Management for Large Language Model Serving with PagedAttention**  
Kwon, W. et al. (2023). arXiv:2309.06180.  
Introduced PagedAttention, managing the KV cache in fixed-size pages to eliminate
fragmentation. The core innovation in vLLM.

**Fast Inference from Transformers via Speculative Decoding**  
Leviathan, Y. et al. (2022). arXiv:2211.17192.  
Formalized speculative decoding with draft model + target model verification. Achieves 2–3×
latency improvement with identical output distribution.

## Evaluation

**Measuring Massive Multitask Language Understanding (MMLU)**  
Hendrycks, D. et al. (2020). arXiv:2009.03300.  
The benchmark across 57 academic subjects that defined capability evaluation for two years,
now saturated for frontier models.

**Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**  
Zheng, L. et al. (2023). arXiv:2306.05685.  
Introduced the LLM-as-judge methodology and the Chatbot Arena crowdsourced evaluation
platform. Documented verbosity and positional biases in LLM judges.

**SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**  
Jimenez, C. et al. (2023). arXiv:2310.06770.  
Introduced SWE-bench: resolve a real GitHub issue and pass the test suite. SWE-bench
Verified (2024) removed ambiguous/broken tasks.

## Retrieval and agents

**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  
Lewis, P. et al. (2020). arXiv:2005.11401.  
Introduced the RAG framework: retrieval plus generation with a pretrained dense retriever and
a seq2seq model. Established the core design pattern.

**Contextual Retrieval**  
Anthropic (2024). Technical blog post.  
Showed that prepending a short LLM-generated context blurb to each chunk before embedding
significantly reduces retrieval failures. Introduced contextual BM25 and contextual embeddings.

**From Local to Global: A Graph RAG Approach to Query-Focused Summarization**  
Edge, D. et al. (2024). arXiv:2404.16130.  
Introduced GraphRAG, extracting entities and relationships into a knowledge graph, clustering
into communities, and pre-summarizing. Handles multi-hop and global questions that vector RAG
cannot.

**ReAct: Synergizing Reasoning and Acting in Language Models**  
Yao, S. et al. (2022). arXiv:2210.03629.  
Introduced the Thought → Action → Observation loop that is the foundation of virtually every
production agent today.

**Reflexion: Language Agents with Verbal Reinforcement Learning**  
Shinn, N. et al. (2023). arXiv:2303.11366.  
Added verbal self-reflection after failed attempts. The agent writes what went wrong and
retries with improved context.

## Tutorials and learning resources

**The Illustrated Transformer.** Jay Alammar
Visual walkthrough of the original transformer architecture. The clearest introduction to
attention mechanics for readers new to the field.

**The Annotated Transformer.** Harvard NLP
Side-by-side paper and implementation in PyTorch. Executable line-by-line. A reference for
Chapter 11's encoder-decoder implementation.

**Let's build GPT: from scratch, in code, explained.** Andrej Karpathy (2023)
YouTube lecture (~2 hours) building a character-level GPT from scratch. The primary
inspiration for Chapter 10's build-a-GPT chapter.

**Neural Networks: Zero to Hero.** Andrej Karpathy
Lecture series progressing from micrograd (backpropagation from scratch) through nanoGPT.
Recommended prerequisite if Chapters 2 and 10 felt fast.

**microGPT.** Andrej Karpathy (2026). karpathy.github.io/2026/02/12/microgpt/
A complete GPT in ~200 lines of pure Python: a from-scratch autograd engine, tokenizer, a
GPT-2-like transformer, the Adam optimizer, and both training and inference loops, with no external
libraries. The capstone of the micrograd → nanoGPT learning arc. Recommended after
Part III for readers who want to understand backpropagation itself, not just use it.

**But what is a GPT? Visual intro to transformers.** 3Blue1Brown (2024)
Animated visual explanation of embeddings, attention, and the transformer forward pass.
Excellent companion to Chapters 3–8.

**Lil'Log.** Lilian Weng
Blog covering attention mechanisms, RLHF, agents, and scaling. "LLM Powered Autonomous
Agents" (2023) is the canonical survey of the agent landscape.

## Code and tools

**nanoGPT.** Andrej Karpathy
Minimal, readable GPT-2 implementation in PyTorch. The direct ancestor of Chapter 10's
code.

**Hugging Face Transformers**  
Production transformer library. The industry standard for loading, fine-tuning, and deploying  
transformer models. Extensive documentation and model hub.

**llama.cpp.** Georgi Gerganov
C++ inference engine for GGUF quantized models. The engine under Ollama, LM Studio, and
most CPU inference tools. Made local LLM deployment practical.

**LangGraph**  
State-machine-based agent orchestration framework for Python. The recommended foundation for  
production stateful agents.

**vLLM**  
High-throughput GPU inference server with PagedAttention and continuous batching. The standard  
for production cloud deployment.

:::
