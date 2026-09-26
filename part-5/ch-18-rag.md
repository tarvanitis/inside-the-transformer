# Retrieval-Augmented Generation

Every language model has a knowledge cutoff. Ask any chatbot about yesterday's earnings
report and it cannot answer from training data. That event simply did not exist when their
weights were frozen. Fine-tuning could bake in new information, but it is expensive, slow, and
the model may still hallucinate because parametric memory is opaque: you cannot easily inspect
what the model "knows" or trace where an answer came from.

**Retrieval-Augmented Generation (RAG)** takes a different approach. Instead of storing knowledge
in weights, keep it in a searchable database. At query time, retrieve the relevant passages and
inject them into the prompt. The LLM reasons over the retrieved evidence, which it can cite
by source, rather than relying on weights alone.

The acronym is self-explanatory: **R**etrieval (find relevant documents) + **A**ugmentation
(inject them as context) + **G**eneration (LLM answers using that context).

**In this chapter**

- Why knowledge lives better in a database than in weights.
- The two phases: offline indexing and online retrieval + generation.
- Embeddings as semantic addresses: how cosine similarity finds "meaning-near" chunks.
- Chunking strategy: the one parameter that most affects retrieval quality.
- Agent vs. chain: two wiring patterns and when to choose each.
- Advanced retrieval: hybrid search, reranking, contextual embeddings, GraphRAG.
- Long-context windows and why they complement rather than replace retrieval.

## Why not fine-tuning?

The alternative to RAG for incorporating new knowledge is to fine-tune: continue training on the
new documents. The table below makes the trade-off concrete:

| Concern | Fine-tuning | RAG |
|---|---|---|
| Update knowledge | Retrain, hours to days | Re-index, minutes |
| Add/remove a document | Retrain | Delete from vector store |
| Cite sources | Model cannot reliably | Retrieved chunk is the source |
| Scale to large corpora | Context window is fixed | Corpus can be arbitrarily large |
| Hallucination risk | Model still confabulates | Grounded in retrieved text |
| Training cost | High GPU budget | Inference + vector DB only |

::: {.caption}
**Table 18.1.** Fine-tuning against RAG, concern by concern.
:::

::: {.callout .plain}
RAG does not add new *skills* to the model, only new knowledge. If your problem is that the
model cannot write well-structured SQL, RAG won't help. If the problem is that the model doesn't
know your company's internal policy documents, RAG is exactly right.
:::

::: {.callout .caution}
**When RAG struggles.** RAG depends on semantic similarity between the query and the right
document. It fails when the right chunk cannot be found by meaning alone (e.g., looking up an
exact product SKU), when the corpus is too structured for text search (use a SQL agent instead),
or when answering requires synthesizing many documents across the whole corpus ("what are the
recurring themes across our 10,000 customer tickets?"). For the last case, see GraphRAG below.
:::

## The two-phase pipeline

RAG splits cleanly into two phases with different timing: indexing runs offline once, retrieval
and generation run on every query.

::: {.figure}
![](assets/figures/ch18/fig-rag-pipeline.svg)
:::

::: {.caption}
**Figure 18.1.** The two-phase RAG pipeline: offline indexing, then online retrieval and generation.
:::

### Phase 1 (Indexing)

**Step 1: Load.** Ingest source documents: PDFs, web pages, databases, wikis. Each becomes a
`Document` object with text content and metadata (source URL, page number, author).

**Step 2: Chunk.** Documents are split into smaller passages before embedding. Two reasons:
(a) the LLM context window is finite, so you cannot inject a 300-page manual into every prompt;
(b) smaller, more focused chunks embed into a cleaner semantic signal and retrieve more precisely.

The main parameters:

- **chunk_size.** Maximum characters per chunk (typically 500–1,500). Shorter = more precise
  retrieval, while longer chunks carry more context per passage.
- **chunk_overlap.** Characters shared between adjacent chunks (typically 10–20% of chunk_size),
  so facts that straddle a boundary are not lost.

::: {.callout .note}
**Chunking is a design decision.** There is no universal optimum. A legal contract benefits from
large chunks that preserve clause context. A QA dataset benefits from small, fact-dense chunks.
Experiment with your data: measure retrieval precision on a hand-curated set of (question,
right-chunk) pairs before fixing chunk_size.
:::

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,   # tracks position within source document
)
chunks = splitter.split_documents(docs)
# RecursiveCharacterTextSplitter splits on paragraph → line → word → character, in order
# — it respects natural text boundaries before resorting to mid-word cuts
```

**Step 3: Embed.** Each chunk is converted into a dense vector, typically 768 or 1,536
dimensions, by an embeddings model. Semantically similar chunks land close together in this
high-dimensional space. "Paris is the capital of France" and "the French capital city" embed
near each other. "Paris" and "banana" embed far apart.

The embeddings model is a small encoder-only transformer (the encoder half of Chapter 11's
architecture, without the decoder or cross-attention, trained with contrastive objectives on
paired texts). Popular choices: OpenAI
`text-embedding-3-small` (1,536 dimensions), `nomic-embed-text` (768 dimensions, runs locally),
`BGE-M3` (multi-lingual, open-weight).

**Step 4: Store.** Vectors and their source text are persisted in a **vector store**, a database
that supports approximate nearest-neighbor (ANN) search. Common options:

| Store | Best for |
|---|---|
| **PGVector** | Already running PostgreSQL, add `CREATE EXTENSION vector;` |
| **Chroma** | Local development, zero setup |
| **Pinecone** | Managed cloud, high scale, serverless tier |
| **Qdrant** | Self-hosted, rich filtering, Docker-friendly |

::: {.caption}
**Table 18.2.** Vector stores, and what each is best suited to.
:::

::: {.callout .plain}
Comparing your query against every stored vector is exact but slow. At a million chunks it is a
million dot products per question. So vector databases build an index that finds *almost* the closest
matches by checking a small fraction of the candidates. The three you will meet: **flat** compares
everything (exact, fine up to roughly 10,000 chunks), **IVF** sorts vectors into clusters and
searches only the nearest few, and **HNSW** builds a navigable graph and walks it (fastest, the usual
default, but slower to build and hungrier for memory). All of them can miss a genuine match. That
missed fraction is called *recall*, and every index exposes a knob that buys recall back with latency.
:::

```python
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection="postgresql+psycopg://user:pass@localhost/mydb",
)
doc_ids = vector_store.add_documents(documents=chunks)
print(f"Indexed {len(doc_ids)} chunks")
```

### Phase 2 (Retrieval and generation)

**Step 5: Embed the query.** The user's question is converted to a vector using the *same*
embeddings model used at index time. This is essential: both query and documents must live in
the same vector space for cosine similarity to be meaningful.

**Step 6: Similarity search.** Find the stored chunk vectors closest to the query vector.
"Closest" is measured by cosine similarity, the cosine of the angle between two vectors:

$$\mathrm{cosine\ similarity}(q, c) = \frac{q \cdot c}{\lVert q \rVert \cdot \lVert c \rVert}$$

A value of 1 means the vectors point in the same direction (maximally similar). A value of 0 means
orthogonal (unrelated). A value of −1 would mean pointing in opposite directions, which is rare for text
embeddings in practice. The top-k chunks with the highest cosine similarity to the query are
returned. A typical default is k = 4.

::: {.callout .idea}
Cosine similarity measures the *angle* between two vectors and ignores their length. Two chunks
pointing the same direction score 1 whether their vectors are long or short. In practice most
embedding models already return vectors of length 1, in which case cosine similarity and Euclidean
distance rank results identically. Cosine is the convention because it is bounded to a readable −1
to 1, and because it stays meaningful for the models that don't normalize.
:::

::: {.figure}
![](assets/figures/ch18/fig-semantic-search.svg)
:::

::: {.caption}
**Figure 18.2.** Semantic search. The query and every chunk are points in embedding space. Retrieval returns the chunks whose vectors sit at the smallest angle to the query (highest cosine similarity).
:::

**Step 7: Generate.** Retrieved chunks are serialized and injected into the prompt. The LLM
generates a response grounded in the provided context:

```python
query = "What is the company's remote work policy?"
docs  = vector_store.similarity_search(query, k=4)

context = "\n\n".join(
    f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
    for doc in docs
)

response = model.invoke([
    {"role": "system", "content": f"Answer using this context:\n\n{context}"},
    {"role": "user",   "content": query},
])
```

Because the retrieved passages appear verbatim in the prompt, the model can cite them exactly,
and you can audit the answer by checking which chunks it drew from.

## Agent vs. chain: two wiring patterns

Once the components exist, there are two ways to wire retrieval into an application. They trade
off latency, flexibility, and control.

| | RAG Agent | RAG Chain |
|---|---|---|
| **Mechanism** | LLM decides when and how to search | Retrieval is hard-wired before every LLM call |
| **LLM calls per query** | 2 (generate search query → generate answer) | 1 |
| **Multi-step retrieval** | Yes, model can issue several searches | No, single retrieval pass |
| **Off-topic queries** | Model skips search when not needed | Always retrieves, regardless |
| **Latency** | Higher (two inference calls) | Lower |
| **Best for** | General Q&A, conversational agents | High-throughput, always-retrieve pipelines |

::: {.caption}
**Table 18.3.** RAG as an agent against RAG as a fixed chain.
:::

**RAG Agent.** Wrap the vector store in a tool. The LLM decides whether to call it, crafts its
own search query from conversational context, and can call it multiple times for complex questions:

```python
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information from the knowledge base to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

agent = create_react_agent(
    model,
    tools=[retrieve_context],
    prompt=(
        "You are a helpful assistant with access to a knowledge base. "
        "Use retrieve_context when you need information. Always cite sources."
    ),
)
```

**RAG Chain.** Retrieval runs as middleware before the model call. Deterministic, one inference
call, no tool overhead. Preferred when every query needs retrieval and latency matters.

## Advanced retrieval patterns

Basic top-k cosine retrieval works well for clear, fact-seeking queries. Four techniques lift
quality for harder cases:

### Hybrid search

Pure vector search finds semantically similar chunks but may miss exact keyword matches: product
codes, error messages, proper names. **Hybrid search** runs both a vector search and a BM25
keyword search (BM25 is the classical keyword-ranking function from the TF-IDF family: it scores a document by how well its exact words match the query terms) and merges the two ranked
lists with **Reciprocal Rank Fusion (RRF)**, a formula that promotes chunks ranked highly by
either method. The improvement is especially large for queries involving specific strings where
the literal match matters more than the gist.

RRF needs no tuning and no score calibration, which is why it won: it throws away the raw scores and
uses only the *ranks*. Each chunk gets `1 / (60 + rank)` from each list it appears in, and the sums
are sorted. A chunk ranked 1st by BM25 and 8th by vector search scores `1/61 + 1/68 = 0.031`. One
ranked 3rd by both scores `2/63 = 0.032` and edges it out. The constant 60 is conventional. It damps
the advantage of the very top ranks so a single list cannot dominate.

### Reranking

Retrieve a larger candidate set (k = 20) cheaply, then re-score with a slower but more accurate
**cross-encoder reranker**. The embedding search above is a *bi-encoder*: it turns the query and each
chunk into vectors separately, then compares them. A cross-encoder instead reads the query and the
chunk *together*, so it sees both at once and can judge their joint relevance. That is far more
accurate and far too slow to run over the whole corpus, which is why it goes second. Keep only the top 4 re-scored chunks for the prompt:

```python
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
reranker       = CohereRerank(top_n=4)

retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever,
)
```

Think of it as casting a wide net, then hand-sorting the catch. The net is cheap (cosine over
20 chunks). The sorting is accurate (cross-encoder reads query+chunk jointly).

::: {.figure}
![](assets/figures/ch18/fig-rerank-funnel.svg)
:::

::: {.caption}
**Figure 18.3.** Retrieve, then rerank. A fast bi-encoder pulls a wide set of candidates by vector similarity. A slower cross-encoder, which reads the query and chunk together, re-scores them and keeps the best few.
:::

### Contextual embeddings

A cheap, high-impact improvement: before embedding each chunk, prepend a short LLM-generated
blurb that situates it within its source document. Without this, a chunk saying "this applies
to all employees hired after 2022" loses the document context that told you it was about the
parental leave policy. The contextual blurb (one or two sentences generated by a small LLM at
index time) dramatically reduces the "retrieved the right document, wrong chunk" failure mode.

### GraphRAG

Standard RAG retrieves isolated chunks. It struggles with questions that require connecting facts
scattered across many documents ("what regulatory changes affected our product over the past five
years?") or corpus-level summarization ("what are the main themes across all these customer
complaints?").

**GraphRAG** (Microsoft Research, 2024) adds a graph layer: an LLM extracts entities and
relationships into a knowledge graph during indexing, clusters the graph into communities, and
pre-summarizes each community. At query time it retrieves over the graph structure, following
edges and community summaries, rather than over flat chunks.

::: {.callout .deepdive}
**When GraphRAG earns its cost.** Graph construction is an extra LLM-heavy indexing pass: every
document gets read and parsed for entities and relationships. Reserve it for corpora where
relationships are the point (regulations and their cross-references, incident histories and their
causes, codebases and their dependency chains) and where questions are genuinely multi-hop or
global. For simple fact lookup, hybrid search + reranking is cheaper and usually sufficient.
:::

## Long context vs. RAG

Chapter 8 introduced this trade-off from the inference side. Here is the retrieval side of it.
Frontier models now ship very large context windows, and it is tempting to "paste everything in" and
skip retrieval.

> **As of 2026-09-14,** frontier context windows run from 200K to 1M+ tokens. Chapter 1 carries the
> current per-vendor snapshot. The trade-off below is durable even as the numbers move:

| Concern | Full-context stuffing | RAG |
|---|---|---|
| Cost per query | High, 200K tokens per call | Low, top-k chunks only |
| Accuracy | "Lost-in-the-middle" degradation | Focused, relevant context |
| Freshness | Model sees what you pass | Re-index to update |
| Corpus scale | Hard limit: context window | Arbitrarily large |

::: {.caption}
**Table 18.4.** Long-context stuffing against retrieval.
:::

The practical answer is hybrid: use a large context window to hold *more* retrieved evidence and
full conversation history, but keep retrieval as the filter that decides what enters the prompt.
Long context complements RAG. It does not replace it.

::: {.callout .note}
**Prompt caching.** Cloud providers charge less for tokens that hit the prompt cache (a cached
prefix costs ~10% of the first-call price on most providers). A large, stable system prompt,
including frequently-retrieved reference documents, can be cached, which narrows the cost gap
between full-context stuffing and RAG for read-heavy workloads. But caching only helps for
repeated content. Dynamic retrieval results do not benefit.
:::

## Evaluating a RAG system

A RAG system can fail in two independent places, and you have to measure them separately or you will
tune the wrong one.

::: {.callout .plain}
It is tempting to judge a RAG system by reading a few answers and deciding whether they feel right.
Resist that. When an answer is wrong you need to know *which half* broke. Did the search fail to
find the paragraph, or did the model find it and ignore it? Those have completely different fixes,
and you cannot tell them apart by looking at the final answer.
:::

**Retrieval quality** asks: did the right chunk come back at all? Build 30–50 (question,
correct-chunk) pairs by hand from your own corpus. This is a couple of hours of work and it is the
highest-leverage thing you will do. Then measure **recall@k** (what fraction of questions had the
right chunk somewhere in the top k) and **precision@k** (what fraction of returned chunks were
relevant). If recall@10 is low, no amount of prompt engineering will save you: fix chunking, try
hybrid search, or add reranking.

**Generation quality** asks: given the right chunk, did the model use it? Two measures matter.
**Faithfulness** asks whether every claim in the answer is supported by the retrieved text.
**Answer relevance** asks whether the answer addresses the question that was put. These are usually scored with an
LLM-as-judge (Chapter 17), one sentence at a time against the retrieved sources.

The diagnostic value is in the split. High recall with low faithfulness means the model is
confabulating on top of good evidence, so tighten the prompt or force citations in the output
schema. Low recall with high faithfulness means your retriever is the problem, and the model is
faithfully answering from the wrong page.

## What goes wrong

Three failure modes account for most RAG quality issues in production:

**Chunking breaks semantic units.** A paragraph about "employee parental leave eligibility" split
mid-sentence produces two chunks, neither of which embeds the complete idea. Fix: tune overlap,
use semantic chunking (split on topic boundaries rather than character counts), or use contextual
embeddings.

**Query-document vocabulary mismatch.** The user asks "sick days" but the policy document says
"personal leave entitlement." Cosine similarity sees different words, scores low, retrieves
the wrong chunk. Fix: hybrid BM25+vector search, or a query-expansion step where the agent
rewrites the query before searching.

**Retrieved ≠ grounded.** The model retrieves the right chunks but then generates text that goes
beyond them or contradicts them, the classic hallucination problem. RAG reduces but does not
eliminate hallucination. The model can still confabulate. Fix: post-generation fact-checking,
structured output formats that force citations, or a "grading" step that checks each sentence
against the retrieved sources.

## Summary

- **RAG = Retrieval + Augmentation + Generation.** Knowledge lives in a searchable vector store,
  not in model weights, making it updatable, auditable, and scalable beyond any context window.
- **Indexing** (offline): load → chunk → embed → store. Chunk size is the single most important
  design decision. Experiment with your data.
- **Retrieval** (query time): embed query → cosine similarity search → inject top-k chunks.
  Same embeddings model at index and query time. This is required.
- **Agent vs. chain**: agent is flexible (multi-step, skips unnecessary retrieval). Chain is
  faster (single inference call, always retrieves).
- **Advanced patterns**: hybrid BM25+vector search for exact terms, cross-encoder reranking for
  precision, contextual embeddings for chunk ambiguity, and GraphRAG for multi-hop and global queries.
- **Long context complements RAG.** It does not replace it. Retrieval decides what enters the
  prompt. A larger context window holds more retrieved evidence.

> **Coming up:** Chapter 19 takes RAG a step further, to agents that not only retrieve but take
> actions: calling APIs, running code, managing files, and coordinating with other agents in
> multi-agent systems.
