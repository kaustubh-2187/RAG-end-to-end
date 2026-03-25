# Multi-Document Chat

**Live Demo:** https://multi-doc-chat-655964848309.us-central1.run.app

A conversational RAG system for uploading documents and chatting with them. Built with FastAPI and LangChain, evaluated with a custom RAGAS pipeline, deployed on Google Cloud Run.

---

## Pipeline Design

Before retrieval, the user's message and the full conversation history are passed to an LLM that rewrites the input into a self-contained standalone question. Follow-up questions like *"what about the second point?"* carry full context into the retriever rather than relying on the raw message alone.

```
Query
  └── Question Rewriter (LLM + chat history)
        └── Standalone question → FAISS Retriever
              └── Retrieved chunks → Answer Generator (LLM) → Response
```

---

## Evaluation

The `experiments/` directory contains an offline evaluation pipeline across 7 metrics.

### Dataset

The dataset contains **120 questions** built against **20 diverse documents** spanning game wikis, policy papers, technical blogs, fiction, recipes, and more. Questions are synthetically generated using RAGAS `TestsetGenerator` — which builds a knowledge graph from the document chunks and synthesizes questions across different reasoning patterns.

- **`single_passage`** — answer contained in a single chunk.
- **`multi_passage`** — answer requires synthesizing across multiple chunks.
- **`no_answer`** — answer is not in the document; ground truth is *"I don't know."* ⚠️

### Metrics

| Metric | What it measures |
|---|---|
| **Context Recall** | Did the retriever fetch the chunks containing the answer? |
| **Context Precision** | Of the retrieved chunks, how many were actually relevant? |
| **Faithfulness** | Does the generated answer stay grounded in the retrieved context? |
| **Answer Relevancy** | Is the answer responsive to the question asked? |
| **Factual Correctness (F1)** | Does the answer match the reference facts? |
| **Semantic Similarity** | Embedding-level similarity between generated and reference answer. |
| **MRR** | Where does the first relevant chunk appear in the retrieved list? |

LLM-based metrics use **GPT-4o-mini** as both the RAG generator and RAGAS judge. MRR is computed independently.

### Experiment Configs

Each group isolates one variable against the baseline (`chunk_size=1000`, `overlap=200`, `k=5`, similarity search).

**Chunk Size** — `chunk_500` (size=500, overlap=100), `chunk_1500` (size=1500, overlap=300)

**Retrieval Depth** — `k_3` (retrieve 3 chunks), `k_8` (retrieve 8 chunks)

**Search Algorithm** — MMR with `lambda_mult` controlling the relevance–diversity trade-off: `search_mmr_balanced` (λ=0.5), `search_mmr_relevant` (λ=0.8), `search_mmr_diverse` (λ=0.2)

---

### Results

---

#### Single-Passage Questions
> *The answer is fully contained within a single retrieved chunk.*

| Config | Recall | Faithfulness | Ans. Relevancy | Factual F1 | Sem. Sim | MRR |
|---|---|---|---|---|---|---|
| baseline | 0.762 | 0.781 | 0.573 | 0.508 | 0.688 | 0.698 |
| chunk_500 | 0.645 | 0.766 | 0.639 | 0.478 | 0.672 | 0.675 |
| chunk_1500 | 0.773 | 0.798 | 0.635 | 0.520 | 0.709 | 0.798 |
| k_3 | 0.633 | 0.683 | 0.503 | 0.443 | 0.624 | 0.683 |
| k_8 | **0.789** | **0.869** | 0.626 | 0.514 | **0.741** | 0.716 |
| search_mmr_balanced | 0.661 | 0.709 | 0.515 | 0.395 | 0.637 | 0.675 |
| search_mmr_relevant | 0.791 | 0.726 | 0.596 | 0.488 | 0.680 | 0.702 |
| search_mmr_diverse | 0.666 | 0.685 | 0.573 | 0.411 | 0.617 | 0.647 |

This is the pipeline's strongest category. When the answer lives in a single chunk, the generation step is straightforward. Faithfulness scores reflect this: they are consistently high across every config (0.68–0.87), meaning the model almost never hallucinates when the right chunk is present.

The main differentiator here is retrieval quality, not generation. Configs that retrieve more or larger chunks surface the answer chunk more reliably, which is why `k_8` and `chunk_1500` lead on recall and MRR. Notably, `k_8` achieves the highest faithfulness (0.869) of any config across any question type.

Answer relevancy (0.50–0.64) is the weakest metric for this category, suggesting the model occasionally produces answers that are factually correct but not directly responsive to the specific question asked.

---

#### Multi-Passage Questions
> *The answer must be synthesised across multiple retrieved chunks.*

| Config | Recall | Faithfulness | Ans. Relevancy | Factual F1 | Sem. Sim | MRR |
|---|---|---|---|---|---|---|
| baseline | 0.513 | 0.713 | **0.668** | 0.326 | 0.659 | 0.744 |
| chunk_500 | 0.460 | 0.719 | 0.521 | 0.261 | 0.601 | 0.669 |
| chunk_1500 | **0.640** | 0.754 | 0.556 | 0.288 | 0.657 | **0.824** |
| k_3 | 0.379 | 0.671 | 0.577 | 0.294 | 0.588 | 0.725 |
| k_8 | 0.612 | **0.783** | 0.659 | **0.352** | **0.676** | 0.744 |
| search_mmr_balanced | 0.397 | 0.706 | 0.594 | 0.242 | 0.626 | 0.691 |
| search_mmr_relevant | 0.523 | 0.769 | 0.625 | 0.339 | 0.657 | 0.729 |
| search_mmr_diverse | 0.345 | 0.688 | 0.533 | 0.219 | 0.560 | 0.674 |

This is where the pipeline struggles most. The retriever must simultaneously surface multiple relevant chunks. Recall drops sharply compared to single-passage (0.35–0.64 vs 0.63–0.79), and factual F1 collapses to 0.22–0.35, exposing the core limitation: the model's retrieval coverage for distributed answers is unreliable.

Faithfulness remains relatively high (0.67–0.78) even in this category — the model stays grounded in whatever context it receives. The problem is not hallucination; it is incomplete retrieval. The model answers faithfully from partial evidence, producing answers that are plausible but factually incomplete.

The contrast between factual F1 in single-passage (0.44–0.52) and multi-passage (0.22–0.35) quantifies exactly how much the pipeline degrades when synthesis is required.

Larger chunks (`chunk_1500`) help the most here because a single chunk is more likely to contain a larger portion of a distributed answer, reducing how many chunks the retriever needs to find. This is the one category where chunk size has the most meaningful impact on recall.

---

#### No-Answer Questions (Abstention)
> *The answer is not in the document. The correct response is "I don't know."*

| Config | Abstention Accuracy | Hallucination Rate |
|---|---|---|
| baseline | 0.900 | 0.100 |
| chunk_500 | 0.900 | 0.100 |
| chunk_1500 | 0.875 | 0.125 |
| k_3 | 0.875 | 0.125 |
| k_8 | 0.875 | 0.125 |
| search_mmr_balanced | **0.925** | **0.075** |
| search_mmr_relevant | 0.900 | 0.100 |
| search_mmr_diverse | **0.925** | **0.075** |

The pipeline handles out-of-scope questions well. But these results reveal a meaningful tension with the previous two categories: the configs that help most with answerable questions tend to hurt here.

Configs that retrieve more content (`k_8`, `chunk_1500`) have a slightly higher hallucination rate (12.5%) compared to baseline (10%).

MMR configs are the exception. `search_mmr_balanced` and `search_mmr_diverse` achieve the lowest hallucination rate (7.5%), outperforming similarity search on this dimension. By promoting diversity in retrieved chunks, MMR may reduce the chance that multiple chunks all weakly point in the same wrong direction.

This category also reveals a fault in the pipeline's self-awareness. A 10–12.5% hallucination rate means the model fabricates an answer roughly 1-in-8 to 1-in-10 times when there is genuinely nothing to find. For applications where out-of-scope queries are common, this is a meaningful failure mode.

---

### Cross-Type Analysis

The three question types expose three distinct failure modes in the pipeline:

| Question Type | Primary failure mode | Root cause |
|---|---|---|
| **Single-passage** | Weak answer relevancy | Generation: model answers factually but misses the specific ask |
| **Multi-passage** | Low recall and factual F1 | Retrieval: distributed answers are only partially covered |
| **No-answer** | Hallucination on missing context | Generation: model confabulates from weakly relevant chunks |

**The pipeline's biggest unsolved problem is multi-passage retrieval.** The ~0.18 F1 gap between single- and multi-passage questions persists across every config tested. No chunking strategy, retrieval depth, or search algorithm closes it — because the root cause is architectural: FAISS retrieves the top-k most similar chunks independently, with no mechanism to ensure that all pieces of a distributed answer are retrieved together.

**Faithfulness is not the bottleneck.** Across all question types and configs, faithfulness is consistently high (0.67–0.87). The model reliably stays grounded in whatever context it receives. The failures are upstream — in what gets retrieved — not in how the model uses it.

**The abstention–coverage trade-off is real but manageable.** Configs that improve answerable question coverage (`k_8`, `chunk_1500`) introduce a small increase in hallucination rate on no-answer questions.

---

## Project Structure

```
RAG/
├── main.py
├── requirements.txt
├── Dockerfile
├── Jenkinsfile
│
├── multi_doc_chat/
│   ├── config/                  # YAML config
│   ├── model/                   # Pydantic schemas
│   ├── prompts/                 # Prompt templates
│   ├── src/
│   │   ├── document_ingestion/  # Parsing, chunking, indexing
│   │   └── document_chat/       # Retrieval and chat logic
│   └── utils/
│
├── experiments/
│   ├── configs.py               # Experiment configurations
│   ├── dataset.json             # Evaluation dataset (120 questions, 20 documents)
│   ├── evaluators.py            # RAGAS + MRR scoring
│   ├── run_experiment.py        # Experiment runner
│   └── outputs/                 # Experiment Results
│
└── templates/index.html
```

---

## Setup

```bash
git clone <repo-url> && cd RAG
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add GOOGLE_API_KEY, GROQ_API_KEY, HF_TOKEN
python main.py        # visit http://localhost:8000
```

---

## Deployment

```bash
docker build -t multi-doc-chat .
docker run -p 8000:8000 -e GOOGLE_API_KEY=$GOOGLE_API_KEY -e GROQ_API_KEY=$GROQ_API_KEY multi-doc-chat
```
