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

The dataset contains **45 questions** built against a technical blog post on GPU deep learning. 6 are handcrafted and cover the core question types. The rest are synthetically generated using RAGAS `TestsetGenerator` — which builds a knowledge graph from the document chunks and synthesizes questions across different reasoning patterns.

- **`single_passage`** — answer contained in a single chunk.
- **`multi_passage`** — answer requires synthesizing across multiple chunks.
- **`no_answer`** — answer is not in the document; ground truth is *"I don't know."* ⚠️
- **`multi_passage_specific`** — synthetically generated with realistic noise (typos, vague phrasing).

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

LLM-based metrics use **Gemini 2.5 Flash** as judge. The RAG generator uses **Llama 3.3 70B via Groq**. MRR is computed independently.

### Experiment Configs

Each group isolates one variable against the baseline (`chunk_size=1000`, `overlap=200`, `k=5`, similarity search).

**Chunk Size**
- `chunk_500` — chunk_size=500, overlap=100
- `chunk_1500` — chunk_size=1500, overlap=300

**Retrieval Depth**
- `k_3` — retrieve 3 chunks
- `k_8` — retrieve 8 chunks

**Search Algorithm** — MMR balances relevance with diversity. `lambda_mult` controls the trade-off.
- `search_mmr_balanced` — λ=0.5
- `search_mmr_relevant` — λ=0.8
- `search_mmr_diverse` — λ=0.2

### Results

| Config | Recall | Precision | Faithfulness | Ans. Relevancy | Factual F1 | Sem. Sim | MRR |
|---|---|---|---|---|---|---|---|
| **baseline** | 0.600 | 0.608 | 0.393 | 0.567 | 0.403 | 0.708 | 0.778 |
| chunk_500 | 0.571 | 0.440 | 0.192 | 0.490 | 0.291 | 0.661 | 0.662 |
| chunk_1500 | 0.667 | 0.374 | 0.260 | 0.559 | 0.263 | 0.657 | 0.789 |
| k_3 | 0.667 | 0.625 | 0.333 | 0.538 | 0.333 | 0.880 | 0.750 |
| k_8 | 0.667 | 0.333 | 0.333 | 0.645 | 0.400 | 0.715 | 0.750 |
| search_mmr_balanced | 0.731 | 0.600 | 0.423 | 0.550 | 0.187 | 0.725 | 0.762 |
| search_mmr_relevant | 0.808 | 0.596 | 0.222 | 0.520 | 0.224 | 0.687 | 0.770 |
| search_mmr_diverse | 1.000 | 0.625 | 0.250 | 0.856 | 0.400 | 0.892 | 0.750 |

#### Insights by group

**Baseline**
- Strongest factual correctness (0.403) and best faithfulness outside MMR, making it the most reliable all-round config.

**Chunk size**
- `chunk_500` is the weakest config overall — small chunks fragment context and collapse faithfulness to 0.192.
- `chunk_1500` wins on MRR (0.789) but faithfulness drops, suggesting larger chunks aid retrieval ranking yet introduce LLM-confusing noise.

**Retrieval depth (k)**
- `k_3` achieves the highest context precision (0.625) and semantic similarity (0.880) — fewer chunks forces the retriever to be more selective.
- `k_8` improves answer relevancy (0.645) but halves context precision (0.333), showing that more chunks dilute retrieval quality.

**MMR search**
- `mmr_diverse` (λ=0.2) dominates context recall (1.000), answer relevancy (0.856), and semantic similarity (0.892), but at the cost of faithfulness (0.250).
- `mmr_balanced` (λ=0.5) is the best MMR config for faithfulness (0.423), offering a practical middle ground between diversity and groundedness.
- Increasing λ toward relevance (`mmr_relevant`, λ=0.8) boosts recall but sharply reduces faithfulness, revealing a recall–groundedness trade-off within MMR.

#### Key takeaways

- **Recall vs. faithfulness is the central trade-off** — configs that retrieve more (mmr_diverse, k_8) generate less grounded answers.
- **MMR outperforms similarity search on recall** across all three λ values, confirming it is the better retrieval algorithm for this dataset.
- **Baseline is a strong default** — no single variant beats it across all metrics simultaneously; improvements are always at the expense of something else.
- **chunk_500 should be avoided** — it is the only config that degrades performance on every single metric relative to baseline.
- **`mmr_diverse` is the best config if answer quality (relevancy + semantic similarity) is the priority**; `k_3` or `baseline` if factual precision and faithfulness matter more.

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
│   ├── config/                  # YAML config (LLM, retriever, embeddings)
│   ├── model/                   # Pydantic schemas
│   ├── prompts/                 # Prompt templates
│   ├── src/
│   │   ├── document_ingestion/  # Parsing, chunking, indexing
│   │   └── document_chat/       # Retrieval and chat logic
│   └── utils/
│
├── experiments/
│   ├── configs.py               # Experiment configurations
│   ├── dataset.json             # 45-question evaluation dataset
│   ├── evaluators.py            # RAGAS + MRR scoring
│   ├── run_experiment.py        # Experiment runner (skip-if-done, save-as-you-go)
│   └── outputs/                 # Results CSVs
│
└── templates/index.html
```

---

## Setup

```bash
git clone <repo-url> && cd RAG
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
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
