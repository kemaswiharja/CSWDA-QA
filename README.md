# CSWDA-QA — Agentic Knowledge Routing for Hybrid Enterprise Data Ecosystems

Code accompanying our paper **"Agentic Knowledge Routing for Hybrid Enterprise Data Ecosystems"** (under review).

CSWDA-QA is a question-answering service that routes a natural-language question to the right mix of heterogeneous data sources, queries them, and synthesises one answer with an LLM:

| Source | Role | How it is queried |
|---|---|---|
| MySQL | Structured enterprise records | LLM-generated SQL (read-only, safety-checked) |
| FAISS | Semantic search over documents | `text-embedding-3-small` vectors |
| Internal Wikibase | Organisational knowledge graph | LLM-generated SPARQL |
| MongoDB | Partnership / news documents | Filter + FAISS-linked lookup |
| Wikidata | External world knowledge | SPARQL + web search fallback |

## Pipeline

1. **Context determination:** classify the question (internal, external, or hybrid).
2. **Query reasoning:** build a query plan that says which sources to hit and in what order (e.g. SQL-first, then FAISS enrichment).
3. **Execution:** run the per-source executors and log LLM calls and databases used.
4. **Answer synthesis:** merge the retrieved evidence into one grounded answer.

The main logic is in `rag_system.py` (`UnifiedRAGSystem`). `main.py` wraps it in a FastAPI service.

## Repository layout

```
main.py                  FastAPI app (/, /health, /query)
rag_system.py            Routing, executors and answer synthesis
wikidata_faiss.index     Prebuilt FAISS index
faiss_document.json      Metadata for the FAISS index
Datasets/
  simplequestion_200.csv 200 SimpleQuestions-style QA pairs used for evaluation
requirements.txt
Procfile                 Railway / Heroku-style start command
.env.example             Environment variables the service needs
```

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your own credentials
uvicorn main:app --reload
```

Query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "where was sasha vujačić born"}'
```

The response includes `answer`, `elapsed_seconds`, `llm_calls` and `databases_used`.

## Configuration

All credentials come from environment variables. See `.env.example` for the full list: OpenAI, SerpAPI, MySQL, Wikibase and MongoDB. Nothing secret is stored in this repository.

> CORS is open (`allow_origins=["*"]`) for the demo deployment. Restrict it before any production use.

## Citation

The paper is currently under review. Citation details will be added on publication.
