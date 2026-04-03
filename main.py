"""
=================================================================================
FastAPI wrapper for the Unified Multi-Database RAG System
=================================================================================
Exposes a /query endpoint for the frontend to call.
All credentials are loaded from environment variables (set in Railway dashboard).
=================================================================================
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── lazy-loaded system singleton ───────────────────────────────────────────────
_rag_system = None


def get_rag_system():
    """
    Return the singleton UnifiedRAGSystem, initialising it on first call.
    We do this lazily so the app boots fast and Railway's health-check passes
    before the (slow) FAISS index load finishes.
    """
    global _rag_system
    if _rag_system is None:
        logger.info("Initialising UnifiedRAGSystem …")
        from rag_system import UnifiedRAGSystem  # imported here to keep startup fast
        _rag_system = UnifiedRAGSystem()
        logger.info("UnifiedRAGSystem ready.")
    return _rag_system


# ── lifespan (replaces deprecated on_event) ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # warm-up: initialise the RAG system before accepting traffic
    get_rag_system()
    yield
    # shutdown: nothing special to do
    logger.info("Shutting down.")


# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Unified RAG API",
    description="Multi-database RAG system: MySQL · FAISS · Wikibase · MongoDB · Wikidata",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # open to any origin – tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    # optional: client can pass a session_id for future history support
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    session_id: Optional[str]
    elapsed_seconds: float
    llm_calls: int
    databases_used: list[str]


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "message": "Unified RAG API is running."}


@app.get("/health")
async def health():
    """Railway uses this to confirm the service is alive."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main endpoint.  The frontend POSTs:
        { "question": "your question here" }
    and receives a structured JSON response.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="'question' must not be empty.")

    logger.info(f"[/query] question={question!r}  session={request.session_id}")

    try:
        rag = get_rag_system()
        t0 = time.time()
        answer, stats = rag.ask(question)
        elapsed = round(time.time() - t0, 2)

        logger.info(f"[/query] answered in {elapsed}s | llm_calls={stats.llm_calls}")

        return QueryResponse(
            answer=answer,
            session_id=request.session_id,
            elapsed_seconds=elapsed,
            llm_calls=stats.llm_calls,
            databases_used=stats.databases_used,
        )

    except Exception as e:
        logger.error(f"[/query] error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
