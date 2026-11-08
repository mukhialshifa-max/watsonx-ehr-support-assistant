"""FastAPI web app: a chat page plus a JSON API."""

from __future__ import annotations

import logging
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .knowledge_base import KnowledgeBase
from .llm import Answerer, OfflineAnswerer, answerer_from_env

load_dotenv()
logger = logging.getLogger("assistant")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Source(BaseModel):
    id: str
    category: str
    question: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    engine: str


def create_app(kb: KnowledgeBase | None = None, answerer: Answerer | None = None) -> FastAPI:
    kb = kb or KnowledgeBase.from_file()
    answerer = answerer or answerer_from_env()
    app = FastAPI(
        title="watsonx EHR Support Assistant",
        description="RAG support assistant for EHR end users, powered by IBM watsonx.ai.",
        version="1.0.0",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "engine": answerer.name, "articles": len(kb.entries)}

    @app.get("/categories")
    def categories() -> list[str]:
        return sorted({e.category for e in kb.entries})

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        question = req.question.strip()
        if len(question) < 3:
            raise HTTPException(status_code=422, detail="Question is too short")
        results = kb.search(question)
        engine = answerer.name
        try:
            answer = answerer.answer(question, results)
        except requests.RequestException as exc:
            # If watsonx.ai is unreachable, fall back to the knowledge base answer.
            logger.warning("watsonx.ai call failed, using offline answer: %s", exc)
            answer = OfflineAnswerer().answer(question, results)
            engine = "offline-fallback"
        return AskResponse(
            answer=answer,
            engine=engine,
            sources=[
                Source(id=r.entry.id, category=r.entry.category, question=r.entry.question, score=r.score)
                for r in results
            ],
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
