from __future__ import annotations

import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

LOGGER = logging.getLogger("hotstar_api")


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class SearchHit(BaseModel):
    id: str
    score: float
    title: str | None = None
    url: str | None = None
    description: str | None = None
    genre: str | None = None
    release_year: int | None = None
    cast: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    original_language: str | None = None
    runtime_minutes: int | None = None
    average_rating: float | None = None
    tmdb_id: int | None = None
    tmdb_media_type: str | None = None


class SearchResponse(BaseModel):
    query: str
    k: int
    top_k: int
    search_ms: float
    results: list[SearchHit]


@dataclass(slots=True)
class AppConfig:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "hotstar_catalog")
    hf_model: str = os.getenv("HF_MODEL", "BAAI/bge-m3")
    hf_token: str = os.getenv("HF_TOKEN", "")
    max_cache_size: int = int(os.getenv("QUERY_EMBED_CACHE_SIZE", "2048"))


class QueryEmbeddingCache:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._data: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: list[float]) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self.max_size:
                self._data.popitem(last=False)


class HuggingFaceQueryEmbedder:
    def __init__(self, api_key: str, model_name: str) -> None:
        if not api_key.strip():
            raise ValueError("HF_TOKEN is required")
        self.client = InferenceClient(provider="hf-inference", api_key=api_key)
        self.model_name = model_name

    def _embed_once(self, query: str) -> list[float]:
        result = self.client.feature_extraction(query, model=self.model_name)
        if not isinstance(result, np.ndarray):
            result = np.array(result)

        if result.ndim == 1:
            vector = result
        elif result.ndim == 2:
            vector = result.mean(axis=0)
        elif result.ndim == 3:
            vector = result.mean(axis=1)[0]
        else:
            raise ValueError(f"Unexpected embedding shape: {result.shape}")

        return [float(v) for v in vector.tolist()]

    def embed_query(self, query: str, max_attempts: int = 4) -> list[float]:
        for attempt in range(1, max_attempts + 1):
            try:
                return self._embed_once(query)
            except Exception as exc:
                if attempt == max_attempts:
                    raise
                sleep_seconds = min(2 ** attempt, 10)
                LOGGER.warning(
                    "Query embedding failed attempt %d/%d: %s; retrying in %.2fs",
                    attempt,
                    max_attempts,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        raise RuntimeError("Embedding retry loop exited unexpectedly")


app = FastAPI(title="Search-Hotstar API", version="1.0.0")
config = AppConfig()
query_cache = QueryEmbeddingCache(max_size=config.max_cache_size)

frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    configure_logging()
    LOGGER.info("Starting Search-Hotstar API")

    app.state.qdrant = QdrantClient(url=config.qdrant_url)
    app.state.embedder = HuggingFaceQueryEmbedder(api_key=config.hf_token, model_name=config.hf_model)

    try:
        app.state.qdrant.get_collection(collection_name=config.collection_name)
        LOGGER.info("Qdrant collection ready: %s", config.collection_name)
    except Exception as exc:
        LOGGER.exception("Qdrant collection check failed: %s", exc)
        raise


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "Search-Hotstar API",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "search": "/search?q=telugu+movies&k=5",
            "docs": "/docs",
        },
    }


def _build_search_hits(points: list[Any]) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for point in points:
        payload = point.payload or {}
        hits.append(
            SearchHit(
                id=str(point.id),
                score=float(point.score),
                title=payload.get("title"),
                url=payload.get("url"),
                description=payload.get("description"),
                genre=payload.get("genre"),
                release_year=payload.get("release_year"),
                cast=payload.get("cast") or [],
                characters=payload.get("characters") or [],
                keywords=payload.get("keywords") or [],
                original_language=payload.get("original_language"),
                runtime_minutes=payload.get("runtime_minutes"),
                average_rating=payload.get("average_rating"),
                tmdb_id=payload.get("tmdb_id"),
                tmdb_media_type=payload.get("tmdb_media_type"),
            )
        )
    return hits


def _compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _lexical_boost(query: str, payload: dict[str, Any], vector_score: float) -> float:
    query_text = query.strip().lower()
    query_compact = _compact_text(query_text)
    if not query_compact:
        return vector_score

    title = str(payload.get("title") or "")
    description = str(payload.get("description") or "")
    genre = str(payload.get("genre") or "")
    cast = [str(item) for item in (payload.get("cast") or [])]
    keywords = [str(item) for item in (payload.get("keywords") or [])]

    searchable_parts = [title, description, genre, *cast, *keywords]
    searchable_text = " ".join(searchable_parts).lower()
    searchable_compact = _compact_text(searchable_text)

    boost = 0.0

    if query_compact and query_compact in searchable_compact:
        boost += 0.35

    query_tokens = [token for token in re.findall(r"[a-z0-9]+", query_text) if len(token) > 1]
    if query_tokens:
        token_hits = sum(1 for token in query_tokens if token in searchable_text)
        boost += min(0.2, token_hits * 0.06)

    phrase_candidates = [title, *cast, *keywords]
    max_ratio = 0.0
    for phrase in phrase_candidates:
        compact_phrase = _compact_text(phrase)
        if not compact_phrase:
            continue
        ratio = SequenceMatcher(None, query_compact, compact_phrase).ratio()
        if ratio > max_ratio:
            max_ratio = ratio

    has_query_whitespace = len(query_text.split()) > 1

    if max_ratio >= 0.9:
        boost += 0.3
    elif max_ratio >= 0.8 and (token_hits > 0 or has_query_whitespace):
        boost += 0.18
    elif max_ratio >= 0.7 and token_hits > 0:
        boost += 0.08

    return vector_score + boost


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query text"),
    k: int | None = Query(default=None, ge=1, le=100, description="Number of results to return"),
    top_k: int | None = Query(default=None, ge=1, le=100, description="Deprecated alias for k"),
    year: int | None = Query(default=None, ge=1900, le=2100, description="Optional release year filter"),
    language: str | None = Query(default=None, description="Optional language code filter (e.g., en, hi, ta)"),
) -> SearchResponse:
    embedder: HuggingFaceQueryEmbedder = app.state.embedder
    qdrant: QdrantClient = app.state.qdrant

    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if k is not None and top_k is not None and k != top_k:
        raise HTTPException(status_code=400, detail="Provide either k or top_k, or set both to same value")

    effective_k = k if k is not None else top_k
    if effective_k is None:
        effective_k = 5

    query_vector = query_cache.get(query)
    if query_vector is None:
        try:
            query_vector = embedder.embed_query(query)
            query_cache.set(query, query_vector)
        except Exception as exc:
            LOGGER.exception("Failed to embed query: %s", exc)
            raise HTTPException(status_code=502, detail="Embedding service unavailable") from exc

    try:
        start = time.perf_counter()
        conditions: list[qdrant_models.FieldCondition] = []
        if year is not None:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="release_year",
                    match=qdrant_models.MatchValue(value=year),
                )
            )
        if language:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="original_language",
                    match=qdrant_models.MatchValue(value=language.strip().lower()),
                )
            )

        query_filter = qdrant_models.Filter(must=conditions) if conditions else None
        candidate_limit = min(max(effective_k * 5, 25), 100)

        if hasattr(qdrant, "search"):
            points = qdrant.search(
                collection_name=config.collection_name,
                query_vector=query_vector,
                limit=candidate_limit,
                with_payload=True,
                query_filter=query_filter,
            )
        else:
            response = qdrant.query_points(
                collection_name=config.collection_name,
                query=query_vector,
                limit=candidate_limit,
                with_payload=True,
                query_filter=query_filter,
            )
            points = response.points

        ranked_points = sorted(
            points,
            key=lambda point: _lexical_boost(
                query,
                point.payload or {},
                float(point.score),
            ),
            reverse=True,
        )
        points = ranked_points[:effective_k]

        elapsed_ms = (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        LOGGER.exception("Qdrant search failed: %s", exc)
        raise HTTPException(status_code=500, detail="Search failed") from exc

    return SearchResponse(
        query=query,
        k=effective_k,
        top_k=effective_k,
        search_ms=round(elapsed_ms, 3),
        results=_build_search_hits(points),
    )
