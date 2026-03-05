from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

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


app = FastAPI(title="Hotstar Semantic Search API", version="1.0.0")
config = AppConfig()
query_cache = QueryEmbeddingCache(max_size=config.max_cache_size)


@app.on_event("startup")
def startup_event() -> None:
    configure_logging()
    LOGGER.info("Starting Hotstar Semantic Search API")

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


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=2, description="Search query text"),
    top_k: int = Query(default=5, ge=1, le=20),
    year: int | None = Query(default=None, ge=1900, le=2100, description="Optional release year filter"),
    language: str | None = Query(default=None, description="Optional language code filter (e.g., en, hi, ta)"),
) -> SearchResponse:
    embedder: HuggingFaceQueryEmbedder = app.state.embedder
    qdrant: QdrantClient = app.state.qdrant

    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

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
        points = qdrant.search(
            collection_name=config.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        LOGGER.exception("Qdrant search failed: %s", exc)
        raise HTTPException(status_code=500, detail="Search failed") from exc

    return SearchResponse(
        query=query,
        top_k=top_k,
        search_ms=round(elapsed_ms, 3),
        results=_build_search_hits(points),
    )
