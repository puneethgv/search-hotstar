from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

LOGGER = logging.getLogger("hotstar_embedder")


@dataclass(slots=True)
class ItemInput:
    id: str
    url: str
    title: str
    description: str
    genre: str
    release_year: int | None
    scraped_at: str | None
    cast: list[str]
    characters: list[str]
    keywords: list[str]
    original_language: str | None
    runtime_minutes: int | None
    average_rating: float | None

    @property
    def rich_text(self) -> str:
        cast_text = ", ".join(self.cast[:8]) if self.cast else "Unknown"
        character_text = ", ".join(self.characters[:8]) if self.characters else "Unknown"
        keyword_text = ", ".join(self.keywords[:12]) if self.keywords else "Unknown"
        year_text = str(self.release_year) if self.release_year is not None else "Unknown"
        runtime_text = str(self.runtime_minutes) if self.runtime_minutes is not None else "Unknown"
        rating_text = f"{self.average_rating:.1f}" if self.average_rating is not None else "Unknown"
        return (
            f"Title: {self.title}\n"
            f"Release Year: {year_text}\n"
            f"Genre: {self.genre or 'Unknown'}\n"
            f"Cast: {cast_text}\n"
            f"Characters: {character_text}\n"
            f"Keywords: {keyword_text}\n"
            f"Language: {self.original_language or 'Unknown'}\n"
            f"Runtime Minutes: {runtime_text}\n"
            f"Average Rating: {rating_text}\n"
            f"Description: {self.description or 'No description available.'}"
        )


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class HuggingFaceInferenceEmbeddingBackend:
    def __init__(self, api_key: str, model_name: str = "BAAI/bge-m3") -> None:
        self.client = InferenceClient(provider="hf-inference", api_key=api_key)
        self.model_name = model_name

    def _embed_once(self, text: str) -> list[float]:
        result = self.client.feature_extraction(text, model=self.model_name)
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

    def _embed_with_backoff(self, text: str, max_attempts: int = 6) -> list[float]:
        for attempt in range(1, max_attempts + 1):
            try:
                return self._embed_once(text)
            except Exception as exc:
                if attempt == max_attempts:
                    raise
                sleep_seconds = min(2 ** attempt, 30)
                LOGGER.warning(
                    "Embedding attempt %d/%d failed: %s. Retrying in %.2fs",
                    attempt,
                    max_attempts,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        raise RuntimeError("Unexpected embedding retry loop termination")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_with_backoff(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_with_backoff(text)


class SentenceTransformerEmbeddingBackend:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _safe_list_of_str(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def load_scraped_items(path: Path) -> list[ItemInput]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list of objects")

    items: list[ItemInput] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            continue
        url = _safe_str(row.get("url"))
        title = _safe_str(row.get("title"), default="Unknown Title")
        description = _safe_str(row.get("description"), default="")
        genre = _safe_str(row.get("genre"), default="")
        release_year_value = row.get("release_year")
        try:
            release_year = int(release_year_value) if release_year_value is not None else None
        except (TypeError, ValueError):
            release_year = None

        runtime_raw = row.get("runtime_minutes")
        try:
            runtime_minutes = int(runtime_raw) if runtime_raw is not None else None
        except (TypeError, ValueError):
            runtime_minutes = None

        rating_raw = row.get("average_rating")
        try:
            average_rating = float(rating_raw) if rating_raw is not None else None
        except (TypeError, ValueError):
            average_rating = None

        items.append(
            ItemInput(
                id=_safe_str(row.get("id"), default=f"item-{idx}"),
                url=url,
                title=title,
                description=description,
                genre=genre,
                release_year=release_year,
                scraped_at=_safe_str(row.get("scraped_at")) or None,
                cast=_safe_list_of_str(row.get("cast")),
                characters=_safe_list_of_str(row.get("characters")),
                keywords=_safe_list_of_str(row.get("keywords")),
                original_language=_safe_str(row.get("original_language")) or None,
                runtime_minutes=runtime_minutes,
                average_rating=average_rating,
            )
        )
    return items


T = TypeVar("T")


def batched(values: list[T], batch_size: int) -> list[list[T]]:
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


def build_backend(backend: str, hf_model: str) -> EmbeddingBackend:
    if backend == "hf-inference":
        api_key = os.getenv("HF_TOKEN", "").strip()
        if not api_key:
            raise ValueError("HF_TOKEN is required when backend=hf-inference")
        return HuggingFaceInferenceEmbeddingBackend(api_key=api_key, model_name=hf_model)

    if backend == "local":
        return SentenceTransformerEmbeddingBackend()

    raise ValueError(f"Unsupported backend: {backend}")


def write_output(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(
    input_path: Path,
    output_path: Path,
    backend_name: str,
    hf_model: str,
    batch_size: int,
) -> None:
    items = load_scraped_items(input_path)
    LOGGER.info("Loaded %d scraped items", len(items))

    backend = build_backend(backend=backend_name, hf_model=hf_model)
    output: list[dict[str, Any]] = []

    item_batches = batched(items, batch_size=max(batch_size, 1))
    for batch_index, item_batch in enumerate(item_batches, start=1):
        texts = [item.rich_text for item in item_batch]
        LOGGER.info("Embedding batch %d/%d (size=%d)", batch_index, len(item_batches), len(texts))
        vectors = backend.embed_documents(texts)
        for item, vector in zip(item_batch, vectors):
            output.append(
                {
                    "id": item.id,
                    "url": item.url,
                    "title": item.title,
                    "description": item.description,
                    "genre": item.genre,
                    "release_year": item.release_year,
                    "scraped_at": item.scraped_at,
                    "cast": item.cast,
                    "characters": item.characters,
                    "keywords": item.keywords,
                    "original_language": item.original_language,
                    "runtime_minutes": item.runtime_minutes,
                    "average_rating": item.average_rating,
                    "combined_text": item.rich_text,
                    "embedding": vector,
                }
            )

    write_output(output, output_path)
    LOGGER.info("Saved %d embedded records to %s", len(output), output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate embeddings for Hotstar catalog JSON")
    parser.add_argument("--input", type=str, default="data/hotstar_scraped.json")
    parser.add_argument("--output", type=str, default="data/hotstar_embedded.json")
    parser.add_argument("--backend", choices=["hf-inference", "local"], default="hf-inference")
    parser.add_argument("--hf-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(verbose=args.verbose)

    try:
        run_pipeline(
            input_path=Path(args.input),
            output_path=Path(args.output),
            backend_name=args.backend,
            hf_model=args.hf_model,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        LOGGER.exception("Embedding pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
