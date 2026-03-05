from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

LOGGER = logging.getLogger("hotstar_qdrant")


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class QdrantCatalogStore:
    def __init__(self, url: str, collection_name: str) -> None:
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def ensure_collection(self, vector_size: int, distance: models.Distance = models.Distance.COSINE) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            LOGGER.info("Collection %s already exists", self.collection_name)
            return

        LOGGER.info("Creating collection %s with vector size %d", self.collection_name, vector_size)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )

    def upsert_batch(self, records: list[dict[str, Any]]) -> None:
        points: list[models.PointStruct] = []
        for row in records:
            vector = row.get("embedding")
            if not isinstance(vector, list) or not vector:
                LOGGER.warning("Skipping record with invalid embedding: %s", row.get("id"))
                continue

            record_id = row.get("id")
            point_id: int | str
            if isinstance(record_id, int):
                point_id = record_id
            elif isinstance(record_id, str) and record_id.isdigit():
                point_id = int(record_id)
            elif isinstance(record_id, str) and record_id.startswith("item-") and record_id[5:].isdigit():
                point_id = int(record_id[5:])
            elif record_id:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(record_id)))
            else:
                stable_source = str(row.get("url") or row.get("title") or uuid.uuid4())
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_source))

            payload = {
                "url": row.get("url"),
                "title": row.get("title"),
                "description": row.get("description"),
                "genre": row.get("genre"),
                "release_year": row.get("release_year"),
                "scraped_at": row.get("scraped_at"),
                "cast": row.get("cast"),
                "characters": row.get("characters"),
                "keywords": row.get("keywords"),
                "original_language": row.get("original_language"),
                "runtime_minutes": row.get("runtime_minutes"),
                "average_rating": row.get("average_rating"),
                "tmdb_id": row.get("tmdb_id"),
                "tmdb_media_type": row.get("tmdb_media_type"),
                "combined_text": row.get("combined_text"),
            }

            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        if not points:
            LOGGER.warning("No valid points to upsert in this batch")
            return

        self.client.upsert(collection_name=self.collection_name, points=points, wait=False)
        LOGGER.info("Upserted %d points", len(points))


def load_embedded_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Embedded data must be a JSON list")
    return [row for row in payload if isinstance(row, dict)]


def ingest_embeddings(
    embedded_path: Path,
    qdrant_url: str,
    collection_name: str,
    batch_size: int,
) -> None:
    rows = load_embedded_records(embedded_path)
    if not rows:
        raise ValueError("No embedded rows available for ingestion")

    first_vector = rows[0].get("embedding")
    if not isinstance(first_vector, list) or not first_vector:
        raise ValueError("First row does not contain a valid embedding")

    store = QdrantCatalogStore(url=qdrant_url, collection_name=collection_name)
    store.ensure_collection(vector_size=len(first_vector))

    effective_batch_size = max(batch_size, 1)
    for start in range(0, len(rows), effective_batch_size):
        batch = rows[start : start + effective_batch_size]
        store.upsert_batch(batch)

    LOGGER.info("Completed ingestion of %d records into %s", len(rows), collection_name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest embedded Hotstar records into Qdrant")
    parser.add_argument("--input", type=str, default="data/hotstar_embedded.json")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--collection", type=str, default="hotstar_catalog")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(verbose=args.verbose)

    try:
        ingest_embeddings(
            embedded_path=Path(args.input),
            qdrant_url=args.qdrant_url,
            collection_name=args.collection,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        LOGGER.exception("Qdrant ingestion failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
