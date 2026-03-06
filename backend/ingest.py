from __future__ import annotations

import ast
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = "BAAI/bge-m3"
VECTOR_SIZE = 1024
BATCH_SIZE = 4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "movies_metadata.csv"
COLLECTION_NAME = "movies"
QDRANT_URL = "http://localhost:6333"

client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
q_client = QdrantClient(url=QDRANT_URL)


def _embed_one(text: str) -> list[float] | None:
    for attempt in range(5):
        try:
            result = client.feature_extraction(text, model=HF_MODEL)
            array = result if isinstance(result, np.ndarray) else np.array(result)

            if array.ndim == 3:
                return np.mean(array, axis=1)[0].tolist()
            if array.ndim == 2:
                return array.tolist()
            if array.ndim == 1:
                return array.tolist()
            return None
        except Exception as exc:
            error_msg = str(exc)
            if "503" in error_msg:
                wait_seconds = 10 * (attempt + 1)
                print(f"Model is loading (503). Waiting {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue
            if "504" in error_msg:
                wait_seconds = 20 * (attempt + 1)
                print(f"Gateway Time-out (504). Waiting {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue
            if "429" in error_msg:
                print("Rate limit reached (429). Waiting 60s...")
                time.sleep(60)
                continue

            print(f"Error embedding attempt {attempt + 1}: {exc}")
            time.sleep(5)

    return None


def get_embeddings(texts: str | list[str]) -> list[list[float]] | None:
    if isinstance(texts, str):
        texts = [texts]

    embeddings: list[list[float]] = []
    for text in texts:
        vector = _embed_one(text)
        if vector is None:
            return None
        embeddings.append(vector)
    return embeddings


def parse_genres(genre_str: str) -> str:
    try:
        genres = ast.literal_eval(genre_str)
        if not isinstance(genres, list):
            return ""
        return ", ".join([g["name"] for g in genres if isinstance(g, dict) and "name" in g])
    except (SyntaxError, ValueError, TypeError):
        return ""


def main() -> None:
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH, low_memory=False, nrows=1000)

    df["overview"] = df["overview"].fillna("")
    df["title"] = df["title"].fillna("unknown title")
    df["genres_clean"] = df["genres"].apply(parse_genres)

    print(f"Creating collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}...")
    q_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    print("Starting ingestion with batching...")
    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_df = df.iloc[start : start + BATCH_SIZE]

        texts_to_embed: list[str] = []
        indices: list[int] = []
        payloads: list[dict[str, str]] = []

        for offset, (_, row) in enumerate(batch_df.iterrows()):
            texts_to_embed.append(f"{row['title']}: {row['overview']}")
            indices.append(start + offset)
            payloads.append(
                {
                    "title": row["title"],
                    "overview": row["overview"],
                    "genres": row["genres_clean"],
                }
            )

        if not texts_to_embed:
            continue

        embeddings = get_embeddings(texts_to_embed)
        if not embeddings or len(embeddings) != len(texts_to_embed):
            print(f"Failed to process batch {start}")
            continue

        batch_points = [
            PointStruct(id=indices[index], vector=embedding, payload=payloads[index])
            for index, embedding in enumerate(embeddings)
        ]
        q_client.upsert(collection_name=COLLECTION_NAME, points=batch_points)

    print("Ingestion complete!")


if __name__ == "__main__":
    print("Testing embedding API with BAAI/bge-m3...")
    test_embedding = get_embeddings(["test"])
    if test_embedding:
        print(f"API working. Vector dim: {len(test_embedding[0])}")
        main()
    else:
        print("API check failed. Please check your HF_TOKEN.")
