from __future__ import annotations

import argparse
import ast
import difflib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("hotstar_enricher")

LANG_NAME_TO_CODE: dict[str, str] = {
    "english": "en",
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
    "kannada": "kn",
    "bengali": "bn",
    "marathi": "mr",
    "gujarati": "gu",
    "punjabi": "pa",
    "urdu": "ur",
    "japanese": "ja",
    "korean": "ko",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "russian": "ru",
    "chinese": "zh",
}


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def safe_literal_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
    except (SyntaxError, ValueError):
        return []
    return []


def clean_hotstar_title(raw_title: str | None) -> str:
    if not raw_title:
        return ""
    title = raw_title.strip()
    title = re.sub(r"\s*[-–—]\s*(Disney\+\s*)?Hotstar\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def normalize_title(title: str | None) -> str:
    cleaned = clean_hotstar_title(title).lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@dataclass(slots=True)
class MovieRow:
    movie_id: int
    title: str
    normalized_title: str
    release_year: int | None
    genres: list[str]
    original_language: str | None
    runtime_minutes: int | None
    average_rating: float | None


def parse_year(release_date: Any) -> int | None:
    if release_date is None:
        return None
    text = str(release_date).strip()
    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group()) if match else None


def parse_number(value: Any, as_int: bool = False) -> float | int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text)) if as_int else float(text)
    except ValueError:
        return None


def infer_language_code(record: dict[str, Any]) -> str | None:
    existing = str(record.get("original_language") or "").strip().lower()
    if existing:
        return existing

    text = " ".join(
        [
            str(record.get("title") or ""),
            str(record.get("description") or ""),
        ]
    ).lower()
    for lang_name, code in LANG_NAME_TO_CODE.items():
        if re.search(rf"\b{re.escape(lang_name)}\b", text):
            return code
    return None


def load_movies_index(metadata_csv: Path) -> tuple[dict[int, MovieRow], dict[str, list[MovieRow]], list[str]]:
    df = pd.read_csv(
        metadata_csv,
        usecols=["id", "title", "genres", "release_date", "original_language", "runtime", "vote_average"],
        low_memory=False,
    )

    by_id: dict[int, MovieRow] = {}
    by_title: dict[str, list[MovieRow]] = {}

    for _, row in df.iterrows():
        try:
            movie_id = int(str(row.get("id")).strip())
        except (TypeError, ValueError):
            continue

        title = str(row.get("title") or "").strip()
        normalized = normalize_title(title)
        genres = [g.get("name", "").strip() for g in safe_literal_list(row.get("genres")) if g.get("name")]
        movie = MovieRow(
            movie_id=movie_id,
            title=title,
            normalized_title=normalized,
            release_year=parse_year(row.get("release_date")),
            genres=genres,
            original_language=(str(row.get("original_language")).strip() if pd.notna(row.get("original_language")) else None),
            runtime_minutes=parse_number(row.get("runtime"), as_int=True),
            average_rating=parse_number(row.get("vote_average"), as_int=False),
        )
        by_id[movie_id] = movie
        if normalized:
            by_title.setdefault(normalized, []).append(movie)

    all_titles = list(by_title.keys())
    return by_id, by_title, all_titles


def fuzzy_candidates(
    normalized_title: str,
    by_title: dict[str, list[MovieRow]],
    all_titles: list[str],
    max_results: int = 8,
) -> list[MovieRow]:
    if not normalized_title:
        return []

    close = difflib.get_close_matches(normalized_title, all_titles, n=max_results, cutoff=0.84)
    output: list[MovieRow] = []
    for key in close:
        output.extend(by_title.get(key, []))
    return output


def load_keywords_index(keywords_csv: Path) -> dict[int, list[str]]:
    df = pd.read_csv(keywords_csv, usecols=["id", "keywords"], low_memory=False)
    result: dict[int, list[str]] = {}
    for _, row in df.iterrows():
        try:
            movie_id = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        names = [k.get("name", "").strip() for k in safe_literal_list(row.get("keywords")) if k.get("name")]
        result[movie_id] = names[:20]
    return result


def load_credits_index(credits_csv: Path) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    df = pd.read_csv(credits_csv, usecols=["id", "cast"], low_memory=False)
    cast_by_id: dict[int, list[str]] = {}
    characters_by_id: dict[int, list[str]] = {}

    for _, row in df.iterrows():
        try:
            movie_id = int(row.get("id"))
        except (TypeError, ValueError):
            continue

        entries = safe_literal_list(row.get("cast"))
        cast_names = [e.get("name", "").strip() for e in entries if e.get("name")][:12]
        character_names = [e.get("character", "").strip() for e in entries if e.get("character")][:12]
        cast_by_id[movie_id] = cast_names
        characters_by_id[movie_id] = character_names

    return cast_by_id, characters_by_id


def pick_best_candidate(
    candidates: list[MovieRow],
    release_year: int | None,
    source_language: str | None,
) -> MovieRow | None:
    if not candidates:
        return None

    def score(row: MovieRow) -> float:
        score_value = 0.0

        if release_year is not None:
            if row.release_year is None:
                score_value -= 4.0
            else:
                delta = abs(row.release_year - release_year)
                if delta > 5:
                    score_value -= 8.0
                else:
                    score_value += max(0.0, 6.0 - float(delta))

        if source_language:
            row_lang = (row.original_language or "").lower()
            if row_lang == source_language:
                score_value += 4.0
            elif row_lang:
                score_value -= 1.5

        if row.average_rating is not None:
            score_value += min(row.average_rating / 10.0, 1.0)

        return score_value

    return sorted(candidates, key=score, reverse=True)[0]


def enrich_records(
    records: list[dict[str, Any]],
    by_title: dict[str, list[MovieRow]],
    all_titles: list[str],
    keywords_by_id: dict[int, list[str]],
    cast_by_id: dict[int, list[str]],
    characters_by_id: dict[int, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counters = {
        "matched": 0,
        "genre_filled": 0,
        "cast_filled": 0,
        "keywords_filled": 0,
        "year_filled": 0,
        "fuzzy_matched": 0,
    }

    enriched: list[dict[str, Any]] = []
    for row in records:
        normalized = normalize_title(row.get("title"))
        source_year = row.get("release_year")
        source_language = infer_language_code(row)
        try:
            source_year_int = int(source_year) if source_year is not None else None
        except (TypeError, ValueError):
            source_year_int = None

        candidates = by_title.get(normalized, [])
        candidate = pick_best_candidate(candidates, source_year_int, source_language)
        if candidate is None:
            fuzzy = fuzzy_candidates(normalized, by_title=by_title, all_titles=all_titles)
            candidate = pick_best_candidate(fuzzy, source_year_int, source_language)
            if candidate is not None:
                counters["fuzzy_matched"] += 1
        if not candidate:
            enriched.append(row)
            continue

        counters["matched"] += 1
        movie_id = candidate.movie_id

        genres = candidate.genres
        cast = cast_by_id.get(movie_id, [])
        characters = characters_by_id.get(movie_id, [])
        keywords = keywords_by_id.get(movie_id, [])

        if genres:
            row["genre"] = ", ".join(genres)
            counters["genre_filled"] += 1
        if cast:
            row["cast"] = cast
            counters["cast_filled"] += 1
        if characters:
            row["characters"] = characters
        if keywords:
            row["keywords"] = keywords
            counters["keywords_filled"] += 1

        if row.get("release_year") is None and candidate.release_year is not None:
            row["release_year"] = candidate.release_year
            counters["year_filled"] += 1

        row["tmdb_id"] = movie_id
        row["tmdb_media_type"] = "movie"
        row["original_language"] = candidate.original_language
        row["runtime_minutes"] = candidate.runtime_minutes
        row["average_rating"] = candidate.average_rating

        enriched.append(row)

    return enriched, counters


def is_high_quality(record: dict[str, Any]) -> bool:
    genre = str(record.get("genre") or "").strip()
    cast = record.get("cast")
    keywords = record.get("keywords")
    release_year = record.get("release_year")
    description = str(record.get("description") or "").strip()

    has_cast = isinstance(cast, list) and len(cast) >= 3
    has_keywords = isinstance(keywords, list) and len(keywords) >= 3
    has_genre = bool(genre)
    has_year = release_year is not None
    has_desc = len(description) >= 30
    return has_cast and has_keywords and has_genre and has_year and has_desc


def select_quality_records(records: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    seen_keys: set[str] = set()
    selected: list[dict[str, Any]] = []

    for row in records:
        if not is_high_quality(row):
            continue
        key = normalize_title(str(row.get("title") or ""))
        if not key:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(row)
        if len(selected) >= target_count:
            break
    return selected


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Input file must contain a JSON list")

    LOGGER.info("Loading public movie database CSV indexes...")
    _, by_title, all_titles = load_movies_index(Path(args.movies_metadata))
    keywords_by_id = load_keywords_index(Path(args.keywords))
    cast_by_id, characters_by_id = load_credits_index(Path(args.credits))

    LOGGER.info("Enriching %d scraped records...", len(records))
    enriched, counters = enrich_records(
        records=records,
        by_title=by_title,
        all_titles=all_titles,
        keywords_by_id=keywords_by_id,
        cast_by_id=cast_by_id,
        characters_by_id=characters_by_id,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.quality_output:
        quality_rows = select_quality_records(enriched, target_count=args.target_count)
        quality_output_path = Path(args.quality_output)
        quality_output_path.parent.mkdir(parents=True, exist_ok=True)
        quality_output_path.write_text(json.dumps(quality_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Saved %d high-quality records to %s", len(quality_rows), quality_output_path)

    LOGGER.info("Saved enriched dataset to %s", output_path)
    LOGGER.info("Matches: %d/%d", counters["matched"], len(records))
    LOGGER.info("Fuzzy matches: %d", counters["fuzzy_matched"])
    LOGGER.info("Genre filled: %d", counters["genre_filled"])
    LOGGER.info("Cast filled: %d", counters["cast_filled"])
    LOGGER.info("Keywords filled: %d", counters["keywords_filled"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich scraped Hotstar data using public movie metadata CSVs")
    parser.add_argument("--input", type=str, default="data/hotstar_scraped_test_1000.json")
    parser.add_argument("--output", type=str, default="data/hotstar_scraped_test_1000_enriched.json")
    parser.add_argument("--movies-metadata", type=str, default="data/movies_metadata.csv")
    parser.add_argument("--credits", type=str, default="data/credits.csv")
    parser.add_argument("--keywords", type=str, default="data/keywords.csv")
    parser.add_argument("--quality-output", type=str, default="data/hotstar_quality_5000.json")
    parser.add_argument("--target-count", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(verbose=args.verbose)
    run(args)


if __name__ == "__main__":
    main()
