from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.parse import urlparse, urlunparse

import aiohttp
from bs4 import BeautifulSoup
from bs4.element import Tag
from dotenv import load_dotenv
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

LOGGER = logging.getLogger("hotstar_scraper")


DEFAULT_REGIONS: list[str] = ["in"]
DEFAULT_SITEMAP_HINTS: list[str] = [
    "MOVIE",
    "SHOWS",
    "SERIES",
]

FALLBACK_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": "https://www.hotstar.com/in",
}


@dataclass(slots=True)
class CatalogItem:
    url: str
    title: str | None
    description: str | None
    genre: str | None
    release_year: int | None
    cast: list[str]
    characters: list[str]
    tmdb_id: int | None
    tmdb_media_type: str | None
    keywords: list[str]
    original_language: str | None
    runtime_minutes: int | None
    average_rating: float | None
    scraped_at: str


@dataclass(slots=True)
class SitemapDoc:
    kind: str
    locs: list[str]


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_sitemap_document(xml_text: str) -> SitemapDoc:
    root = ElementTree.fromstring(xml_text)
    namespace_match = re.match(r"\{(.*)\}", root.tag)
    namespace = {"sm": namespace_match.group(1)} if namespace_match else {}

    root_name = root.tag.split("}")[-1].lower()
    if root_name == "sitemapindex":
        path = ".//sm:sitemap/sm:loc" if namespace else ".//sitemap/loc"
        locs = [((node.text or "").strip()) for node in root.findall(path, namespace)]
        return SitemapDoc(kind="index", locs=[loc for loc in locs if loc])

    if root_name == "urlset":
        path = ".//sm:url/sm:loc" if namespace else ".//url/loc"
        locs = [((node.text or "").strip()) for node in root.findall(path, namespace)]
        return SitemapDoc(kind="urlset", locs=[loc for loc in locs if loc])

    path = ".//sm:loc" if namespace else ".//loc"
    locs = [((node.text or "").strip()) for node in root.findall(path, namespace)]
    return SitemapDoc(kind="unknown", locs=[loc for loc in locs if loc])


def extract_sitemaps_from_robots(robots_text: str) -> list[str]:
    matches = re.findall(r"(?im)^\s*Sitemap\s*:\s*(https?://\S+)\s*$", robots_text)
    return list(dict.fromkeys(matches))


def sitemap_matches_hints(url: str, hints: list[str]) -> bool:
    upper_url = url.upper()
    return any(hint.upper() in upper_url for hint in hints)


def canonicalize_content_url(url: str) -> str:
    parsed = urlparse(url.strip())
    path = re.sub(r"/+", "/", parsed.path)

    if path.endswith("/watch"):
        path = path[: -len("/watch")]
    path = path.rstrip("/")
    if not path:
        path = "/"

    canonical = parsed._replace(path=path, query="", fragment="")
    return urlunparse(canonical)


class TmdbEnricher:
    def __init__(self, api_key: str, concurrency: int = 5) -> None:
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.semaphore = asyncio.Semaphore(max(concurrency, 1))

    async def _request_json(self, session: aiohttp.ClientSession, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        query_params = {"api_key": self.api_key, **params}
        url = f"{self.base_url}{endpoint}"
        async with self.semaphore:
            for attempt in range(1, 4):
                try:
                    timeout = aiohttp.ClientTimeout(total=20)
                    async with session.get(url, params=query_params, timeout=timeout) as response:
                        if response.status in {429, 500, 502, 503, 504} and attempt < 3:
                            await asyncio.sleep(2 ** (attempt - 1))
                            continue
                        response.raise_for_status()
                        payload = await response.json()
                        if isinstance(payload, dict):
                            return payload
                        return {}
                except aiohttp.ClientError as exc:
                    if attempt == 3:
                        LOGGER.warning("TMDB request failed (%s): %s", endpoint, exc)
                        return {}
                    await asyncio.sleep(2 ** (attempt - 1))
        return {}

    async def enrich(
        self,
        session: aiohttp.ClientSession,
        title: str | None,
        release_year: int | None,
    ) -> dict[str, Any]:
        if not title:
            return {}

        search_payload = await self._request_json(
            session,
            endpoint="/search/multi",
            params={"query": title, "include_adult": "false", "language": "en-US"},
        )

        results = search_payload.get("results")
        if not isinstance(results, list) or not results:
            return {}

        best = self._pick_best_result(results, title=title, release_year=release_year)
        if not best:
            return {}

        media_type = str(best.get("media_type") or "")
        if media_type not in {"movie", "tv"}:
            return {}

        tmdb_id = best.get("id")
        if not isinstance(tmdb_id, int):
            return {}

        details = await self._request_json(
            session,
            endpoint=f"/{media_type}/{tmdb_id}",
            params={"append_to_response": "credits,keywords"},
        )

        credits_raw = details.get("credits")
        credits = credits_raw if isinstance(credits_raw, dict) else {}
        cast_raw = credits.get("cast")
        cast_rows = cast_raw if isinstance(cast_raw, list) else []
        cast = [str(row.get("name")) for row in cast_rows[:10] if isinstance(row, dict) and row.get("name")]
        characters = [
            str(row.get("character"))
            for row in cast_rows[:10]
            if isinstance(row, dict) and row.get("character")
        ]

        keyword_rows: list[Any] = []
        keywords_payload = details.get("keywords")
        if isinstance(keywords_payload, dict):
            keywords_list = keywords_payload.get("keywords")
            results_list = keywords_payload.get("results")
            if isinstance(keywords_list, list):
                keyword_rows = keywords_list
            elif isinstance(results_list, list):
                keyword_rows = results_list
        keywords = [str(k.get("name")) for k in keyword_rows[:20] if isinstance(k, dict) and k.get("name")]

        runtime = details.get("runtime") if isinstance(details.get("runtime"), int) else None
        episode_run_time = details.get("episode_run_time")
        if runtime is None and isinstance(episode_run_time, list):
            run_times = [v for v in episode_run_time if isinstance(v, int)]
            runtime = run_times[0] if run_times else None

        vote_average = details.get("vote_average")
        rating = float(vote_average) if isinstance(vote_average, (float, int)) else None

        return {
            "tmdb_id": tmdb_id,
            "tmdb_media_type": media_type,
            "cast": cast,
            "characters": characters,
            "keywords": keywords,
            "original_language": details.get("original_language"),
            "runtime_minutes": runtime,
            "average_rating": rating,
        }

    @staticmethod
    def _pick_best_result(results: list[Any], title: str, release_year: int | None) -> dict[str, Any] | None:
        target = title.strip().lower()
        best: dict[str, Any] | None = None
        best_score = float("-inf")

        for row in results:
            if not isinstance(row, dict):
                continue
            media_type = row.get("media_type")
            if media_type not in {"movie", "tv"}:
                continue

            candidate_title = str(row.get("title") or row.get("name") or "").strip().lower()
            if not candidate_title:
                continue

            score = 0.0
            if candidate_title == target:
                score += 10.0
            elif target in candidate_title or candidate_title in target:
                score += 6.0

            popularity = row.get("popularity")
            if isinstance(popularity, (int, float)):
                score += min(float(popularity) / 100.0, 2.0)

            if release_year is not None:
                date_value = str(row.get("release_date") or row.get("first_air_date") or "")
                match = re.search(r"(19|20)\d{2}", date_value)
                if match:
                    year = int(match.group())
                    score += max(0.0, 3.0 - abs(year - release_year) * 0.5)

            if score > best_score:
                best_score = score
                best = row

        return best


def _first_text(soup: BeautifulSoup, selectors: list[tuple[str, dict[str, Any]]]) -> str | None:
    for name, attrs in selectors:
        node = soup.find(name, attrs=attrs)
        if isinstance(node, Tag):
            if node.has_attr("content"):
                content = str(node.get("content", "")).strip()
                if content:
                    return content
            text = node.get_text(" ", strip=True)
            if text:
                return text
    return None


def _extract_release_year(soup: BeautifulSoup, fallback_text: str) -> int | None:
    candidate = _first_text(
        soup,
        [
            ("meta", {"property": "video:release_date"}),
            ("meta", {"name": "release_date"}),
            ("meta", {"name": "datePublished"}),
        ],
    )
    if candidate:
        year_match = re.search(r"(19|20)\d{2}", candidate)
        if year_match:
            return int(year_match.group())

    match = re.search(r"(19|20)\d{2}", fallback_text)
    return int(match.group()) if match else None


def _extract_json_ld(soup: BeautifulSoup) -> dict[str, Any]:
    for node in soup.find_all("script", attrs={"type": "application/ld+json"}):
        if not isinstance(node, Tag):
            continue
        try:
            if not node.string:
                continue
            parsed = json.loads(node.string)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        return item
        except (json.JSONDecodeError, TypeError):
            continue
    return {}


def parse_catalog_item(url: str, html: str) -> CatalogItem:
    soup = BeautifulSoup(html, "html.parser")
    json_ld = _extract_json_ld(soup)

    title = (
        json_ld.get("name")
        or _first_text(soup, [("meta", {"property": "og:title"}), ("title", {})])
        or None
    )

    description = (
        json_ld.get("description")
        or _first_text(
            soup,
            [
                ("meta", {"property": "og:description"}),
                ("meta", {"name": "description"}),
            ],
        )
        or None
    )

    genre_val = json_ld.get("genre")
    if isinstance(genre_val, list):
        genre = ", ".join(str(item) for item in genre_val if item)
    elif isinstance(genre_val, str):
        genre = genre_val
    else:
        genre = _first_text(soup, [("meta", {"property": "video:tag"}), ("meta", {"name": "genre"})])

    doc_text = soup.get_text(" ", strip=True)
    release_year = _extract_release_year(soup, doc_text)

    return CatalogItem(
        url=url,
        title=title,
        description=description,
        genre=genre,
        release_year=release_year,
        cast=[],
        characters=[],
        tmdb_id=None,
        tmdb_media_type=None,
        keywords=[],
        original_language=None,
        runtime_minutes=None,
        average_rating=None,
        scraped_at=datetime.now(tz=timezone.utc).isoformat(),
    )


async def fetch_text(
    session: aiohttp.ClientSession,
    url: str,
    timeout_seconds: int = 30,
    headers: dict[str, str] | None = None,
) -> str:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with session.get(url, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        return await response.text()


async def discover_latest_sitemaps(
    session: aiohttp.ClientSession,
    regions: list[str],
    hints: list[str],
) -> list[str]:
    discovered: list[str] = []
    for region in regions:
        robots_url = f"https://www.hotstar.com/{region}/robots.txt"
        try:
            robots_text = await fetch_text(session, robots_url)
            robots_sitemaps = extract_sitemaps_from_robots(robots_text)
            region_sitemaps = [url for url in robots_sitemaps if f"/{region}/" in url]
            discovered.extend(region_sitemaps)
            LOGGER.info("Discovered %d region sitemap roots from %s", len(region_sitemaps), robots_url)
        except Exception as exc:
            LOGGER.warning("Failed to parse robots for region %s: %s", region, exc)

    if not discovered:
        discovered = ["https://www.hotstar.com/in/new-sitemap.xml"]

    final_sitemaps: list[str] = []
    queue = list(dict.fromkeys(discovered))
    seen: set[str] = set()

    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)

        try:
            xml_text = await fetch_text(session, current)
            doc = parse_sitemap_document(xml_text)
        except Exception as exc:
            LOGGER.warning("Failed to load sitemap %s: %s", current, exc)
            continue

        if doc.kind == "index":
            nested = [urljoin(current, loc) for loc in doc.locs]
            if hints:
                nested = [s for s in nested if sitemap_matches_hints(s, hints)]
            nested = [s for s in nested if "WATCH_PAGES" not in s.upper()]
            queue.extend(nested)
        else:
            final_sitemaps.append(current)

    return list(dict.fromkeys(final_sitemaps))


async def scrape_url(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    tmdb_enricher: TmdbEnricher | None = None,
    retry_attempts: int = 3,
) -> CatalogItem | None:
    async with semaphore:
        for attempt in range(1, retry_attempts + 1):
            try:
                html = await fetch_text(session, url)
                item = parse_catalog_item(url, html)
                if tmdb_enricher is not None:
                    enrichment = await tmdb_enricher.enrich(
                        session=session,
                        title=item.title,
                        release_year=item.release_year,
                    )
                    if enrichment:
                        item.cast = enrichment.get("cast", item.cast)
                        item.characters = enrichment.get("characters", item.characters)
                        item.tmdb_id = enrichment.get("tmdb_id", item.tmdb_id)
                        item.tmdb_media_type = enrichment.get("tmdb_media_type", item.tmdb_media_type)
                        item.keywords = enrichment.get("keywords", item.keywords)
                        item.original_language = enrichment.get("original_language", item.original_language)
                        item.runtime_minutes = enrichment.get("runtime_minutes", item.runtime_minutes)
                        item.average_rating = enrichment.get("average_rating", item.average_rating)
                return item
            except aiohttp.ClientResponseError as exc:
                status = exc.status
                if status in {404, 410}:
                    try:
                        html = await fetch_text(
                            session,
                            url,
                            headers=FALLBACK_BROWSER_HEADERS,
                        )
                        LOGGER.info("Recovered URL after fallback headers: %s", url)
                        item = parse_catalog_item(url, html)
                        if tmdb_enricher is not None:
                            enrichment = await tmdb_enricher.enrich(
                                session=session,
                                title=item.title,
                                release_year=item.release_year,
                            )
                            if enrichment:
                                item.cast = enrichment.get("cast", item.cast)
                                item.characters = enrichment.get("characters", item.characters)
                                item.tmdb_id = enrichment.get("tmdb_id", item.tmdb_id)
                                item.tmdb_media_type = enrichment.get("tmdb_media_type", item.tmdb_media_type)
                                item.keywords = enrichment.get("keywords", item.keywords)
                                item.original_language = enrichment.get("original_language", item.original_language)
                                item.runtime_minutes = enrichment.get("runtime_minutes", item.runtime_minutes)
                                item.average_rating = enrichment.get("average_rating", item.average_rating)
                        return item
                    except aiohttp.ClientResponseError:
                        LOGGER.info("Skipping unavailable URL (HTTP %d): %s", status, url)
                        return None
                    except Exception:
                        LOGGER.info("Skipping unavailable URL after fallback attempt: %s", url)
                        return None

                is_retriable = status in {408, 425, 429, 500, 502, 503, 504}
                if not is_retriable:
                    LOGGER.warning("Skipping URL due to non-retriable HTTP %d: %s", status, url)
                    return None

                backoff = 2 ** (attempt - 1)
                LOGGER.warning(
                    "Transient HTTP %d while scraping %s on attempt %d/%d. Retrying in %ds",
                    status,
                    url,
                    attempt,
                    retry_attempts,
                    backoff,
                )
                if attempt == retry_attempts:
                    LOGGER.error("Failed to scrape %s after %d attempts", url, retry_attempts)
                    return None
                await asyncio.sleep(backoff)
            except aiohttp.ClientError as exc:
                backoff = 2 ** (attempt - 1)
                LOGGER.warning("Error scraping %s on attempt %d: %s", url, attempt, exc)
                if attempt == retry_attempts:
                    LOGGER.error("Failed to scrape %s after %d attempts", url, retry_attempts)
                    return None
                await asyncio.sleep(backoff)
            except Exception as exc:
                LOGGER.exception("Unexpected scrape error for %s: %s", url, exc)
                return None


async def scrape_from_sitemaps(
    sitemap_urls: list[str],
    concurrency: int,
    user_agent: str,
    discover_latest: bool,
    discovery_regions: list[str],
    sitemap_hints: list[str],
    tmdb_api_key: str | None,
    tmdb_concurrency: int,
    max_urls: int | None,
) -> list[CatalogItem]:
    connector = aiohttp.TCPConnector(limit=concurrency)
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
        "Referer": "https://www.hotstar.com/in",
    }
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, headers=headers, timeout=timeout) as session:
        effective_sitemaps = sitemap_urls
        if discover_latest or not effective_sitemaps:
            effective_sitemaps = await discover_latest_sitemaps(
                session=session,
                regions=discovery_regions,
                hints=sitemap_hints,
            )
            LOGGER.info("Discovered %d latest sitemap files", len(effective_sitemaps))

        tmdb_enricher = TmdbEnricher(api_key=tmdb_api_key, concurrency=tmdb_concurrency) if tmdb_api_key else None

        all_content_urls: list[str] = []
        for sitemap_url in effective_sitemaps:
            LOGGER.info("Fetching sitemap: %s", sitemap_url)
            xml_text = await fetch_text(session, sitemap_url)
            doc = parse_sitemap_document(xml_text)
            if doc.kind == "urlset":
                urls = [canonicalize_content_url(urljoin(sitemap_url, loc)) for loc in doc.locs]
                LOGGER.info("Found %d URLs in sitemap %s", len(urls), sitemap_url)
                all_content_urls.extend(urls)
            else:
                LOGGER.debug("Skipping non-urlset sitemap file: %s", sitemap_url)

        unique_urls = list(dict.fromkeys(all_content_urls))
        if max_urls is not None and max_urls > 0:
            unique_urls = unique_urls[:max_urls]
            LOGGER.info("Applying max URL limit: %d", len(unique_urls))
        LOGGER.info("Total unique content URLs to scrape: %d", len(unique_urls))

        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            scrape_url(
                session,
                url,
                semaphore=semaphore,
                tmdb_enricher=tmdb_enricher,
            )
            for url in unique_urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    return [item for item in results if item is not None]


def save_items(items: list[CatalogItem], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [asdict(item) for item in items]
    output_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved %d catalog items to %s", len(items), output_path)


async def async_main(args: argparse.Namespace) -> None:
    regions = list(dict.fromkeys(args.region or DEFAULT_REGIONS))
    hints = list(dict.fromkeys(args.sitemap_hint or DEFAULT_SITEMAP_HINTS))
    items = await scrape_from_sitemaps(
        sitemap_urls=args.sitemap or [],
        concurrency=args.concurrency,
        user_agent=args.user_agent,
        discover_latest=args.discover_latest,
        discovery_regions=regions,
        sitemap_hints=hints,
        tmdb_api_key=args.tmdb_api_key,
        tmdb_concurrency=args.tmdb_concurrency,
        max_urls=args.max_urls,
    )
    save_items(items, Path(args.output))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Async Hotstar catalog scraper from sitemap XML")
    parser.add_argument(
        "--sitemap",
        action="append",
        default=None,
        help="Sitemap URL. Repeat for multiple values.",
    )
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "data" / "hotstar_scraped.json"))
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--discover-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-discover latest sitemap files from robots.txt and sitemap indexes.",
    )
    parser.add_argument(
        "--region",
        action="append",
        default=None,
        help="Region code for robots sitemap discovery (repeatable).",
    )
    parser.add_argument(
        "--sitemap-hint",
        action="append",
        default=None,
        help="Only include sitemap files whose URL contains these hints.",
    )
    parser.add_argument(
        "--tmdb-api-key",
        type=str,
        default=os.getenv("TMDB_API_KEY", "").strip() or None,
        help="Optional TMDB API key for enrichment (cast/characters/keywords).",
    )
    parser.add_argument("--tmdb-concurrency", type=int, default=5)
    parser.add_argument(
        "--max-urls",
        type=int,
        default=None,
        help="Optional cap on number of content URLs to scrape (for test runs).",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default="Mozilla/5.0 (compatible; HotstarSemanticSearchBot/1.0)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        LOGGER.warning("Scraper interrupted by user")


if __name__ == "__main__":
    main()
