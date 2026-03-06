from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import urlopen

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "screenshots"
BASE_URL = "http://127.0.0.1:8010"

QUERIES = [
    "al pacino",
    "romantic comedy",
    "horror supernatural",
]


def fetch_results(query: str, k: int = 5) -> dict[str, Any]:
    url = f"{BASE_URL}/search?q={quote_plus(query)}&k={k}"
    with urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def line_wrap(text: str, width: int) -> list[str]:
    if not text:
        return []
    return textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False)


def render_shot(query: str, payload: dict[str, Any], out_path: Path) -> None:
    width = 1600
    height = 1040
    margin = 36

    image = Image.new("RGB", (width, height), color=(246, 248, 252))
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("arial.ttf", 42)
        subtitle_font = ImageFont.truetype("arial.ttf", 27)
        body_font = ImageFont.truetype("arial.ttf", 22)
        small_font = ImageFont.truetype("arial.ttf", 19)
    except OSError:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((margin, margin), "Search-Hotstar - Search Shot", fill=(31, 41, 55), font=title_font)
    draw.text(
        (margin, margin + 58),
        f"Query: {query}   |   Returned: {len(payload.get('results', []))}   |   Search ms: {payload.get('search_ms')}",
        fill=(55, 65, 81),
        font=subtitle_font,
    )

    y = margin + 118
    card_h = 268
    card_gap = 22

    for idx, item in enumerate((payload.get("results") or [])[:3], start=1):
        card_top = y + (idx - 1) * (card_h + card_gap)
        card_left = margin
        card_right = width - margin
        card_bottom = card_top + card_h

        draw.rounded_rectangle(
            [(card_left, card_top), (card_right, card_bottom)],
            radius=18,
            fill=(255, 255, 255),
            outline=(203, 213, 225),
            width=2,
        )

        title = str(item.get("title") or "Untitled")
        score = float(item.get("score") or 0.0)
        year = item.get("release_year") or "NA"
        lang = item.get("original_language") or "NA"
        genre = str(item.get("genre") or "NA")
        description = str(item.get("description") or "")

        draw.text((card_left + 24, card_top + 18), f"{idx}. {title}", fill=(17, 24, 39), font=subtitle_font)
        draw.text(
            (card_left + 24, card_top + 64),
            f"Score: {score:.3f}   Year: {year}   Language: {lang}   Genre: {genre}",
            fill=(75, 85, 99),
            font=body_font,
        )

        wrapped = line_wrap(description, width=102)
        desc_lines = wrapped[:6] if wrapped else ["No description available."]
        draw.text(
            (card_left + 24, card_top + 102),
            "\n".join(desc_lines),
            fill=(55, 65, 81),
            font=small_font,
            spacing=6,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> None:
    slug_map = {
        "al pacino": "search_al_pacino.png",
        "romantic comedy": "search_romantic_comedy.png",
        "horror supernatural": "search_horror_supernatural.png",
    }

    for query in QUERIES:
        payload = fetch_results(query)
        output_name = slug_map[query]
        render_shot(query=query, payload=payload, out_path=OUTPUT_DIR / output_name)
        print(f"Saved {OUTPUT_DIR / output_name}")


if __name__ == "__main__":
    main()
