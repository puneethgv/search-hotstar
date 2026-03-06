from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus
from urllib.request import urlopen


@dataclass(slots=True)
class CheckResult:
    query: str
    ok: bool
    http_ms: float
    search_ms: float | None
    result_count: int
    relevance_score: int
    reason: str


DEFAULT_TESTS: list[dict[str, str]] = [
    {"query": "crime thriller", "expects": "crime thriller mystery"},
    {"query": "romantic comedy", "expects": "romance comedy love"},
    {"query": "action hero", "expects": "action fight war adventure"},
    {"query": "family animation", "expects": "family animation kids"},
    {"query": "horror supernatural", "expects": "horror ghost supernatural"},
]


def fetch_search(base_url: str, query: str, top_k: int, timeout_s: int) -> tuple[dict[str, Any], float]:
    url = f"{base_url.rstrip('/')}/search?q={quote_plus(query)}&k={top_k}"
    start = time.perf_counter()
    with urlopen(url, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return payload, elapsed_ms


def score_relevance(results: list[dict[str, Any]], expected_terms: list[str]) -> int:
    if not results:
        return 0
    score = 0
    top = results[0]
    fields = " ".join(
        [
            str(top.get("title") or ""),
            str(top.get("description") or ""),
            str(top.get("genre") or ""),
            " ".join(top.get("keywords") or []),
        ]
    ).lower()
    for term in expected_terms:
        if term in fields:
            score += 1
    return score


def run_checks(base_url: str, tests: list[dict[str, str]], top_k: int, timeout_s: int) -> list[CheckResult]:
    output: list[CheckResult] = []
    for test in tests:
        query = test["query"]
        expected_terms = [part.strip().lower() for part in test["expects"].split() if part.strip()]
        try:
            payload, http_ms = fetch_search(base_url, query, top_k=top_k, timeout_s=timeout_s)
            results = payload.get("results") if isinstance(payload, dict) else None
            search_ms = payload.get("search_ms") if isinstance(payload, dict) else None

            if not isinstance(results, list):
                output.append(
                    CheckResult(
                        query=query,
                        ok=False,
                        http_ms=http_ms,
                        search_ms=None,
                        result_count=0,
                        relevance_score=0,
                        reason="Invalid response schema",
                    )
                )
                continue

            relevance = score_relevance(results, expected_terms)
            ok = len(results) > 0 and relevance >= 1
            reason = "OK" if ok else "Low relevance or empty results"
            output.append(
                CheckResult(
                    query=query,
                    ok=ok,
                    http_ms=round(http_ms, 2),
                    search_ms=float(search_ms) if isinstance(search_ms, (int, float)) else None,
                    result_count=len(results),
                    relevance_score=relevance,
                    reason=reason,
                )
            )
        except Exception as exc:
            output.append(
                CheckResult(
                    query=query,
                    ok=False,
                    http_ms=0.0,
                    search_ms=None,
                    result_count=0,
                    relevance_score=0,
                    reason=f"Request failed: {exc}",
                )
            )
    return output


def print_summary(results: list[CheckResult]) -> int:
    passed = sum(1 for item in results if item.ok)
    total = len(results)
    print("\n=== Search Quality Smoke Test ===")
    for item in results:
        status = "PASS" if item.ok else "FAIL"
        print(
            f"[{status}] query='{item.query}' results={item.result_count} "
            f"relevance={item.relevance_score} http_ms={item.http_ms} "
            f"search_ms={item.search_ms} reason={item.reason}"
        )

    print(f"\nPassed {passed}/{total} checks")
    return 0 if passed == total else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick quality check for /search endpoint")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results = run_checks(
        base_url=args.base_url,
        tests=DEFAULT_TESTS,
        top_k=args.top_k,
        timeout_s=args.timeout,
    )
    exit_code = print_summary(results)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
