"""Free web search (DuckDuckGo HTML endpoint) used to discover competitors."""
from __future__ import annotations

import time
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup

from ..textutil import clean

ENDPOINT = "https://html.duckduckgo.com/html/"


def region_code(country: str, lang: str) -> str:
    country = {"GB": "uk"}.get(country.upper(), country.lower())
    return f"{country}-{(lang or 'en').lower()}"


def _real_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return target or href
    return href


def parse_results(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for block in soup.select("div.result"):
        if "result--ad" in (block.get("class") or []):
            continue
        link = block.select_one("a.result__a")
        if not link or not link.get("href"):
            continue
        url = _real_url(link["href"])
        if not url.startswith("http") or "duckduckgo.com/y.js" in url:
            continue
        snippet = block.select_one(".result__snippet")
        out.append({"url": url, "title": clean(link.get_text(" ")),
                    "snippet": clean(snippet.get_text(" ")) if snippet else ""})
    return out


def search(client: httpx.Client, query: str, country: str = "US", lang: str = "en", pause: float = 1.5) -> list[dict]:
    """Returns [] on any failure so a blocked search never stops a run.

    The endpoint throttles bursts, so every call is followed by a pause and a throttled
    call is retried once after a longer wait.
    """
    results: list[dict] = []
    for attempt in range(2):
        try:
            resp = client.post(ENDPOINT, data={"q": query, "kl": region_code(country, lang)}, timeout=15.0)
            if resp.status_code == 200:
                results = parse_results(resp.text)
        except httpx.HTTPError:
            pass
        if results or pause == 0:
            break
        if attempt == 0:
            time.sleep(pause * 5)
    time.sleep(pause)
    for rank, row in enumerate(results, 1):
        row["rank"] = rank
        row["query"] = query
    return results
