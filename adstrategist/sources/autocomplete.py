"""Google Autocomplete: real phrases people type. Gives ideas and relative interest, not volumes."""
from __future__ import annotations

import time

import httpx

ENDPOINT = "https://suggestqueries.google.com/complete/search"


def suggest(client: httpx.Client, query: str, country: str = "US", lang: str = "en", pause: float = 0.15) -> list[str]:
    try:
        resp = client.get(ENDPOINT, params={"client": "firefox", "q": query, "hl": lang, "gl": country.lower()},
                          timeout=8.0)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return []
    time.sleep(pause)
    if not isinstance(data, list) or len(data) < 2 or not isinstance(data[1], list):
        return []
    return [str(s).strip().lower() for s in data[1] if str(s).strip()]
