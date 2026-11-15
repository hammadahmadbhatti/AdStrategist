"""Shared fixtures: a fake web (own site, two competitors, search, autocomplete) served without any network."""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from adstrategist.config import Settings
from adstrategist.web_fetch import Fetcher

OWN_HTML = """<!doctype html><html lang="en"><head>
<title>GreenLeaf | Organic Tea Subscription</title>
<meta name="description" content="GreenLeaf delivers organic loose leaf tea to your door every month. Pick your blend, pause any time.">
<meta property="og:site_name" content="GreenLeaf">
<meta property="og:image" content="/img/hero.jpg">
<meta name="theme-color" content="#1d6b4f">
<script type="application/ld+json">{"@type":"Organization","name":"GreenLeaf","address":{"@type":"PostalAddress","addressLocality":"Berlin"}}</script>
<script>fbq('init','123');</script>
</head><body>
<nav><ul><li><a href="/pricing">Pricing and plans page</a></li></ul></nav>
<h1>Organic tea subscription, delivered monthly</h1>
<h2>Why people choose our tea subscription</h2>
<ul><li>Hand picked organic loose leaf tea from small farms</li><li>Free shipping on every tea subscription box</li>
<li>Pause or cancel your subscription any time</li></ul>
<h2>How does it work?</h2>
<p>Choose an organic tea blend. We ship your tea subscription box each month. Organic tea, loose leaf tea, green tea
and black tea blends. Prices start at €12.90 per box. Read reviews from our customers. Add to cart today.</p>
<a href="/pricing">Pricing</a> <a href="/about">About our tea</a> <a href="/impressum">Impressum</a>
<button>Start your box</button>
</body></html>"""

COMPETITOR_HTML = {
    "teabox.example": """<html lang="en"><head><title>TeaBox | Organic Tea Subscription Boxes</title>
<meta name="description" content="Monthly organic tea subscription boxes with a money-back guarantee and gift boxes.">
<script src="https://www.googletagmanager.com/gtm.js?id=GTM-ABC1234"></script><script>gtag('config','AW-123456789');</script>
</head><body><h1>Organic tea subscription boxes and gift boxes</h1>
<ul><li>Tea gift boxes for every occasion and budget</li></ul><p>Try it free trial for 14 days. From $15.00.</p></body></html>""",
    "leafclub.example": """<html lang="en"><head><title>LeafClub: Tea Gift Boxes and Organic Tea Subscription</title>
<meta name="description" content="LeafClub ships tea gift boxes and an organic tea subscription with a guarantee.">
<script>fbq('track','PageView'); ttq.load('X');</script></head>
<body><h1>Tea gift boxes people love</h1><p>Money-back guarantee on all gift boxes.</p></body></html>""",
}

SEARCH_HTML = """<html><body>
<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fteabox.example%2F">TeaBox</a>
<a class="result__snippet">Organic tea subscription boxes.</a></div>
<div class="result"><a class="result__a" href="https://leafclub.example/">LeafClub tea</a>
<a class="result__snippet">Tea gift boxes.</a></div>
<div class="result"><a class="result__a" href="https://www.wikipedia.org/wiki/Tea">Tea</a></div>
<div class="result result--ad"><a class="result__a" href="https://ads.example/">Ad</a></div>
</body></html>"""

SUGGESTIONS = {
    "organic tea subscription": ["organic tea subscription", "organic tea subscription box",
                                 "organic tea subscription free", "organic tea subscription jobs"],
    "best organic tea subscription": ["best organic tea subscription box"],
    "organic tea subscription price": ["organic tea subscription price"],
}


def handler(request: httpx.Request) -> httpx.Response:
    host, path = request.url.host, request.url.path
    if path == "/robots.txt":
        return httpx.Response(404)
    if host == "html.duckduckgo.com":
        return httpx.Response(200, text=SEARCH_HTML)
    if host == "suggestqueries.google.com":
        query = parse_qs(urlparse(str(request.url)).query).get("q", [""])[0].strip()
        return httpx.Response(200, json=[query, SUGGESTIONS.get(query, [])])
    if host == "greenleaf.example":
        if path == "/img/hero.jpg":
            return httpx.Response(404)
        return httpx.Response(200, text=OWN_HTML, headers={"content-type": "text/html"})
    if host in COMPETITOR_HTML:
        return httpx.Response(200, text=COMPETITOR_HTML[host], headers={"content-type": "text/html"})
    return httpx.Response(404)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)


@pytest.fixture
def fetcher():
    f = Fetcher(transport=httpx.MockTransport(handler))
    yield f
    f.close()


@pytest.fixture
def settings(tmp_path):
    return Settings(research_provider="offline", writing_provider="offline", output_dir=tmp_path / "runs",
                    rag="off")


class FakeProvider:
    """Stands in for a model: returns canned JSON per task keyword and records every prompt."""

    id, label, is_local = "fake", "Fake model", True

    def __init__(self, answers: dict | None = None, model: str = "fake-1"):
        self.model = model
        self.answers = answers or {}
        self.prompts: list[str] = []

    def status(self):
        from adstrategist.llm import ProviderStatus
        return ProviderStatus(True, "fake")

    def complete(self, prompt, *, system="", max_tokens=1024, json_schema=None):
        from adstrategist.llm import LLMError, LLMResult
        self.prompts.append(prompt)
        for needle, answer in self.answers.items():
            if needle in prompt:
                if isinstance(answer, Exception):
                    raise answer
                return LLMResult(answer if isinstance(answer, str) else json.dumps(answer), 10, 20)
        raise LLMError("no canned answer")
