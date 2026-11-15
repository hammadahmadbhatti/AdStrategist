"""Polite page fetching and the deterministic facts we read off a page."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .textutil import clean, dedupe, host_of, normalize_url, registrable_domain

BOT_NAME = "AdStrategist"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              f"Chrome/126.0 Safari/537.36 {BOT_NAME}/0.1")

# Script fingerprints that show which ad platforms a site is set up to advertise on.
TRACKERS = {
    "meta": [r"connect\.facebook\.net", r"\bfbq\s*\("],
    "google_ads": [r"\bAW-\d{6,}", r"googleadservices\.com", r"googleads\.g\.doubleclick\.net"],
    "google_analytics": [r"\bG-[A-Z0-9]{8,}\b", r"google-analytics\.com"],
    "google_tag_manager": [r"googletagmanager\.com/gtm\.js", r"\bGTM-[A-Z0-9]{5,}"],
    "tiktok": [r"analytics\.tiktok\.com", r"\bttq\.(?:load|track|page)"],
    "linkedin": [r"snap\.licdn\.com"],
    "pinterest": [r"\bpintrk\s*\(", r"s\.pinimg\.com/ct"],
    "microsoft_ads": [r"bat\.bing\.com"],
    "snapchat": [r"sc-static\.net/scevent"],
}
TRACKER_LABELS = {
    "meta": "Meta Pixel", "google_ads": "Google Ads tag", "google_analytics": "Google Analytics",
    "google_tag_manager": "Google Tag Manager", "tiktok": "TikTok Pixel", "linkedin": "LinkedIn Insight",
    "pinterest": "Pinterest tag", "microsoft_ads": "Microsoft Ads UET", "snapchat": "Snap Pixel",
}

# Offer mechanics a visitor can see. English and German phrasing.
OFFER_SIGNALS = {
    "free_trial": [r"free trial", r"try (?:it )?(?:for )?free", r"kostenlos testen", r"gratis testen", r"testphase"],
    "free_shipping": [r"free shipping", r"free delivery", r"kostenloser versand", r"versandkostenfrei", r"gratis versand"],
    "guarantee": [r"money[- ]back", r"guarantee", r"geld[- ]zurück", r"garantie"],
    "discount": [r"\d{1,2}\s?% (?:off|rabatt)", r"\bdiscount\b", r"\brabatt\b", r"\bsale\b", r"\bgutschein\b", r"\bcoupon\b"],
    "reviews": [r"\breviews?\b", r"testimonials?", r"trustpilot", r"bewertungen", r"kundenstimmen", r"erfahrungen"],
    "free_quote": [r"free quote", r"free estimate", r"kostenloses angebot", r"unverbindlich"],
    "online_booking": [r"book (?:now|online)", r"online buchen", r"termin (?:buchen|vereinbaren)", r"jetzt buchen"],
}
OFFER_LABELS = {
    "free_trial": "free trial", "free_shipping": "free shipping", "guarantee": "guarantee", "discount": "discounts",
    "reviews": "reviews or testimonials", "free_quote": "free quote", "online_booking": "online booking",
    "pricing_shown": "prices shown on the page",
}

_PRICE = re.compile(
    r"(?:€|\$|£|EUR|USD|CHF|RM)\s?\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})?|\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s?(?:€|EUR|USD|CHF)"
)
_SUBPAGE_HINTS = re.compile(
    r"pric|preis|kosten|tarif|feature|funktion|leistung|service|product|produkt|shop|about|ueber|über|angebot", re.I
)


@dataclass
class Page:
    url: str
    final_url: str = ""
    status: int = 0
    error: str = ""
    title: str = ""
    description: str = ""
    site_name: str = ""
    lang: str = ""
    h1: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    ctas: list[str] = field(default_factory=list)
    text: str = ""
    prices: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    trackers: list[str] = field(default_factory=list)
    offers: list[str] = field(default_factory=list)
    schema_types: list[str] = field(default_factory=list)
    locality: str = ""
    has_phone: bool = False
    app_links: list[str] = field(default_factory=list)
    theme_color: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.title or self.text)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["text"] = data["text"][:1500]
        data["links"] = data["links"][:20]
        return data


def detect_trackers(html: str) -> list[str]:
    return [name for name, patterns in TRACKERS.items() if any(re.search(p, html) for p in patterns)]


def detect_offers(text: str, prices: list[str]) -> list[str]:
    low = text.lower()
    found = [name for name, patterns in OFFER_SIGNALS.items() if any(re.search(p, low) for p in patterns)]
    if prices:
        found.append("pricing_shown")
    return found


def _walk_jsonld(node, types: list[str], out: dict) -> None:
    if isinstance(node, list):
        for item in node:
            _walk_jsonld(item, types, out)
    elif isinstance(node, dict):
        kind = node.get("@type")
        for t in kind if isinstance(kind, list) else [kind]:
            if isinstance(t, str):
                types.append(t)
        address = node.get("address")
        if isinstance(address, dict) and address.get("addressLocality") and not out.get("locality"):
            out["locality"] = str(address["addressLocality"])
        if node.get("name") and isinstance(kind, str) and kind in ("Organization", "LocalBusiness", "WebSite") \
                and not out.get("name"):
            out["name"] = str(node["name"])
        for value in node.values():
            if isinstance(value, (dict, list)):
                _walk_jsonld(value, types, out)


def parse_html(url: str, html: str) -> Page:
    """Everything here is read straight from the markup. No model, no guessing."""
    page = Page(url=url, final_url=url)
    soup = BeautifulSoup(html, "html.parser")
    page.trackers = detect_trackers(html)

    def meta(*names: str) -> str:
        for name in names:
            tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return clean(tag["content"])
        return ""

    page.title = clean(soup.title.get_text()) if soup.title else meta("og:title")
    page.description = meta("description", "og:description", "twitter:description")
    page.site_name = meta("og:site_name", "application-name")
    page.theme_color = meta("theme-color")
    html_tag = soup.find("html")
    page.lang = ((html_tag.get("lang") if html_tag else "") or meta("og:locale") or "")[:2].lower()

    types: list[str] = []
    extra: dict = {}
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            _walk_jsonld(json.loads(script.string or ""), types, extra)
        except (ValueError, TypeError):
            continue
    page.schema_types = dedupe(types)
    page.locality = extra.get("locality", "")
    if not page.site_name and extra.get("name"):
        page.site_name = clean(extra["name"])

    images = [meta("og:image", "twitter:image")]
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src or src.startswith("data:") or re.search(r"\.(svg|gif)(\?|$)|logo|icon|sprite|pixel|avatar", src, re.I):
            continue
        try:
            if int(img.get("width") or 400) < 300:
                continue
        except ValueError:
            pass
        images.append(src)
    page.images = dedupe([urljoin(url, i) for i in images if i])[:10]

    own = registrable_domain(url)
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        if href.startswith("tel:"):
            page.has_phone = True
        if re.search(r"apps\.apple\.com|play\.google\.com/store", href):
            page.app_links.append(href)
        if not href.startswith("http"):
            continue
        label = clean(a.get_text(" "))
        if registrable_domain(href) == own and label and len(label) < 60:
            page.links.append({"url": href.split("#")[0], "text": label})
    page.links = dedupe(page.links, key=lambda l: l["url"].rstrip("/"))
    page.app_links = dedupe(page.app_links)

    ctas = [clean(b.get_text(" ")) for b in soup.find_all("button")]
    ctas += [clean(a.get_text(" ")) for a in soup.find_all("a", class_=re.compile(r"btn|button|cta", re.I))]
    page.ctas = dedupe([c for c in ctas if 2 < len(c) < 40])[:8]

    for tag in soup(["script", "style", "noscript", "svg", "template", "iframe"]):
        tag.decompose()
    page.h1 = dedupe([clean(h.get_text(" ")) for h in soup.find_all("h1") if clean(h.get_text(" "))])[:3]
    page.headings = dedupe([clean(h.get_text(" ")) for h in soup.find_all(["h2", "h3"])
                            if 3 < len(clean(h.get_text(" "))) < 120])[:30]
    bullets = []
    for li in soup.find_all("li"):
        text = clean(li.get_text(" "))
        if not 3 <= len(text.split()) <= 16 or li.find_parent(["nav", "header", "footer"]):
            continue
        link = li.find("a")
        if link and clean(link.get_text(" ")) == text:
            continue  # a menu or link-list entry
        bullets.append(text)
    page.bullets = dedupe(bullets)[:30]
    page.text = clean(soup.get_text(" "))[:20000]
    page.prices = dedupe([clean(m.group(0)) for m in _PRICE.finditer(page.text)])[:8]
    page.offers = detect_offers(page.text, page.prices)
    return page


class Fetcher:
    """One HTTP client for a run: honours robots.txt, caches pages, never raises."""

    def __init__(self, timeout: float = 15.0, respect_robots: bool = True, transport=None) -> None:
        self.client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en,de;q=0.8"},
            timeout=timeout, follow_redirects=True, transport=transport,
        )
        self.respect_robots = respect_robots
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self._cache: dict[str, Page] = {}

    def close(self) -> None:
        self.client.close()

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            parser: robotparser.RobotFileParser | None = robotparser.RobotFileParser()
            try:
                resp = self.client.get(origin + "/robots.txt", timeout=6.0)
                if resp.status_code == 200 and "<html" not in resp.text[:300].lower():
                    parser.parse(resp.text.splitlines())
                else:
                    parser = None
            except httpx.HTTPError:
                parser = None
            self._robots[origin] = parser
        parser = self._robots[origin]
        return True if parser is None else parser.can_fetch(BOT_NAME, url)

    def fetch(self, url: str) -> Page:
        url = normalize_url(url)
        if url in self._cache:
            return self._cache[url]
        page = Page(url=url)
        if urlparse(url).scheme not in ("http", "https"):
            page.error = "only http and https links are supported"
        elif not self.allowed(url):
            page.error = "blocked by the site's robots.txt"
        else:
            try:
                resp = self.client.get(url)
                if resp.status_code >= 400:
                    page.status, page.error = resp.status_code, f"the site answered HTTP {resp.status_code}"
                elif "html" not in resp.headers.get("content-type", "html"):
                    page.status, page.error = resp.status_code, "the link is not a web page"
                else:
                    page = parse_html(str(resp.url), resp.text)
                    page.url, page.status = url, resp.status_code
            except httpx.HTTPError as exc:
                page.error = f"could not load the page ({type(exc).__name__})"
        self._cache[url] = page
        return page

    def fetch_subpages(self, page: Page, limit: int = 2) -> list[Page]:
        """A couple of pricing/feature/about pages give the brief far more to work with."""
        home = page.final_url.rstrip("/")
        picks = [l["url"] for l in page.links
                 if _SUBPAGE_HINTS.search(l["url"] + " " + l["text"]) and l["url"].rstrip("/") != home]
        out = []
        for link in dedupe(picks)[:limit]:
            sub = self.fetch(link)
            if sub.ok:
                out.append(sub)
        return out

    def download(self, url: str, max_bytes: int = 8_000_000) -> bytes | None:
        try:
            resp = self.client.get(url, timeout=20.0)
            if resp.status_code == 200 and len(resp.content) <= max_bytes:
                return resp.content
        except httpx.HTTPError:
            pass
        return None


def domain_label(url: str) -> str:
    return registrable_domain(url).split(".")[0]


__all__ = ["Fetcher", "Page", "parse_html", "detect_trackers", "detect_offers", "TRACKER_LABELS", "OFFER_LABELS",
           "domain_label", "host_of"]
