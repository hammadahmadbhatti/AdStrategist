"""Step 2: find competitors through web search, then read their homepages."""
from __future__ import annotations

import re
from urllib.parse import urlparse

from ..context import Context
from ..sources.search import search
from ..textutil import dedupe, normalize_url, registrable_domain, tokens, truncate, word_caps
from ..web_fetch import Page
from .brief import _selling_points, guess_brand

# Platforms, directories and publishers that rank for everything but sell nothing comparable.
NOT_COMPETITORS = {
    "facebook", "instagram", "youtube", "wikipedia", "amazon", "ebay", "yelp", "reddit", "linkedin", "tiktok",
    "pinterest", "x", "twitter", "trustpilot", "g2", "capterra", "quora", "medium", "forbes", "techradar",
    "pcmag", "google", "apple", "gelbeseiten", "11880", "check24", "idealo", "tripadvisor", "github",
    "crunchbase", "glassdoor", "indeed", "getapp", "softwareadvice", "producthunt", "nytimes", "cnet",
    "businessinsider", "wired", "theverge", "zdnet", "statista", "kununu", "xing", "kleinanzeigen", "etsy",
    "bing", "duckduckgo", "yahoo", "golocal", "meinestadt", "dasoertliche", "provenexpert", "tomsguide",
}
LISTICLE = re.compile(r"\b(best|top \d+|\d+ best|\d+ beste|review|reviews|vs\.?|alternatives?|alternativen|beste[nr]?|"
                      r"vergleich|test \d{4}|im test|ranking|guide)\b", re.I)

FILTER_SCHEMA = {
    "type": "object",
    "properties": {"competitors": {"type": "array", "items": {"type": "integer"}}},
    "required": ["competitors"],
    "additionalProperties": False,
}
PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "positioning": {"type": "string", "description": "one sentence: how they present themselves"},
        "audience": {"type": "string"},
        "claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["positioning", "audience", "claims"],
    "additionalProperties": False,
}


def build_queries(brief: dict) -> list[str]:
    german = brief.get("language") == "de"
    seeds = brief.get("seed_keywords", [])[:2]
    category = brief.get("category", "")
    location = brief.get("location", "")
    queries = [f"{s} {location}".strip() if brief.get("business_type") == "local_service" else s for s in seeds]
    if category:
        queries.append(f"{'beste' if german else 'best'} {category} {location}".strip())
    if brief.get("brand"):
        queries.append(f"{brief['brand']} {'Alternative' if german else 'alternatives'}")
    return dedupe([q for q in queries if q])[:4]


MEDIA = re.compile(r"\b(news|magazine|blog|media platform|newsletter|podcast|magazin|nachrichten|ratgeber|"
                   r"zeitung|journal)\b", re.I)
VS_NOISE = {"comparison", "reddit", "pricing", "price", "review", "reviews", "which", "is", "better", "features",
            "vergleich", "test", "erfahrungen", "unterschied", "difference", "or", "oder"}


def versus_names(ctx: Context, brand: str) -> list[str]:
    """Names people compare the brand with, straight from autocomplete ('basecamp vs asana' -> 'asana')."""
    from ..sources.autocomplete import suggest
    key = brand.lower().strip()
    if len(key) < 3:
        return []
    names = []
    for phrase in suggest(ctx.fetcher.client, f"{key} vs ", ctx.config.country, ctx.lang):
        match = re.match(rf"^{re.escape(key)}\s+vs\.?\s+(.+)$", phrase)
        if not match:
            continue
        words = [w for w in match.group(1).split() if w not in VS_NOISE and not w.isdigit()]
        name = " ".join(words[:3])
        if 2 < len(name) <= 30 and key not in name:
            names.append(name)
    return dedupe(names)[:6]


def resolve_site(ctx: Context, name: str, own_domain: str) -> str:
    """The official site for a competitor name.

    First try the obvious address (name.com, name.<country>) and accept it only if the page names the
    brand. That costs no search. Otherwise take the first search result whose domain matches the name.
    """
    key = re.sub(r"[^a-z0-9]", "", name.lower().split()[0])
    joined = re.sub(r"[^a-z0-9]", "", name.lower())
    for tld in dict.fromkeys(["com", ctx.config.country.lower()]):
        page = ctx.fetcher.fetch(f"https://{joined}.{tld}/")
        shown = re.sub(r"[^a-z0-9]", "", f"{page.title} {page.site_name}".lower())
        if page.ok and joined in shown and registrable_domain(page.final_url) != own_domain:
            return page.final_url
    for row in search(ctx.fetcher.client, name, ctx.config.country, ctx.lang)[:6]:
        domain = registrable_domain(row["url"])
        label = re.sub(r"[^a-z0-9]", "", domain.split(".")[0])
        if domain != own_domain and label not in NOT_COMPETITORS and key and (key in label or label in key):
            parsed = urlparse(row["url"])
            return f"{parsed.scheme}://{parsed.netloc}/"
    return ""


def rank_candidates(results: list[dict], own_domain: str, known: set[str]) -> list[dict]:
    candidates: dict[str, dict] = {}
    for row in results:
        domain = registrable_domain(row["url"])
        if not domain or domain == own_domain or domain in known or domain.split(".")[0] in NOT_COMPETITORS:
            continue
        weight = 1.0 / row.get("rank", 10)
        if LISTICLE.search(row.get("title", "")):
            weight *= 0.3
        if len([p for p in urlparse(row["url"]).path.split("/") if p]) > 1:
            weight *= 0.7
        entry = candidates.setdefault(domain, {"domain": domain, "url": row["url"], "title": row.get("title", ""),
                                               "snippet": row.get("snippet", ""), "score": 0.0, "queries": []})
        entry["score"] += weight
        entry["queries"].append(row.get("query", ""))
    return sorted(candidates.values(), key=lambda c: -c["score"])


def profile(page: Page, domain: str, found_via: list[str], source: str) -> dict:
    usps = _selling_points(page)[:5]
    return {
        "name": guess_brand(page) if page.ok else domain.split(".")[0].title(),
        "domain": domain, "url": page.final_url or page.url, "source": source,
        "found_via": dedupe([q for q in found_via if q]),
        "title": page.title, "description": page.description, "h1": page.h1[:2], "usps": usps,
        "prices": page.prices[:5], "offers": page.offers, "trackers": page.trackers, "ctas": page.ctas[:4],
        "positioning": truncate(page.description or (page.h1[0] if page.h1 else page.title), 180),
        "audience": "", "claims": [], "error": page.error,
    }


def market_vocabulary(competitors: list[dict], own_text: str = "", limit: int = 5) -> list[str]:
    """Two and three word phrases that at least two competitors use in their title, description or H1.

    Phrases our own page also uses rank first: they describe the market we are actually in."""
    from collections import Counter

    from ..textutil import _grams
    used: Counter[str] = Counter()
    brand_words = {w for c in competitors for w in c["name"].lower().split()}
    for comp in competitors:
        text = " . ".join([comp["title"], comp["description"], *comp["h1"]])
        for gram in set(_grams(text)):
            if not (set(gram.split()) & brand_words):
                used[gram] += 1
    own = " ".join(tokens(own_text))
    ranked = sorted((g for g, n in used.items() if n >= 2),
                    key=lambda g: (g not in own, -used[g], -len(g.split()), g))
    chosen: list[str] = []
    for gram in ranked:
        if not any(gram in other or other in gram for other in chosen):
            chosen.append(gram)
    return chosen[:limit]


def run(ctx: Context) -> dict:
    brief = ctx.brief
    own = brief["domain"]
    limit = max(1, ctx.config.max_competitors)
    notes: list[str] = []

    manual = dedupe([normalize_url(u) for u in ctx.config.competitors if u.strip()])
    known = {registrable_domain(u) for u in manual}
    picked = [{"domain": registrable_domain(u), "url": u, "queries": [], "source": "you"} for u in manual]

    if len(picked) < limit:
        for name in versus_names(ctx, brief["brand"]):
            if len(picked) >= limit:
                break
            ctx.emit("info", f"People compare {brief['brand']} with {name}: looking up the site")
            site = resolve_site(ctx, name, own)
            domain = registrable_domain(site) if site else ""
            if domain and domain not in known:
                known.add(domain)
                picked.append({"domain": domain, "url": site, "queries": [f"{brief['brand'].lower()} vs {name}"],
                               "source": "autocomplete"})

    queries = build_queries(brief)
    if len(picked) < limit and queries:
        results: list[dict] = []
        for query in queries:
            ctx.emit("info", f"Searching: {query}")
            results += search(ctx.fetcher.client, query, ctx.config.country, ctx.lang)
        candidates = rank_candidates(results, own, known)
        if not results:
            notes.append("Web search returned nothing (it may be rate limited). Add competitor URLs by hand.")
        shortlist = candidates[:12]
        if shortlist and ctx.llm.enabled("research"):
            listing = "\n".join(f"{i}. {c['domain']}: {c['title']}. {c['snippet'][:140]}" for i, c in enumerate(shortlist))
            answer = ctx.llm.ask_json(
                "competitor filter",
                f"Our product: {brief['brand']}, {brief['category']}. {brief['one_liner']}\n\n"
                "Which of these search results are direct competitors: companies selling a similar product or "
                "service to the same customers? Leave out news sites, blogs, directories, marketplaces and review "
                f"sites. Answer with their numbers.\n\n{listing}",
                FILTER_SCHEMA, max_tokens=120)
            chosen = [shortlist[i] for i in dict.fromkeys(answer["competitors"]) if 0 <= i < len(shortlist)] if answer else []
            if len(chosen) >= 2:
                shortlist = chosen
        for cand in shortlist[: limit - len(picked) + 4]:  # a few spares in case some turn out to be media sites
            parsed = urlparse(cand["url"])
            picked.append({"domain": cand["domain"], "url": f"{parsed.scheme}://{parsed.netloc}/",
                           "queries": cand["queries"], "source": "search"})

    competitors = []
    for item in picked:
        if len(competitors) >= limit:
            break
        ctx.emit("info", f"Reading competitor: {item['domain']}")
        page = ctx.fetcher.fetch(item["url"])
        if item["source"] == "search" and page.ok and (
                MEDIA.search(f"{page.title} {page.description}") or
                {"NewsMediaOrganization", "NewsArticle", "Blog", "BlogPosting"} & set(page.schema_types)):
            ctx.emit("info", f"Skipping {item['domain']}: it is a publication, not a competitor")
            continue
        ctx.competitor_pages[item["domain"]] = page
        ctx.index_page(page, item["domain"])
        entry = profile(page, item["domain"], item["queries"], item["source"])
        if page.ok and ctx.llm.enabled("research"):
            found = ctx.passages("what they sell, who it is for and why customers choose them", k=4,
                                 site=item["domain"])
            if found:
                from ..rag import join_passages
                body = "Most relevant passages:\n" + join_passages(found, limit=1600)
            else:
                body = f"Text: {page.text[:1500]}"
            answer = ctx.llm.ask_json(
                f"competitor profile {item['domain']}",
                f"Describe this competitor in {ctx.language_name} from its homepage. Use only what it says. "
                "claims: up to 3 short selling points.\n\n"
                f"Title: {page.title}\nDescription: {page.description}\n"
                f"Headings: {' | '.join(page.h1 + page.headings[:10])}\n{body}",
                PROFILE_SCHEMA, max_tokens=260, repeat={"claims": 3})
            if answer:
                entry["positioning"] = truncate(answer["positioning"], 220) or entry["positioning"]
                entry["audience"] = truncate(answer["audience"], 120)
                entry["claims"] = [truncate(c, 100) for c in answer["claims"]][:3]
        competitors.append(entry)

    readable = [c for c in competitors if not c["error"]]
    # What the market calls this kind of product: phrases several competitors put in their own titles.
    # Real data, and a far better category than a slogan from our own page.
    own_text = " ".join([ctx.own_page.text if ctx.own_page else ""] + [p.text for p in ctx.sub_pages])
    market_terms = market_vocabulary(readable, own_text)
    if market_terms and not ctx.config.seed_keywords:
        if brief.get("category_from") == "page phrase":
            brief["category"], brief["category_from"] = word_caps(market_terms[0]), "competitor titles"
        if not brief.get("written_by_model"):
            brief["seed_keywords"] = dedupe(market_terms[:3] + brief["seed_keywords"])[:8]
        elif len(brief["seed_keywords"]) < 5:  # a model gave only a few seeds: add what the market calls it
            brief["seed_keywords"] = dedupe(brief["seed_keywords"] + market_terms[:3])[:6]
    if not competitors:
        notes.append("No competitors found. Pass them with --competitor or in the web form.")
    return {"queries": queries, "competitors": competitors, "readable": len(readable),
            "market_terms": market_terms, "notes": notes}
