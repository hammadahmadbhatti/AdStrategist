"""Step 4: keyword research from real autocomplete data, with optional Keyword Planner volumes."""
from __future__ import annotations

import csv
import re

from ..context import Context
from ..sources.autocomplete import suggest
from ..sources.keyword_csv import parse_keyword_csv
from ..textutil import content_tokens, dedupe

TEMPLATES = {
    "en": ["{s}", "best {s}", "{s} price", "{s} vs", "{s} for"],
    "de": ["{s}", "beste {s}", "{s} preis", "{s} kosten", "{s} vergleich", "{s} für"],
}
LOCAL_TEMPLATES = {"en": "{s} near me", "de": "{s} in der nähe"}

INTENT_PATTERNS = [
    ("transactional", r"\b(buy|order|price|prices|pricing|cost|costs|cheap|discount|coupon|deal|book|booking|hire|"
                      r"near me|quote|rent|kaufen|bestellen|preis|preise|kosten|günstig|guenstig|buchen|angebot|"
                      r"mieten|in der nähe|rabatt|gutschein)\b"),
    ("commercial", r"\b(best|top|review|reviews|vs|versus|compare|comparison|alternative|alternatives|beste|bester|"
                   r"besten|test|vergleich|erfahrungen|bewertungen|alternativen)\b"),
    ("informational", r"\b(how|what|why|when|guide|tutorial|tips|ideas|examples|meaning|definition|wie|was|warum|"
                      r"wann|anleitung|tipps|ideen|beispiele|bedeutung)\b"),
]
NEGATIVE = re.compile(
    r"\b(jobs?|salary|career|careers|course|training|diy|definition|meaning|wiki|wikipedia|pdf|reddit|login|"
    r"gehalt|ausbildung|stellenangebote|karriere|selber machen|bedeutung)\b")
FREE = re.compile(r"\b(free|kostenlos|gratis|umsonst)\b")


def classify_intent(keyword: str, brand_names: set[str], location: str = "") -> str:
    low = keyword.lower()
    if any(name and name in low for name in brand_names):
        return "navigational"
    for intent, pattern in INTENT_PATTERNS:
        if re.search(pattern, low):
            return intent
    if location and location.lower() in low:
        return "transactional"
    return "commercial" if len(low.split()) <= 3 else "informational"


# Words a searcher adds to any product query. They never count as "off topic".
MODIFIERS = set(
    "best top cheap cheapest price prices pricing cost costs free online near review reviews alternative "
    "alternatives compare comparison tool tools software app apps service services company companies platform "
    "small business beste bester besten günstig günstige preis preise kosten kostenlos nähe test vergleich "
    "erfahrungen alternativen anbieter firma dienst dienstleister".split()
)


_MODIFIER_STEMS = {m[:5] for m in MODIFIERS}


def page_vocabulary(ctx: Context) -> set[str]:
    """Word stems used on the user's own pages. Empty when the page is too thin to judge by."""
    text = " ".join([ctx.own_page.text if ctx.own_page else ""] + [p.text for p in ctx.sub_pages])
    if len(text) < 400:
        return set()
    return {t[:5] for t in content_tokens(text)} | {m[:5] for m in MODIFIERS}


def on_topic(phrase: str, seed_stems: set[str], vocabulary: set[str], location: str) -> bool:
    """Autocomplete drifts (a seed like 'benefits' returns 'benefits of chia seeds'). Keep a suggestion
    only if its words are ones the seed or the page itself uses; long phrases may carry one new word."""
    stems = [t[:5] for t in content_tokens(phrase)]
    if seed_stems and not (set(stems) & seed_stems):
        return False
    if not vocabulary:
        return True
    place = location.lower()[:5]
    topical = [s for s in stems if s not in _MODIFIER_STEMS]  # "best", "price" say nothing about the topic
    unknown = [s for s in topical if s not in vocabulary and s not in seed_stems and s != place]
    return len(unknown) <= (1 if len(topical) >= 3 else 0)


RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {"not_relevant": {"type": "array", "items": {"type": "integer"}}},
    "required": ["not_relevant"], "additionalProperties": False,
}


def model_filter(ctx: Context, keywords: list[dict], seeds: list[str]) -> list[str]:
    """Let the research model flag searches a buyer of this product would not make. Seeds are never removed,
    and an answer that throws out most of the list is ignored as unreliable."""
    brief = ctx.brief
    candidates = [k for k in keywords if k["keyword"] not in seeds][:60]
    if len(candidates) < 5 or not ctx.llm.enabled("research"):
        return []
    listing = "\n".join(f"{i}. {k['keyword']}" for i, k in enumerate(candidates))
    answer = ctx.llm.ask_json(
        "keyword relevance",
        f"Product: {brief.get('brand')}, {brief.get('category')}. {brief.get('one_liner')}\n\n"
        "Below are Google searches. List the numbers of the searches made by people who are NOT looking for "
        "this kind of product, for example people looking for a job, a course, a certification, a definition, "
        f"a book, or a different kind of product.\n\n{listing}",
        RELEVANCE_SCHEMA, max_tokens=250, repeat={"not_relevant": 3})
    flagged = {i for i in (answer or {}).get("not_relevant", []) if 0 <= i < len(candidates)}
    if not flagged or len(flagged) > 0.6 * len(candidates):
        return []
    return [candidates[i]["keyword"] for i in sorted(flagged)]


def expand(ctx: Context, seeds: list[str], local: bool, location: str) -> dict[str, dict]:
    """Ask autocomplete for each seed and a few buyer-intent variations of it."""
    lang = ctx.lang
    vocabulary = page_vocabulary(ctx)
    templates = list(TEMPLATES.get(lang, ["{s}"]))
    if local and lang in LOCAL_TEMPLATES:
        templates.append(LOCAL_TEMPLATES[lang])
    if local and location:
        templates.append("{s} " + location.lower())
    found: dict[str, dict] = {}
    for seed in seeds:
        seed_tokens = {t[:5] for t in content_tokens(seed)}
        entry = found.setdefault(seed, {"keyword": seed, "cluster": seed, "score": 0.0, "sources": ["seed"]})
        for template in templates:
            query = template.format(s=seed)
            suggestions = suggest(ctx.fetcher.client, query, ctx.config.country, lang)
            for position, phrase in enumerate(suggestions[:10]):
                if not on_topic(phrase, seed_tokens, vocabulary, location):
                    continue
                if len(phrase.split()) > 7:
                    continue
                weight = (10 - position) / 10 * (1.5 if template == "{s}" else 1.0)
                row = found.setdefault(phrase, {"keyword": phrase, "cluster": seed, "score": 0.0, "sources": []})
                row["score"] += weight
                if "autocomplete" not in row["sources"]:
                    row["sources"].append("autocomplete")
        if entry["score"] == 0:
            entry["score"] = 0.5
    return found


def run(ctx: Context) -> dict:
    brief = ctx.brief
    seeds = [s for s in brief.get("seed_keywords", []) if s][:8]
    if not seeds:
        seeds = [brief.get("category", "").lower()] if brief.get("category") else []
    local = brief.get("business_type") == "local_service"
    ctx.emit("info", f"Expanding {len(seeds)} seed keywords with autocomplete")
    found = expand(ctx, seeds, local, brief.get("location", ""))

    competitors = ctx.results.get("competitors", {}).get("competitors", [])
    brand_names = {brief.get("brand", "").lower()} | {c["name"].lower() for c in competitors if len(c["name"]) > 3}
    brand_names.discard("")
    free_offer = "free_trial" in brief.get("offers", [])

    notes: list[str] = []
    metrics: dict[str, dict] = {}
    if ctx.config.keyword_csv:
        try:
            metrics = parse_keyword_csv(ctx.config.keyword_csv)
            ctx.emit("info", f"Loaded {len(metrics)} keywords with real metrics from the CSV")
        except (OSError, ValueError) as exc:
            notes.append(f"Could not read the keyword CSV: {exc}")
    for keyword, values in metrics.items():
        if keyword not in found and values.get("volume"):
            cluster = max(seeds, key=lambda s: len(set(content_tokens(s)) & set(content_tokens(keyword))), default="other")
            found[keyword] = {"keyword": keyword, "cluster": cluster, "score": 0.0, "sources": []}
        if keyword in found:
            found[keyword].update(values)
            found[keyword]["sources"].append("keyword_planner")

    top = max((row["score"] for row in found.values()), default=1.0) or 1.0
    keywords, negatives = [], []
    for row in found.values():
        phrase = row["keyword"]
        if NEGATIVE.search(phrase) or (FREE.search(phrase) and not free_offer):
            negatives.append(phrase)
            continue
        row["intent"] = classify_intent(phrase, brand_names, brief.get("location", ""))
        row["interest"] = round(100 * row.pop("score") / top)
        keywords.append(row)
    has_volumes = any("volume" in k for k in keywords)
    keywords.sort(key=lambda k: (-(k.get("volume") or 0), -k["interest"]))
    removed_by_model = model_filter(ctx, keywords, seeds)
    keywords = [k for k in keywords if k["keyword"] not in removed_by_model]

    clusters = []
    for seed in seeds:
        members = [k for k in keywords if k["cluster"] == seed]
        if not members:
            continue
        intents = [k["intent"] for k in members]
        clusters.append({
            "name": seed, "count": len(members),
            "main_intent": max(set(intents), key=intents.count),
            "interest": round(sum(k["interest"] for k in members) / len(members)),
            "volume": sum(k.get("volume") or 0 for k in members) if has_volumes else None,
            "top": [k["keyword"] for k in members[:6]],
        })
    clusters.sort(key=lambda c: (-(c["volume"] or 0), -c["interest"] * c["count"]))

    # Words that waste budget: negative terms seen in autocomplete, reduced to the offending word.
    negative_terms = dedupe([m.group(0) for n in negatives for m in [NEGATIVE.search(n) or FREE.search(n)] if m])
    conquest = [f"{c['name'].lower()} alternative" for c in competitors if not c.get("error")][:5]

    with open(ctx.run_dir / "keywords.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["keyword", "cluster", "intent", "relative_interest", "avg_monthly_searches",
                         "bid_low", "bid_high", "competition", "sources"])
        for k in keywords:
            writer.writerow([k["keyword"], k["cluster"], k["intent"], k["interest"], k.get("volume", ""),
                             k.get("cpc_low", ""), k.get("cpc_high", ""), k.get("competition", ""),
                             "+".join(k["sources"])])
    (ctx.run_dir / "keywords_for_keyword_planner.txt").write_text(
        "\n".join(k["keyword"] for k in keywords[:200]) + "\n", encoding="utf-8")

    if not has_volumes:
        notes.append(
            "Search volumes and bids are not included: no free source publishes them. Paste "
            "keywords_for_keyword_planner.txt into Google Keyword Planner (free with a Google Ads account), "
            "download the CSV and run again with --keyword-csv to get real numbers.")
    notes.append("Relative interest is a 0 to 100 score from autocomplete position and frequency. "
                 "It ranks keywords against each other and is not a search volume.")
    if not any("autocomplete" in k["sources"] for k in keywords):
        notes.append("Autocomplete returned nothing, so only the seed keywords are listed.")
    if removed_by_model:
        notes.append(f"The research model removed {len(removed_by_model)} searches it judged off target "
                     f"(for example: {', '.join(removed_by_model[:4])}). They are listed in result.json.")
    return {"seeds": seeds, "keywords": keywords, "clusters": clusters, "negative_keywords": negative_terms,
            "removed_by_model": removed_by_model,
            "excluded_examples": negatives[:10], "conquest": conquest, "has_volumes": has_volumes, "notes": notes,
            "files": ["keywords.csv", "keywords_for_keyword_planner.txt"]}
