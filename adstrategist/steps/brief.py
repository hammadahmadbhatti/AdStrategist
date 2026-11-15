"""Step 1: understand the product from its page."""
from __future__ import annotations

import re

from ..context import Context, StepError
from ..textutil import (WEB_STOP, clean, content_tokens, dedupe, registrable_domain, tokens, top_phrases,
                        truncate, truncate_sentence, word_caps)
from ..web_fetch import Page, domain_label

BUSINESS_TYPES = ["ecommerce", "local_service", "saas", "app", "other"]
LOCAL_TYPES = {
    "LocalBusiness", "Restaurant", "Dentist", "Plumber", "Electrician", "HomeAndConstructionBusiness",
    "ProfessionalService", "TaxiService", "AutoRepair", "BeautySalon", "HairSalon", "MedicalBusiness",
    "LegalService", "RealEstateAgent", "MovingCompany", "GeneralContractor", "HousePainter", "RoofingContractor",
    "Locksmith", "HVACBusiness", "Physician", "Attorney", "CafeOrCoffeeShop", "Bakery", "DaySpa", "Hotel",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "product_name": {"type": "string", "description": "brand or product name"},
        "category": {"type": "string", "description": "2 to 4 word generic category a buyer would search for"},
        "one_liner": {"type": "string", "description": "one sentence: what it is and for whom"},
        "business_type": {"type": "string", "enum": BUSINESS_TYPES},
        "location": {"type": "string", "description": "city or region served, empty if not local"},
        "audience": {"type": "array", "items": {"type": "string"}},
        "pains": {"type": "array", "items": {"type": "string"}},
        "usps": {"type": "array", "items": {"type": "string"}},
        "seed_keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product_name", "category", "one_liner", "business_type", "location", "audience", "pains",
                 "usps", "seed_keywords"],
    "additionalProperties": False,
}


def _sentence(text: str) -> str:
    text = clean(text)
    if text.isupper() and len(text) > 5:
        text = text.capitalize()  # shouted headings read badly in ads
    return text[:1].upper() + text[1:]


def guess_brand(page: Page) -> str:
    if page.site_name and len(page.site_name) <= 40:
        return page.site_name
    label = domain_label(page.final_url or page.url)
    for segment in _title_segments(page.title):
        if label.replace("-", "") in re.sub(r"[^a-z0-9]", "", segment.lower()) and len(segment) <= 40:
            return segment
    return label.replace("-", " ").title()


def _title_segments(title: str) -> list[str]:
    return [s.strip() for s in re.split(r"\s+[|\-–—·:]\s+|\s*\|\s*", title or "") if s.strip()]


def guess_category(page: Page, brand: str) -> str:
    brand_key = re.sub(r"[^a-z0-9]", "", brand.lower())
    for segment in _title_segments(page.title):
        if brand_key and brand_key in re.sub(r"[^a-z0-9]", "", segment.lower()):
            continue
        if 1 <= len(segment.split()) <= 5 and not re.search(r"[,.!?]", segment):
            return segment
    return ""  # the title is a slogan: the caller falls back to the strongest phrase on the page


# Review widgets and testimonials sit in lists and headings but are not selling points.
_REVIEW_NOISE = re.compile(r"verifiziert|verified|vor \d+ (?:tag|stund|woch|monat)|\d+ (?:days?|hours?|weeks?) ago|"
                           r"basierend auf|based on \d|\bsterne?\b|\bstars?\b", re.I)


def _is_noise(text: str) -> bool:
    words = text.lower().split()
    repeated = any(words[i : i + 3] == words[i + 3 : i + 6] for i in range(len(words) - 5))
    return bool(_REVIEW_NOISE.search(text)) or repeated


def _selling_points(page: Page) -> list[str]:
    points = [_sentence(truncate(b, 90)) for b in page.bullets
              if 3 <= len(b.split()) <= 14 and not _is_noise(b)][:4]
    for heading in page.headings:
        words = content_tokens(heading)
        if len(points) >= 6:
            break
        if 3 <= len(heading.split()) <= 10 and not heading.rstrip().endswith("?") and \
                not _is_noise(heading) and any(w not in WEB_STOP for w in words):
            points.append(_sentence(truncate(heading, 90)))
    return dedupe(points)[:6]


def guess_business_type(page: Page, text: str) -> tuple[str, int]:
    low = text.lower()
    if re.search(r"apps\.apple\.com|play\.google\.com", page.final_url or page.url):
        return "app", 5
    types = set(page.schema_types)
    scores = dict.fromkeys(["ecommerce", "local_service", "saas", "app"], 0)
    if types & {"Product", "Offer", "Store", "OnlineStore", "AggregateOffer"}:
        scores["ecommerce"] += 2
    scores["ecommerce"] += 2 * bool(re.search(r"add to cart|add to bag|in den warenkorb|warenkorb|checkout|zur kasse", low))
    scores["ecommerce"] += bool(re.search(r"free shipping|versand|lieferung|returns|rückgabe", low))
    if types & LOCAL_TYPES:
        scores["local_service"] += 3
    scores["local_service"] += page.has_phone + bool(page.locality)
    scores["local_service"] += bool(re.search(
        r"appointment|termin|free quote|angebot anfordern|book (?:now|online)|opening hours|öffnungszeiten|vor ort", low))
    scores["saas"] += 2 * ("SoftwareApplication" in types or "WebApplication" in types)
    scores["saas"] += len(set(re.findall(
        r"free trial|sign up|log ?in|dashboard|\bapi\b|software|platform|integrations?|kostenlos testen|plattform", low)))
    scores["app"] += 2 * bool(page.app_links) + bool(re.search(r"download the app|app herunterladen|get the app", low))
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] >= 2 else ("other", scores[best])


def heuristic_brief(page: Page, subs: list[Page]) -> dict:
    brand = guess_brand(page)
    all_text = " ".join([page.text] + [s.text for s in subs])
    business_type, confidence = guess_business_type(page, all_text)
    exclude = set(tokens(brand)) | {domain_label(page.final_url or page.url)}
    weighted = [(page.title, 3.0), (" . ".join(page.h1), 3.0), (page.description, 2.0),
                (" . ".join(page.headings[:15]), 1.5)]
    weighted += [(s.title, 1.5) for s in subs] + [(" . ".join(s.h1), 1.5) for s in subs]
    category = guess_category(page, brand)
    seeds = top_phrases(weighted, exclude, limit=8, body=all_text)
    category_from = "page title"
    if category:
        seeds = [category.lower()] + seeds
    else:
        category, category_from = (word_caps(seeds[0]) if seeds else brand), "page phrase"
    usps = _selling_points(page)
    return {
        "brand": brand, "category": category, "category_from": category_from,
        "one_liner": truncate_sentence(page.description or (page.h1[0] if page.h1 else page.title), 200),
        "business_type": business_type, "_type_confidence": confidence,
        "location": page.locality, "audience": [], "pains": [], "usps": dedupe(usps)[:6],
        "seed_keywords": dedupe(seeds)[:8],
    }


# What a brief needs to know, phrased as retrieval queries in the page's language.
BRIEF_QUERIES = {
    "en": ["what the product or service is and who it is for", "prices, plans, offers, shipping, guarantee",
           "benefits and reasons customers choose it"],
    "de": ["was das Produkt oder die Dienstleistung ist und für wen", "Preise, Angebote, Versand, Garantie",
           "Vorteile und Gründe, warum Kunden es wählen"],
}


def retrieved_context(ctx, language: str) -> tuple[str, int]:
    """The passages from all of the site's pages that best answer the brief's questions.
    This replaces "the first 2,200 characters of the homepage" when retrieval is on."""
    from ..rag import join_passages

    found = []
    for query in BRIEF_QUERIES.get(language[:2], BRIEF_QUERIES["en"]):
        found += ctx.passages(query, k=3, site="own")
    found.sort(key=lambda p: -p.score)
    return join_passages(found, limit=2400), len({p.text[:80] for p in found})


def _prompt(page: Page, subs: list[Page], language: str, passages: str = "") -> str:
    lines = [
        f"URL: {page.final_url}",
        f"Title: {page.title}",
        f"Meta description: {page.description}",
        f"Main headings: {' | '.join(page.h1 + page.headings[:12])}",
        f"Buttons: {' | '.join(page.ctas[:6])}",
        f"Prices seen: {', '.join(page.prices[:5]) or 'none'}",
    ]
    if passages:
        lines.append(f"Most relevant passages from the site's pages:\n{passages}")
    else:
        lines.append(f"Page text: {page.text[:2200]}")
        for sub in subs[:2]:
            lines.append(f"Another page ({sub.title}): {sub.text[:700]}")
    return (
        "Read this product page and describe the business for an advertising plan.\n"
        f"Write values in {language}. Use only what the page says.\n"
        "seed_keywords: 6 to 8 short search phrases (2 to 4 words) a buyer would type into Google to find this "
        "kind of product. Generic phrases, no brand name.\n"
        "audience, pains: up to 3 short items each. usps: up to 4 specific selling points taken from the page, "
        "3 to 10 words each.\n\n" + "\n".join(lines)
    )


def run(ctx: Context) -> dict:
    page = ctx.fetcher.fetch(ctx.config.url)
    if not page.ok:
        raise StepError(f"Could not read {ctx.config.url}: {page.error or 'the page is empty'}.")
    # With retrieval, more pages can be read: only the relevant passages reach the prompt.
    subs = ctx.fetcher.fetch_subpages(page, limit=5 if ctx.rag is not None else 2)
    ctx.own_page, ctx.sub_pages = page, subs
    for own in [page, *subs]:
        ctx.index_page(own, "own")

    brief = heuristic_brief(page, subs)
    confidence = brief.pop("_type_confidence")
    language = ctx.config.language or page.lang or "en"
    warnings = []
    if len(page.text) < 400:
        warnings.append("The page has very little readable text. It may be rendered by JavaScript, "
                        "so the brief is thin. Add a competitor list and check the keywords by hand.")

    used_model = False
    if ctx.llm.enabled("research"):
        from ..textutil import LANGUAGE_NAMES
        passages, passage_count = retrieved_context(ctx, language) if ctx.rag is not None else ("", 0)
        answer = ctx.llm.ask_json("product brief",
                                  _prompt(page, subs, LANGUAGE_NAMES.get(language[:2], language), passages),
                                  SCHEMA, max_tokens=700,
                                  repeat={"seed_keywords": 7, "usps": 4, "audience": 3, "pains": 3})
        if answer:
            used_model = True
            brand_tokens = set(tokens(brief["brand"]))
            title_category = brief["category"].lower() if brief.get("category_from") == "page title" else ""
            if answer["product_name"] and len(answer["product_name"]) <= 40:
                brief["brand"] = answer["product_name"]
            if answer["category"] and len(answer["category"].split()) <= 6:
                brief["category"], brief["category_from"] = answer["category"], "model"
            if answer["one_liner"]:
                brief["one_liner"] = _sentence(truncate_sentence(answer["one_liner"], 220))
            if answer["business_type"] != "other" and confidence < 3:
                brief["business_type"] = answer["business_type"]
            if answer["location"] and not brief["location"] and len(answer["location"]) <= 40:
                # only trust a location the page actually mentions
                if answer["location"].lower() in page.text.lower():
                    brief["location"] = answer["location"]
            for key in ("audience", "pains"):
                brief[key] = [_sentence(truncate(v, 110)) for v in answer[key]][:4]
            if answer["usps"]:
                specific = [_sentence(truncate(v, 110)) for v in answer["usps"]
                            if len(v.split()) >= 2 and ctx.supported(v, "selling point")]
                # As with seeds: two points from a model that read the page beat headings picked by rule.
                brief["usps"] = specific[:6] if len(specific) >= 2 else dedupe(specific + brief["usps"])[:6]
            model_seeds = dedupe([clean(s).lower() for s in answer["seed_keywords"]
                                  if 1 <= len(s.split()) <= 5 and not (set(tokens(s)) & brand_tokens)])
            # Single words ("organization") drift far off topic in autocomplete, so phrases only. German
            # is the exception: its keywords are often one long compound ("grillgewürze").
            compound = 8 if language[:2] == "de" else 99
            model_seeds = [s for s in model_seeds if len(s.split()) >= 2 or len(s) >= compound] or model_seeds[:2]
            if title_category and len(title_category.split()) <= 4:
                model_seeds = dedupe([title_category] + model_seeds)
            # Seeds from a model that read the page beat frequency guesses, which pick up page furniture.
            # A short clean list is better than a long polluted one; competitor vocabulary tops it up later.
            brief["seed_keywords"] = (model_seeds if len(model_seeds) >= 2 else
                                      dedupe(model_seeds + brief["seed_keywords"]))[:8]

    given = dedupe([clean(s).lower() for s in ctx.config.seed_keywords if clean(s)])
    if given:  # what the user typed always wins; other seeds only top up a short list
        brief["seed_keywords"] = dedupe(given + (brief["seed_keywords"] if len(given) < 3 else []))[:10]

    brief.update({
        "url": ctx.config.url, "final_url": page.final_url, "domain": registrable_domain(page.final_url),
        "language": language[:2], "prices": page.prices, "offers": page.offers, "trackers": page.trackers,
        "ctas": page.ctas, "images": page.images,
        "evidence": {"title": page.title, "description": page.description, "h1": page.h1,
                     "pages_read": [page.final_url] + [s.final_url for s in subs]},
        "written_by_model": used_model, "warnings": warnings,
        "retrieval": {"used": bool(used_model and ctx.rag is not None and passage_count), "passages": passage_count}
        if used_model else {"used": False, "passages": 0},
    })
    if not used_model and not given:
        warnings.append("No model read this page, so the category and seed keywords were picked by word "
                        "frequency, not by understanding. Check them, or pass your own search phrases.")
    for warning in warnings:
        ctx.emit("warn", warning)
    return brief
