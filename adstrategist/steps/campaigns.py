"""Step 8: campaigns per platform. The model writes copy, code enforces every platform limit."""
from __future__ import annotations

import json
import re

from ..context import Context
from ..exports import write_campaign_files
from ..textutil import clean, dedupe, truncate, truncate_sentence, word_caps

LIMITS = {
    "google": {"headline": 30, "description": 90, "path": 15, "sitelink": 25, "callout": 25},
    "meta": {"primary_text": 300, "headline": 40, "description": 30},
    "tiktok": {"ad_text": 100},
}
COUNTRY_NAMES = {"DE": "Germany", "AT": "Austria", "CH": "Switzerland", "US": "United States",
                 "GB": "United Kingdom", "FR": "France", "ES": "Spain", "IT": "Italy", "NL": "Netherlands",
                 "MY": "Malaysia", "CA": "Canada", "AU": "Australia", "PL": "Poland", "TR": "Turkey"}

# Template copy only says things that are true of any business. Offer lines are added
# only when the matching offer was detected on the page.
PACK = {
    "en": {
        "cta": {"ecommerce": "Shop Now", "local_service": "Contact Us", "saas": "Sign Up", "app": "Download",
                "other": "Learn More"},
        "cta_line": {"ecommerce": "Shop online today.", "local_service": "Get in touch today.",
                     "saas": "Get started today.", "app": "Download the app today.", "other": "Find out more today."},
        "hook": "Looking for {category}?", "by": "{category} by {brand}", "visit": "Visit {brand} Today",
        "official": "{brand} Official Site", "discover": "Discover {brand}", "more": "Learn More About {brand}",
        "offers": {"pricing_shown": "See Prices Online", "free_trial": "Start Your Free Trial",
                   "free_quote": "Get Your Free Quote", "online_booking": "Book Online Now",
                   "free_shipping": "Free Shipping Available", "guarantee": "Backed by Our Guarantee"},
        "angles": ["Problem and solution", "Key benefits", "Direct offer"],
        "retarget": "Website visitors, last 30 days", "broad": "Broad, Advantage+ audience",
    },
    "de": {
        "cta": {"ecommerce": "Jetzt einkaufen", "local_service": "Kontakt aufnehmen", "saas": "Registrieren",
                "app": "Herunterladen", "other": "Mehr dazu"},
        "cta_line": {"ecommerce": "Jetzt online bestellen.", "local_service": "Jetzt Kontakt aufnehmen.",
                     "saas": "Jetzt starten.", "app": "Jetzt App herunterladen.", "other": "Jetzt mehr erfahren."},
        "hook": "Auf der Suche nach {category}?", "by": "{category} von {brand}", "visit": "Jetzt {brand} besuchen",
        "official": "{brand} Offizielle Seite", "discover": "{brand} entdecken", "more": "Mehr über {brand} erfahren",
        "offers": {"pricing_shown": "Preise online ansehen", "free_trial": "Jetzt kostenlos testen",
                   "free_quote": "Kostenloses Angebot anfordern", "online_booking": "Jetzt online buchen",
                   "free_shipping": "Kostenloser Versand möglich", "guarantee": "Mit Garantie"},
        "angles": ["Problem und Lösung", "Wichtigste Vorteile", "Direktes Angebot"],
        "retarget": "Website-Besucher, letzte 30 Tage", "broad": "Breit, Advantage+ Zielgruppe",
    },
}
META_OBJECTIVE = {"ecommerce": "Sales", "local_service": "Leads", "saas": "Leads", "app": "App promotion",
                  "other": "Traffic"}
TIKTOK_OBJECTIVE = {"ecommerce": "Website conversions", "local_service": "Lead generation",
                    "saas": "Website conversions", "app": "App promotion", "other": "Traffic"}
LEGAL_LINK = re.compile(r"impressum|privacy|datenschutz|agb|terms|cookie|login|anmelden|cart|warenkorb|legal", re.I)

RSA_SCHEMA = {
    "type": "object",
    "properties": {"headlines": {"type": "array", "items": {"type": "string"}},
                   "descriptions": {"type": "array", "items": {"type": "string"}}},
    "required": ["headlines", "descriptions"], "additionalProperties": False,
}
META_SCHEMA = {
    "type": "object",
    "properties": {"ads": {"type": "array", "items": {
        "type": "object",
        "properties": {"angle": {"type": "string"}, "primary_text": {"type": "string"},
                       "headline": {"type": "string"}, "description": {"type": "string"},
                       "visual": {"type": "string", "description": "what the image or video should show"}},
        "required": ["angle", "primary_text", "headline", "description", "visual"],
        "additionalProperties": False}}},
    "required": ["ads"], "additionalProperties": False,
}
TIKTOK_SCHEMA = {
    "type": "object",
    "properties": {"ads": {"type": "array", "items": {
        "type": "object",
        "properties": {"hook": {"type": "string", "description": "first 2 seconds, spoken"},
                       "ad_text": {"type": "string"},
                       "script": {"type": "array", "items": {"type": "string"}},
                       "hashtags": {"type": "array", "items": {"type": "string"}}},
        "required": ["hook", "ad_text", "script", "hashtags"], "additionalProperties": False}}},
    "required": ["ads"], "additionalProperties": False,
}


def pack_for(lang: str) -> dict:
    return PACK.get(lang, PACK["en"])


def fit(text: str, limit: int, headline: bool = False) -> str:
    """Clean a line of copy and return '' when it cannot honestly fit the limit."""
    text = clean(text).strip("\"'“”„ ")
    if headline:
        text = text.replace("!", "").rstrip(".")
    if text.isupper() and len(text) > 4:
        text = text.capitalize()
    return text if 0 < len(text) <= limit else ""


def facts_block(brief: dict) -> str:
    return json.dumps({
        "brand": brief.get("brand"), "category": brief.get("category"), "what_it_is": brief.get("one_liner"),
        "audience": brief.get("audience"), "pains": brief.get("pains"), "selling_points": brief.get("usps"),
        "offers_on_page": brief.get("offers"), "prices_on_page": brief.get("prices", [])[:3],
        "location": brief.get("location"),
    }, ensure_ascii=False)


def _template_headlines(brief: dict, keywords: list[str], pack: dict) -> list[str]:
    fmt = {"brand": brief["brand"], "category": word_caps(brief["category"])}
    lines = [word_caps(k) for k in keywords[:5]]
    lines += [brief["brand"], pack["by"].format(**fmt), fmt["category"], pack["official"].format(**fmt)]
    lines += [pack["offers"][o] for o in brief.get("offers", []) if o in pack["offers"]]
    lines += [u for u in brief.get("usps", [])]
    lines += [pack["visit"].format(**fmt), pack["discover"].format(**fmt), pack["more"].format(**fmt),
              pack["cta"][brief["business_type"]]]
    return lines


def _template_descriptions(brief: dict, pack: dict) -> list[str]:
    cta = pack["cta_line"][brief["business_type"]]
    usps = brief.get("usps", [])
    lines = [truncate_sentence(brief.get("one_liner", ""), 90)]
    lines.append(f"{pack['by'].format(brand=brief['brand'], category=word_caps(brief['category']))}. {cta}")
    if len(usps) >= 2:
        lines.append(f"{usps[0].rstrip('.')}. {usps[1].rstrip('.')}.")
    lines += [truncate(u, 90) for u in usps]
    lines.append(f"{pack['discover'].format(brand=brief['brand'])}. {cta}")
    return lines


def build_google(ctx: Context, brief: dict, daily: float | None) -> dict:
    pack = pack_for(ctx.lang)
    limit = LIMITS["google"]
    data = ctx.results.get("keywords", {})
    clusters = sorted(data.get("clusters", []),
                      key=lambda c: c["main_intent"] not in ("transactional", "commercial"))[:4]
    by_cluster = {c["name"]: [k for k in data.get("keywords", []) if k["cluster"] == c["name"]] for c in clusters}
    if not clusters:
        seeds = brief.get("seed_keywords", [])[:6] or [brief["category"].lower()]
        clusters = [{"name": brief["category"].lower(), "main_intent": "commercial"}]
        by_cluster = {clusters[0]["name"]: [{"keyword": s, "intent": "commercial"} for s in seeds]}

    ad_groups = []
    for cluster in clusters:
        members = sorted(by_cluster[cluster["name"]],
                         key=lambda k: k["intent"] not in ("transactional", "commercial"))[:10]
        phrases = [k["keyword"] for k in members]
        model = None
        if ctx.llm.enabled("writing"):
            model = ctx.llm.ask_json(
                f"google ad copy: {cluster['name']}",
                f"Write Google responsive search ad copy in {ctx.language_name}.\n"
                f"8 headlines, each at most {limit['headline']} characters, no exclamation marks.\n"
                f"3 descriptions, each at most {limit['description']} characters.\n"
                "Include the main keyword in at least 2 headlines. Only claim what the facts support.\n\n"
                f"Keywords: {', '.join(phrases[:6])}\nFacts: {facts_block(brief)}{ctx.copy_feedback}",
                RSA_SCHEMA, role="writing", max_tokens=500, repeat={"headlines": 8, "descriptions": 3})
        candidates = (model["headlines"] if model else []) + _template_headlines(brief, phrases, pack)
        headlines = dedupe([h for h in (fit(c, limit["headline"], headline=True) for c in candidates)
                            if h and ctx.supported(h, "Google headline")])[:15]
        candidates = (model["descriptions"] if model else []) + _template_descriptions(brief, pack)
        descriptions = dedupe([d for d in (fit(c, limit["description"]) for c in candidates)
                               if len(d) >= 35 and ctx.supported(d, "Google description")])[:4]
        path = re.sub(r"[^\w]+", "-", cluster["name"].split()[0], flags=re.UNICODE).strip("-")[: limit["path"]]
        ad_groups.append({
            "name": word_caps(cluster["name"]),
            "keywords": [{"text": p, "match": "Phrase"} for p in phrases] +
                        [{"text": p, "match": "Exact"} for p in phrases[:3]],
            "ad": {"headlines": headlines, "descriptions": descriptions, "path1": path, "path2": "",
                   "final_url": brief["final_url"], "written_by_model": bool(model)},
            "ready": len(headlines) >= 3 and len(descriptions) >= 2,
        })

    page = ctx.own_page
    sitelinks = [{"text": l["text"], "url": l["url"]} for l in (page.links if page else [])
                 if len(l["text"]) <= limit["sitelink"] and not LEGAL_LINK.search(l["url"] + l["text"])
                 and l["url"].rstrip("/") != brief["final_url"].rstrip("/")]
    callouts = [u for u in brief.get("usps", []) if len(u) <= limit["callout"]][:4]
    return {
        "platform": "google", "name": f"{brief['brand']} | Search | {ctx.config.country}",
        "type": "Search", "networks": "Google search", "status": "Paused",
        "daily_budget": daily, "bid_strategy": "Maximize clicks",
        "bid_note": "Start with Maximize clicks. Switch to Maximize conversions after about 30 conversions.",
        "location": COUNTRY_NAMES.get(ctx.config.country, ctx.config.country), "language": ctx.lang,
        "ad_groups": ad_groups, "negative_keywords": data.get("negative_keywords", []),
        "sitelinks": dedupe(sitelinks, key=lambda s: s["text"].lower())[:4], "callouts": callouts,
        "conquest_ideas": data.get("conquest", []),
    }


def _template_meta(brief: dict, pack: dict) -> list[dict]:
    fmt = {"brand": brief["brand"], "category": brief["category"]}
    usps = brief.get("usps", [])
    cta = pack["cta_line"][brief["business_type"]]
    bullets = "\n".join(f"✓ {u}" for u in usps[:3])
    offer = next((pack["offers"][o] for o in brief.get("offers", []) if o in pack["offers"]), "")
    bodies = [
        f"{pack['hook'].format(**fmt)}\n\n{brief.get('one_liner', '')}\n\n{cta}",
        f"{pack['by'].format(**fmt)}\n\n{bullets}\n\n{cta}" if bullets else f"{brief.get('one_liner', '')}\n\n{cta}",
        f"{brief.get('one_liner', '')}\n\n{offer + '. ' if offer else ''}{cta}",
    ]
    heads = [pack["by"].format(**fmt), brief["brand"], offer or pack["discover"].format(**fmt)]
    return [{"angle": pack["angles"][i], "primary_text": bodies[i], "headline": heads[i],
             "description": pack["cta"][brief["business_type"]],
             "visual": "Product or service in use, real photo, brand colours, little or no text on the image."}
            for i in range(3)]


def build_meta(ctx: Context, brief: dict, daily: float | None) -> dict:
    pack = pack_for(ctx.lang)
    limit = LIMITS["meta"]
    model = None
    if ctx.llm.enabled("writing"):
        model = ctx.llm.ask_json(
            "meta ad copy",
            f"Write 3 Facebook and Instagram ads in {ctx.language_name}, each with a different angle "
            "(for example: problem and solution, key benefit, direct offer).\n"
            f"primary_text: 2 to 4 short lines, the first line must hook. At most {limit['primary_text']} characters.\n"
            f"headline: at most {limit['headline']} characters. description: at most {limit['description']} characters.\n"
            "Only claim what the facts support. No hashtags.\n\n" + f"Facts: {facts_block(brief)}{ctx.copy_feedback}",
            META_SCHEMA, role="writing", max_tokens=1000, repeat={"ads": 3})
    templates = _template_meta(brief, pack)
    ads = []
    for i in range(3):
        base = templates[i]
        draft = dict(model["ads"][i]) if model and i < len(model["ads"]) else {}
        for key in ("primary_text", "headline", "description"):
            if draft.get(key) and not ctx.supported(draft[key], f"Meta {key.replace('_', ' ')}"):
                draft[key] = ""  # the template line for this field is used instead
        body = clean(draft.get("primary_text", "")) and draft["primary_text"].strip()
        if body and len(body) < 80:  # small models stop after the hook
            body = f"{body}\n\n{brief.get('one_liner', '')}\n\n{pack['cta_line'][brief['business_type']]}".strip()
        ads.append({
            "name": f"Ad {i + 1}: {truncate(draft.get('angle') or base['angle'], 40)}",
            "angle": truncate(draft.get("angle") or base["angle"], 60),
            "primary_text": (body if body and len(body) <= limit["primary_text"] else "") or
                            base["primary_text"][: limit["primary_text"]],
            "headline": fit(draft.get("headline", ""), limit["headline"]) or truncate(base["headline"], limit["headline"]),
            "description": fit(draft.get("description", ""), limit["description"]) or
                           truncate(base["description"], limit["description"]),
            "call_to_action": PACK["en"]["cta"][brief["business_type"]],
            "visual": truncate(draft.get("visual") or base["visual"], 200),
            "link": brief["final_url"], "written_by_model": bool(draft),
        })
    has_pixel = "meta" in brief.get("trackers", [])
    ad_sets = [{"name": pack["broad"], "audience": "Advantage+ audience, no interest targeting",
                "countries": ctx.config.country, "age": "18 to 65+",
                "placements": "Advantage+ placements (Feed, Stories, Reels on Facebook and Instagram)"}]
    if has_pixel:
        ad_sets.append({"name": pack["retarget"], "audience": "Custom audience: website visitors, 30 days",
                        "countries": ctx.config.country, "age": "18 to 65+", "placements": "Advantage+ placements"})
    return {
        "platform": "meta", "name": f"{brief['brand']} | {META_OBJECTIVE[brief['business_type']]} | {ctx.config.country}",
        "objective": META_OBJECTIVE[brief["business_type"]], "status": "Paused", "daily_budget": daily,
        "budget_type": "Campaign budget (Advantage campaign budget)", "ad_sets": ad_sets, "ads": ads,
        "pixel_note": "" if has_pixel else
        "No Meta Pixel was found on the site. Install it before launch, otherwise conversions cannot be "
        "measured and retargeting is not possible.",
    }


def build_tiktok(ctx: Context, brief: dict, daily: float | None) -> dict:
    pack = pack_for(ctx.lang)
    limit = LIMITS["tiktok"]["ad_text"]
    fmt = {"brand": brief["brand"], "category": brief["category"]}
    model = None
    if ctx.llm.enabled("writing"):
        model = ctx.llm.ask_json(
            "tiktok ad copy",
            f"Write 3 TikTok ads in {ctx.language_name}.\n"
            "hook: the spoken first 2 seconds, casual, under 12 words.\n"
            f"ad_text: the caption, 12 to {limit} characters.\n"
            "script: 4 short shots for a 15 to 25 second vertical video, one sentence each, ending with a call "
            "to action.\nhashtags: 3 to 5 relevant hashtags.\nOnly claim what the facts support.\n\n"
            f"Facts: {facts_block(brief)}{ctx.copy_feedback}",
            TIKTOK_SCHEMA, role="writing", max_tokens=1100,
            repeat={"ads": 3, "script": 4, "hashtags": 4})
    usps = brief.get("usps", [])
    cta_line = pack["cta_line"][brief["business_type"]]
    base_script = [pack["hook"].format(**fmt)] + [u.rstrip(".") + "." for u in usps[:2]] + [cta_line]
    base_tags = [re.sub(r"[^\w]", "", w, flags=re.UNICODE).lower()
                 for w in [brief["brand"], *brief["category"].split()[:2]]]
    ads = []
    for i in range(3):
        draft = dict(model["ads"][i]) if model and i < len(model["ads"]) else {}
        for key in ("hook", "ad_text"):
            if draft.get(key) and not ctx.supported(draft[key], f"TikTok {key.replace('_', ' ')}"):
                draft[key] = ""
        if draft.get("script"):
            draft["script"] = [s for s in draft["script"] if ctx.supported(s, "TikTok script")]
            if len(draft["script"]) < 2:
                draft["script"] = []
        text = fit(draft.get("ad_text", ""), limit) or truncate(
            f"{pack['by'].format(**fmt)}. {cta_line}", limit)
        tags = [re.sub(r"[^\w]", "", t, flags=re.UNICODE).lower() for t in draft.get("hashtags", [])]
        ads.append({
            "name": f"Video {i + 1}",
            "hook": truncate(draft.get("hook") or pack["hook"].format(**fmt), 120),
            "ad_text": text,
            "script": [truncate(s, 200) for s in draft.get("script", [])][:6] or base_script,
            "hashtags": dedupe([t for t in tags if t] or [t for t in base_tags if t])[:5],
            "call_to_action": PACK["en"]["cta"][brief["business_type"]],
            "link": brief["final_url"], "written_by_model": bool(draft),
        })
        if not model:
            break  # template copy has one honest variant, not three near-duplicates
    return {
        "platform": "tiktok", "name": f"{brief['brand']} | TikTok | {ctx.config.country}",
        "objective": TIKTOK_OBJECTIVE[brief["business_type"]], "status": "Paused", "daily_budget": daily,
        "ad_group": {"name": "Broad", "placement": "TikTok", "location": ctx.config.country, "age": "18+",
                     "bidding": "Lowest cost", "format": "Vertical 9:16 video, 15 to 25 seconds"},
        "ads": ads,
        "pixel_note": "" if "tiktok" in brief.get("trackers", []) else
        "No TikTok Pixel was found on the site. Install it before launch to measure conversions.",
    }


COPY_PLACES = ("Google", "Meta", "TikTok")


def review_copy(ctx: Context, since: int) -> str:
    """Deterministic reviewer for the LangGraph engine's revise loop.

    Looks at what the claim check threw out of the latest draft and at ad groups left without enough
    usable copy, and turns that into notes for a second draft. Returns '' when the draft needs no rework.
    """
    rejected = [c for c in ctx.rejected_claims[since:] if c["where"].startswith(COPY_PLACES)]
    thin = [g["name"] for c in ctx.results.get("campaigns", {}).get("campaigns", []) if c["platform"] == "google"
            for g in c["ad_groups"] if not g["ready"]]
    if not rejected and not thin:
        return ""
    lines = ["A reviewer rejected parts of your previous draft. Write new lines and do not repeat these:"]
    lines += [f"- \"{c['text'][:90]}\" was removed because {c['reason']}." for c in rejected[:8]]
    if thin:
        lines.append(f"- These ad groups did not get enough usable lines: {', '.join(thin[:4])}.")
    lines.append("Use only the facts given. No numbers, discounts, free offers, guarantees or superlatives "
                 "unless they appear in the facts.")
    return "\n\n" + "\n".join(lines)


BUILDERS = {"google": build_google, "meta": build_meta, "tiktok": build_tiktok}


def run(ctx: Context) -> dict:
    brief = ctx.brief
    budget = ctx.results.get("budget", {})
    daily = {row["platform"]: row["daily"] for row in budget.get("plan", [])}
    not_funded = set(budget.get("not_funded", []))
    campaigns = []
    for platform in ctx.config.platforms:
        if platform not in BUILDERS:
            continue
        ctx.emit("info", f"Building {platform} campaign")
        campaign = BUILDERS[platform](ctx, brief, daily.get(platform))
        campaign["funded"] = platform not in not_funded
        campaigns.append(campaign)
    files = write_campaign_files(ctx.run_dir, campaigns, ctx.config.currency)
    return {
        "campaigns": campaigns, "files": files, "limits": LIMITS,
        "notes": [
            "Everything is created with status Paused. Nothing spends until you enable it.",
            "Read every line before publishing. You are responsible for claims made in your ads.",
            "google_ads_editor.csv follows Google Ads Editor's import columns: use Account > Import > From file "
            "and check the preview. The Meta and TikTok files are campaign sheets to copy into Ads Manager.",
        ] + ([] if ctx.llm.enabled("writing") else
             ["No writing model was used, so the copy is template based. Switch the writing model on for "
              "tailored ad copy."]),
    }
