"""Step 5: market gaps. Facts are computed in code, the model only turns them into a recommendation."""
from __future__ import annotations

from collections import defaultdict

from ..context import Context
from ..textutil import WEB_STOP, content_tokens, truncate
from ..web_fetch import OFFER_LABELS

SCHEMA = {
    "type": "object",
    "properties": {
        "positioning": {"type": "string", "description": "2 sentences: how to position against these competitors"},
        "angles": {"type": "array", "items": {"type": "string"}},
        "quick_wins": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["positioning", "angles", "quick_wins"],
    "additionalProperties": False,
}
PLATFORM_NAMES = {"google": "Google Ads", "meta": "Meta", "tiktok": "TikTok"}


def _terms(text: str) -> set[str]:
    toks = content_tokens(text)
    return set(toks) | {" ".join(toks[i : i + 2]) for i in range(len(toks) - 1)}


def topic_gaps(own_text: str, competitor_headlines: dict[str, str], minimum: int) -> list[dict]:
    """Topics several competitors put in their headlines that our pages never mention."""
    own = set(content_tokens(own_text))
    usage: dict[str, list[str]] = defaultdict(list)
    for name, text in competitor_headlines.items():
        for term in _terms(text):
            usage[term].append(name)
    gaps = []
    for term, names in usage.items():
        words = term.split()
        if len(names) < minimum or all(w in WEB_STOP for w in words):
            continue
        if len(words) == 1 and (len(names) < 3 or len(term) < 6 or term in WEB_STOP):
            continue  # single words are only a finding when most competitors lead with them
        if all(tok in own for tok in term.split()):
            continue
        gaps.append({"topic": term, "used_by": sorted(names)})
    gaps.sort(key=lambda g: (-len(g["used_by"]), -len(g["topic"].split()), g["topic"]))
    kept: list[dict] = []
    for gap in gaps:  # drop single words already covered by a two word topic
        if any(gap["topic"] in k["topic"] and gap["used_by"] == k["used_by"] for k in kept):
            continue
        kept.append(gap)
    return kept[:12]


def add_evidence(ctx: Context, topics: list[dict], offer_gaps: list[dict]) -> int:
    """Retrieval makes each gap checkable: attach the competitor passage it came from, with its link,
    and look for the closest passage on our own pages in case we cover the topic in other words."""
    if ctx.rag is None:
        return 0
    floor = ctx.settings.rag_min_score
    attached = 0
    for finding, query in [(t, t["topic"]) for t in topics] + [(o, o["offer"]) for o in offer_gaps]:
        theirs = [p for p in ctx.passages(query, k=2, exclude_site="own", kind="page") if p.score >= floor]
        if theirs:
            best = theirs[0]
            finding["evidence"] = {"quote": truncate(best.text, 220), "url": best.url, "site": best.site,
                                   "score": best.score}
            attached += 1
        ours = ctx.passages(query, k=1, site="own")
        if ours:
            finding["closest_on_your_site"] = {"quote": truncate(ours[0].text, 160), "url": ours[0].url,
                                               "score": ours[0].score}
            # The word is missing from our pages, but a passage is strongly about the same thing.
            finding["possibly_covered"] = ours[0].score >= max(0.55, floor + 0.25)
    return attached


def run(ctx: Context) -> dict:
    brief = ctx.brief
    competitors = [c for c in ctx.results.get("competitors", {}).get("competitors", []) if not c.get("error")]
    own_text = " ".join([ctx.own_page.text if ctx.own_page else ""] + [p.text for p in ctx.sub_pages])
    total = len(competitors)

    headlines = {c["name"]: " . ".join([c["title"], c["description"]] + c["h1"] + c["usps"]) for c in competitors}
    topics = topic_gaps(own_text, headlines, minimum=2 if total >= 3 else max(1, total)) if total else []

    own_offers = set(brief.get("offers", []))
    offer_gaps, advantages = [], []
    for key, label in OFFER_LABELS.items():
        users = [c["name"] for c in competitors if key in c.get("offers", [])]
        if users and key not in own_offers and len(users) >= max(1, total // 2):
            offer_gaps.append({"offer": label, "used_by": users})
        if key in own_offers and len(users) <= total // 3:
            advantages.append({"offer": label, "competitors_with_it": len(users)})

    intel = ctx.results.get("ad_intel", {})
    channel_gaps = []
    if intel.get("readable"):
        own_platforms = set(intel.get("own_platforms", []))
        hidden = sum(1 for c in intel["competitors"] if c["readable"] and c["tag_manager"] and not c["platforms"])
        for platform, count in intel["summary"].items():
            name = PLATFORM_NAMES[platform]
            if count == 0 and hidden:
                channel_gaps.append({"platform": platform, "kind": "unknown",
                                     "text": f"No {name} tag is visible, but {hidden} of {intel['readable']} "
                                             "competitors load their tags through Google Tag Manager, which hides "
                                             f"them. Check their ad library links before treating {name} as open."})
            elif count == 0:
                channel_gaps.append({"platform": platform, "kind": "open",
                                     "text": f"No competitor shows a {name} tag in its page source. Tags often load only after "
                                             "cookie consent, so confirm in the ad library before treating "
                                             f"{name} as uncontested."})
            elif platform not in own_platforms:
                channel_gaps.append({"platform": platform, "kind": "behind",
                                     "text": f"{count} of {intel['readable']} competitors are set up for {name} "
                                             "and your site is not."})

    keyword_gaps = []
    own_tokens = set(content_tokens(own_text))
    for cluster in ctx.results.get("keywords", {}).get("clusters", []):
        missing = [t for t in content_tokens(cluster["name"]) if t not in own_tokens]
        if missing and cluster["count"] >= 2:
            keyword_gaps.append({"cluster": cluster["name"], "missing_words": missing,
                                 "keywords": cluster["count"]})

    evidence_count = add_evidence(ctx, topics, offer_gaps)
    facts = {
        "evidence_from_competitor_pages": [f"{f['evidence']['site']}: {f['evidence']['quote']}"
                                           for f in topics[:4] + offer_gaps[:3] if f.get("evidence")],
        "product": f"{brief.get('brand')}: {brief.get('one_liner')}",
        "our_selling_points": brief.get("usps", [])[:5],
        "competitors": [f"{c['name']}: {truncate(c['positioning'], 140)}" for c in competitors[:5]],
        "topics_competitors_stress_and_we_do_not": [t["topic"] for t in topics[:8]],
        "offers_competitors_show_and_we_do_not": [g["offer"] for g in offer_gaps],
        "offers_only_we_show": [a["offer"] for a in advantages],
        "channel_findings": [g["text"] for g in channel_gaps],
    }
    recommendation = None
    if competitors and ctx.llm.enabled("writing"):
        import json
        recommendation = ctx.llm.ask_json(
            "gap analysis",
            f"Write in {ctx.language_name}. From these findings, recommend how this product should position "
            "itself in ads. positioning: one single recommendation in 2 sentences, no alternatives. angles: 3 distinct ad angles, one sentence each. quick_wins: up to 4 concrete changes "
            "to the landing page or offer. Refer only to the findings, do not add numbers.\n\n"
            + json.dumps(facts, ensure_ascii=False, indent=1),
            SCHEMA, role="writing", max_tokens=700, repeat={"angles": 3, "quick_wins": 3})
    if not recommendation:
        wins = [f"Add a section about “{t['topic']}”: {len(t['used_by'])} competitors lead with it." for t in topics[:2]]
        wins += [f"Consider showing {g['offer']}: {len(g['used_by'])} of {total} competitors do." for g in offer_gaps[:2]]
        angles = [f"Lead with what only you show: {a['offer']}." for a in advantages[:2]]
        angles += [truncate(u, 120) for u in brief.get("usps", [])[: 3 - len(angles)]]
        recommendation = {
            "positioning": "Rule-based summary (no writing model was used). Review the findings below and "
                           "pick the angle competitors are not using." if competitors else "",
            "angles": angles, "quick_wins": wins,
        }
    notes = []
    if not competitors:
        notes.append("No competitor pages could be read, so there is nothing to compare against.")
    return {"topics": topics, "offer_gaps": offer_gaps, "advantages": advantages, "channel_gaps": channel_gaps,
            "keyword_gaps": keyword_gaps, "recommendation": recommendation, "compared_with": total, "notes": notes,
            "evidence_attached": evidence_count}
