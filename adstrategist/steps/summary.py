"""Final step: an executive summary. This is where a stronger paid model helps most."""
from __future__ import annotations

import json

from ..claims import missing_numbers
from ..context import Context
from ..textutil import truncate


def facts(ctx: Context) -> dict:
    r = ctx.results
    brief = ctx.brief
    out: dict = {"product": f"{brief.get('brand')}: {brief.get('one_liner')}",
                 "business_type": brief.get("business_type")}
    if "competitors" in r:
        out["competitors"] = [f"{c['name']}: {truncate(c['positioning'], 120)}"
                              for c in r["competitors"]["competitors"] if not c.get("error")][:5]
    if "gaps" in r:
        g = r["gaps"]
        out["gaps"] = {"topics_we_miss": [t["topic"] for t in g["topics"][:5]],
                       "offers_we_miss": [o["offer"] for o in g["offer_gaps"]],
                       "channels": [c["text"] for c in g["channel_gaps"]],
                       "recommended_positioning": g["recommendation"].get("positioning", "")}
    if "keywords" in r:
        k = r["keywords"]
        out["keywords"] = {"count": len(k["keywords"]), "strongest_groups": [c["name"] for c in k["clusters"][:4]],
                           "real_volumes_available": k["has_volumes"]}
    if "budget" in r:
        b = r["budget"]
        out["budget"] = {"monthly": b["monthly_budget"], "currency": b["currency"],
                         "budget_source": "proposed by this tool" if b["budget_was_proposed"] else "set by the business owner",
                         "split": {row["label"]: {"monthly": row["monthly"], "share": f"{row['share'] * 100:.0f}%"}
                                   for row in b["plan"]},
                         "expected_clicks_range": b["totals"]["clicks"],
                         "expected_conversions_range": b["totals"]["conversions"],
                         "best_value_platform": b["best_value"], "warnings": b["warnings"][:2]}
    return out


def next_steps(ctx: Context) -> list[str]:
    r = ctx.results
    steps = []
    if "keywords" in r and not r["keywords"]["has_volumes"]:
        steps.append("Paste keywords_for_keyword_planner.txt into Google Keyword Planner, download the CSV and run "
                     "again with it to replace benchmark assumptions with real volumes and bids.")
    if "campaigns" in r:
        for campaign in r["campaigns"]["campaigns"]:
            if campaign.get("pixel_note"):
                steps.append(campaign["pixel_note"])
        steps.append("Review every ad line, then import or copy the campaign files. Everything starts paused.")
    if "ad_intel" in r:
        steps.append("Open the ad library links for your top competitors and note which ads have run longest.")
    if "budget" in r:
        steps.append("After three weeks, replace the benchmark figures with your own cost per click and conversion "
                     "rate and re-check the split.")
    return steps


def run(ctx: Context) -> dict:
    data = facts(ctx)
    text = None
    figures_corrected = False
    if ctx.llm.enabled("writing"):
        given = json.dumps(data, ensure_ascii=False, indent=1)
        brief_prompt = (
            f"Write an executive summary of this marketing research in {ctx.language_name} for the business "
            "owner. 150 to 220 words, plain paragraphs, no headings, no bullet lists. Start directly with the "
            "findings, without an introduction. Cover: where the product "
            "stands against competitors, the biggest opportunity, and what to do with the budget. Use only the "
            "figures given below and present ranges as ranges. If real search volumes are not available, say "
            "that the click and conversion ranges are benchmark assumptions. Do not calculate new figures: "
            "copy figures exactly as given.\n\n" + given)
        text = ctx.llm.ask_text("executive summary", brief_prompt, role="writing", max_tokens=600)
        # The summary is the part people read first, and small models round and recompute. Every figure
        # in it must be one it was given: one corrected attempt, then the rule-based summary.
        wrong = missing_numbers(text, given) if text else []
        if wrong:
            figures_corrected = True
            ctx.emit("info", f"Summary: the model used figures it was not given ({', '.join(wrong[:4])}). "
                             "Asking it to correct them.")
            text = ctx.llm.ask_text(
                "executive summary, corrected",
                brief_prompt + f"\n\nYour previous draft used figures that are not in the data: "
                               f"{', '.join(wrong[:8])}. Write it again and copy every figure exactly, or leave it out.",
                role="writing", max_tokens=600)
            if text and missing_numbers(text, given):
                ctx.rejected_claims.append({"where": "executive summary", "text": text[:160],
                                            "reason": "it kept using figures that are not in the data"})
                text = None
    if text:  # small models open with "Here is a summary:" whatever they are told
        first, _, rest = text.partition("\n")
        if first.rstrip().endswith(":") and len(first) < 140 and rest.strip():
            text = rest.strip()
    written_by_model = bool(text)
    if not text:
        parts = [f"{data['product']}"]
        if data.get("competitors"):
            parts.append(f"{len(data['competitors'])} competitors were analysed.")
        if "budget" in data:
            b = data["budget"]
            parts.append(
                f"{'Proposed' if b['budget_source'].startswith('proposed') else 'Planned'} budget: {b['monthly']:.0f} {b['currency']} "
                f"per month across {', '.join(f"{k} ({v['share']})" for k, v in b['split'].items())}. Benchmark range: {b['expected_clicks_range'][0]} to "
                f"{b['expected_clicks_range'][1]} clicks and {b['expected_conversions_range'][0]} to "
                f"{b['expected_conversions_range'][1]} conversions per month.")
        parts.append("This summary is rule based because no writing model was used.")
        text = " ".join(parts)
    return {"text": text, "written_by_model": written_by_model, "figures_corrected": figures_corrected,
            "next_steps": next_steps(ctx)}
