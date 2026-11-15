"""Step 7: budget plan. Pure arithmetic over stated benchmarks and any real bids, no model involved."""
from __future__ import annotations

import json
import math

from ..config import PACKAGE_DIR
from ..context import Context

DAYS = 30.4


def load_benchmarks() -> dict:
    return json.loads((PACKAGE_DIR / "data" / "benchmarks.json").read_text(encoding="utf-8"))


def _round_money(value: float) -> int:
    step = 10 if value < 500 else 50
    return int(max(step, round(value / step) * step))


def platform_weights(platforms: list[str], fit: dict, signals: dict) -> tuple[dict[str, float], list[str]]:
    """Start from the business-type prior, then nudge it with what the research found."""
    weights = {p: fit[p] for p in platforms}
    reasons = []
    buyers = signals.get("buyer_keywords")
    if "google" in weights and buyers is not None:
        if buyers >= 8:
            weights["google"] *= 1.2
            reasons.append(f"{buyers} buyer-intent keywords found, so search demand exists: Google weighted up.")
        elif buyers < 3:
            weights["google"] *= 0.7
            reasons.append("Few buyer-intent keywords found, so search demand looks thin: Google weighted down.")
    readable = signals.get("competitors_readable") or 0
    for platform, count in (signals.get("competitor_platforms") or {}).items():
        if platform in weights and readable and count / readable >= 0.5:
            weights[platform] *= 1.15
            reasons.append(f"{count} of {readable} competitors are set up on {platform}: a validated channel.")
    if "tiktok" in weights and signals.get("video") == "no":
        weights["tiktok"] *= 0.6
        reasons.append("Video is switched off and TikTok needs video creative: TikTok weighted down.")
    total = sum(weights.values()) or 1.0
    return {p: w / total for p, w in weights.items()}, reasons


def allocate(budget: float, weights: dict[str, float], minimums: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Concentrate rather than spread thin: drop the weakest platform until every one left clears its minimum."""
    active = dict(weights)
    dropped = []
    while len(active) > 1:
        total = sum(active.values())
        short = [p for p in active if budget * active[p] / total / DAYS < minimums[p]]
        if not short:
            break
        weakest = min(short, key=lambda p: active[p])
        dropped.append(weakest)
        del active[weakest]
    total = sum(active.values()) or 1.0
    return {p: budget * w / total for p, w in active.items()}, dropped


def project(amount: float, bench: dict, cpc: list[float]) -> dict:
    clicks = [amount / cpc[1], amount / cpc[0]]
    conversions = [clicks[0] * bench["cvr"][0], clicks[1] * bench["cvr"][1]]
    cpa = [amount / conversions[1], amount / conversions[0]]
    return {"clicks": [round(c) for c in clicks], "conversions": [round(c, 1) for c in conversions],
            "cost_per_conversion": [round(c, 2) for c in cpa]}


def run(ctx: Context) -> dict:
    cfg = ctx.config
    benchmarks = load_benchmarks()
    business_type = ctx.brief.get("business_type", "other")
    table = benchmarks["business_types"].get(business_type, benchmarks["business_types"]["other"])
    platforms = [p for p in cfg.platforms if p in benchmarks["platforms"]]
    minimums = {p: benchmarks["platforms"][p]["min_daily"] for p in platforms}

    keywords = ctx.results.get("keywords", {}).get("keywords")
    intel = ctx.results.get("ad_intel", {})
    signals = {
        "buyer_keywords": None if keywords is None else
        sum(1 for k in keywords if k["intent"] in ("transactional", "commercial")),
        "competitor_platforms": intel.get("summary"), "competitors_readable": intel.get("readable"),
        "video": cfg.video,
    }
    weights, reasons = platform_weights(platforms, table["fit"], signals)

    # Real Google bids beat the benchmark whenever the user supplied a Keyword Planner export.
    cpcs = {p: list(table[p]["cpc"]) for p in platforms}
    cpc_source = {p: "benchmark default" for p in platforms}
    priced = [k for k in (keywords or []) if k.get("volume") and (k.get("cpc_low") or k.get("cpc_high"))
              and k["intent"] in ("transactional", "commercial")]
    if priced and "google" in cpcs:
        weight = sum(k["volume"] for k in priced)
        low = sum((k.get("cpc_low") or k["cpc_high"]) * k["volume"] for k in priced) / weight
        high = sum((k.get("cpc_high") or k["cpc_low"]) * k["volume"] for k in priced) / weight
        cpcs["google"] = [round(low, 2), round(max(high, low * 1.05), 2)]
        cpc_source["google"] = f"Keyword Planner bids for {len(priced)} keywords"

    def mid_cpc(platform: str) -> float:
        return math.sqrt(cpcs[platform][0] * cpcs[platform][1])

    order = sorted(platforms, key=lambda p: -weights[p])
    tiers = []
    for name, count, clicks, purpose in (
        ("Test", 1, 300, "Enough clicks on the strongest channel to see whether the offer converts."),
        ("Steady", 2, 1000, "Two channels with enough volume to optimise week by week."),
        ("Scale", len(order), 3000, "Every selected channel funded properly."),
    ):
        chosen = order[: max(1, min(count, len(order)))]
        share = sum(weights[p] for p in chosen) or 1.0
        needed = sum(clicks * weights[p] / share * mid_cpc(p) for p in chosen)
        floor = sum(minimums[p] * DAYS for p in chosen)
        tiers.append({"name": name, "monthly": _round_money(max(needed, floor)), "platforms": chosen,
                      "purpose": purpose})

    proposed = cfg.budget is None
    budget = float(tiers[0]["monthly"] if proposed else cfg.budget)
    amounts, dropped = allocate(budget, weights, minimums)

    plan = []
    for platform in sorted(amounts, key=lambda p: -amounts[p]):
        amount = amounts[platform]
        row = {"platform": platform, "label": benchmarks["platforms"][platform]["label"],
               "share": round(amount / budget, 3), "monthly": round(amount, 2), "daily": round(amount / DAYS, 2),
               "cpc": cpcs[platform], "cpc_source": cpc_source[platform], "cvr": table[platform]["cvr"],
               **project(amount, table[platform], cpcs[platform])}
        low, high = row["cost_per_conversion"]
        row["value_score"] = round(math.sqrt(low * high), 2)
        if cfg.customer_value:
            row["payback"] = ("pays back across the whole range" if high <= cfg.customer_value else
                              "can pay back, depends on execution" if low <= cfg.customer_value else
                              "unlikely to pay back at this customer value")
        plan.append(row)
    best = min(plan, key=lambda r: r["value_score"])["platform"] if plan else ""

    warnings = []
    for platform in dropped:
        warnings.append(f"{benchmarks['platforms'][platform]['label']} is left out: its share of this budget "
                        f"would fall below a workable {minimums[platform]} per day. Add it once the budget grows.")
    if plan and len(plan) == 1 and plan[0]["daily"] < minimums[plan[0]["platform"]]:
        warnings.append(f"{budget:.0f} per month is below the practical minimum for {plan[0]['label']} "
                        f"({minimums[plan[0]['platform']]} per day). Expect slow, noisy results.")
    for row in plan:
        if row["platform"] == "meta" and row["conversions"][1] / 4.3 < 50:
            warnings.append("Meta's delivery settles after roughly 50 optimisation events per ad set per week. "
                            "This budget will not reach that on purchases or leads, so run one ad set and "
                            "optimise for an earlier event (for example landing page views) at first.")
    phases = [
        "Weeks 1 to 3: launch paused campaigns one at a time, keep budgets fixed, change nothing for the first week.",
        "Week 4: stop ads and keywords with spend but no conversions, move their budget to the best performer.",
        "After that: raise budgets by at most 20% at a time and only on campaigns that hit your target cost.",
    ]
    return {
        "currency": cfg.currency, "business_type": business_type, "monthly_budget": budget,
        "budget_was_proposed": proposed, "tiers": tiers, "plan": plan, "best_value": best,
        "not_funded": dropped, "weighting_reasons": reasons, "customer_value": cfg.customer_value,
        "warnings": warnings, "phases": phases,
        "totals": {
            "clicks": [sum(r["clicks"][0] for r in plan), sum(r["clicks"][1] for r in plan)],
            "conversions": [round(sum(r["conversions"][0] for r in plan), 1),
                            round(sum(r["conversions"][1] for r in plan), 1)],
        },
        "notes": [
            "Clicks and conversions are ranges from benchmark cost per click and conversion rates for this "
            "business type. They are planning assumptions, not forecasts. Edit adstrategist/data/benchmarks.json "
            "or replace them with your own account data after the first weeks.",
            *(benchmarks["platforms"][p]["min_note"] for p in platforms),
        ],
    }
