"""Step 6: traffic potential. Only produces numbers when real search volumes were supplied."""
from __future__ import annotations

from ..context import Context

# Stated assumptions, shown next to every figure they produce.
PAID_IMPRESSION_SHARE = (0.20, 0.50)   # share of searches a new advertiser's ads appear for
PAID_CTR = (0.03, 0.07)                # search ad click rate
ORGANIC_TOP3_CTR = (0.08, 0.20)        # click rate when ranking in the top three organic results


def run(ctx: Context) -> dict:
    data = ctx.results.get("keywords", {})
    keywords = data.get("keywords", [])
    clusters = data.get("clusters", [])
    if not data.get("has_volumes"):
        ranking = [{"cluster": c["name"], "keywords": c["count"], "interest": c["interest"],
                    "main_intent": c["main_intent"]} for c in clusters]
        return {
            "available": False, "ranking": ranking,
            "notes": ["Traffic cannot be estimated without search volumes, and a model must not guess them. "
                      "The ranking shows which keyword groups look strongest relative to each other.",
                      "To get click and cost estimates, add a Keyword Planner CSV (see the keyword section)."],
        }

    with_volume = [k for k in keywords if k.get("volume")]
    buyers = [k for k in with_volume if k["intent"] in ("transactional", "commercial")]
    total = sum(k["volume"] for k in with_volume)
    buyer_volume = sum(k["volume"] for k in buyers)
    paid = [round(buyer_volume * PAID_IMPRESSION_SHARE[i] * PAID_CTR[i]) for i in (0, 1)]
    organic = [round(total * ORGANIC_TOP3_CTR[i]) for i in (0, 1)]

    priced = [k for k in buyers if k.get("cpc_low") or k.get("cpc_high")]
    cost = None
    if priced:
        weight = sum(k["volume"] for k in priced)
        low = sum((k.get("cpc_low") or k.get("cpc_high")) * k["volume"] for k in priced) / weight
        high = sum((k.get("cpc_high") or k.get("cpc_low")) * k["volume"] for k in priced) / weight
        cost = {"cpc": [round(low, 2), round(high, 2)], "monthly": [round(paid[0] * low), round(paid[1] * high)]}

    by_cluster = []
    for cluster in clusters:
        members = [k for k in with_volume if k["cluster"] == cluster["name"]]
        if members:
            by_cluster.append({"cluster": cluster["name"], "volume": round(sum(k["volume"] for k in members)),
                               "buyer_volume": round(sum(k["volume"] for k in members if k in buyers)),
                               "keywords": len(members)})
    by_cluster.sort(key=lambda c: -c["volume"])
    return {
        "available": True, "keywords_with_volume": len(with_volume), "monthly_searches": round(total),
        "buyer_intent_searches": round(buyer_volume), "paid_clicks": paid, "organic_clicks_ceiling": organic,
        "paid_cost": cost, "by_cluster": by_cluster,
        "assumptions": {
            "paid": f"Ads shown for {int(PAID_IMPRESSION_SHARE[0]*100)} to {int(PAID_IMPRESSION_SHARE[1]*100)}% of "
                    f"buyer-intent searches, clicked {int(PAID_CTR[0]*100)} to {int(PAID_CTR[1]*100)}% of the time.",
            "organic": f"A ceiling, not a forecast: {int(ORGANIC_TOP3_CTR[0]*100)} to {int(ORGANIC_TOP3_CTR[1]*100)}% "
                       "of all searches if you ranked in the top three for every keyword.",
        },
        "notes": ["Volumes and bids come from your Keyword Planner export. Click figures apply the stated "
                  "assumptions to them."],
    }
