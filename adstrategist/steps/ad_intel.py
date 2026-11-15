"""Step 3: which ad platforms competitors are set up on, and where to inspect their live ads."""
from __future__ import annotations

from ..context import Context
from ..sources.ad_libraries import library_links, meta_ads
from ..web_fetch import TRACKER_LABELS

AD_TAGS = {"meta": "meta", "google_ads": "google", "tiktok": "tiktok"}  # tracker -> platform id


def run(ctx: Context) -> dict:
    competitors = ctx.results.get("competitors", {}).get("competitors", [])
    country = ctx.config.country
    token = ctx.settings.meta_access_token
    rows = []
    for comp in competitors:
        trackers = comp.get("trackers", [])
        row = {
            "name": comp["name"], "domain": comp["domain"],
            "platforms": [AD_TAGS[t] for t in trackers if t in AD_TAGS],
            "tags": [TRACKER_LABELS.get(t, t) for t in trackers],
            "tag_manager": "google_tag_manager" in trackers,
            "links": library_links(comp["name"], comp["domain"], country),
            "readable": not comp.get("error"),
            "meta_ads": None,
        }
        if token:
            ctx.emit("info", f"Meta Ad Library: {comp['name']}")
            found = meta_ads(ctx.fetcher.client, token, comp["name"], country)
            row["meta_ads"] = {"count": len(found["ads"]), "error": found["error"],
                               "longest_running": found["ads"][:3]}
        rows.append(row)

    readable = [r for r in rows if r["readable"]]
    summary = {p: sum(1 for r in readable if p in r["platforms"]) for p in ("google", "meta", "tiktok")}
    notes = [
        "A tag on the site shows a competitor is set up to advertise there. It does not prove ads are live: "
        "open the library links to see the actual ads.",
        "No platform publishes competitor ad spend. Any spend figure you see elsewhere is an estimate.",
    ]
    if any(r["tag_manager"] for r in readable):
        notes.append("Sites using Google Tag Manager can load ad pixels that are invisible in the page source, "
                     "so their footprint may be larger than shown.")
    if not token:
        notes.append("Set META_ACCESS_TOKEN to list competitors' active Meta ads here instead of opening links.")
    return {"competitors": rows, "summary": summary, "readable": len(readable),
            "own_platforms": [AD_TAGS[t] for t in ctx.brief.get("trackers", []) if t in AD_TAGS], "notes": notes}
