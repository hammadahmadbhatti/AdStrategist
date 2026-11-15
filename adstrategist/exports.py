"""Campaign export files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

GOOGLE_COLUMNS = (
    ["Campaign", "Campaign Type", "Networks", "Budget", "Budget type", "Bid Strategy Type", "Languages", "Location",
     "Campaign Status", "Ad Group", "Ad Group Status", "Keyword", "Criterion Type", "Ad type"]
    + [f"Headline {i}" for i in range(1, 16)] + [f"Description {i}" for i in range(1, 5)]
    + ["Path 1", "Path 2", "Final URL", "Status", "Link Text", "Callout text"]
)


def google_rows(campaign: dict) -> list[dict]:
    """Rows in the layout Google Ads Editor imports: one entity per row, identified by its filled columns."""
    name = campaign["name"]
    rows = [{
        "Campaign": name, "Campaign Type": "Search", "Networks": "Google search",
        "Budget": campaign["daily_budget"] if campaign["daily_budget"] is not None else "",
        "Budget type": "Daily", "Bid Strategy Type": campaign["bid_strategy"],
        "Languages": campaign["language"], "Campaign Status": "Paused",
    }, {"Campaign": name, "Location": campaign["location"]}]
    for group in campaign["ad_groups"]:
        rows.append({"Campaign": name, "Ad Group": group["name"], "Ad Group Status": "Enabled"})
        for keyword in group["keywords"]:
            rows.append({"Campaign": name, "Ad Group": group["name"], "Keyword": keyword["text"],
                         "Criterion Type": keyword["match"]})
        ad = group["ad"]
        row = {"Campaign": name, "Ad Group": group["name"], "Ad type": "Responsive search ad",
               "Path 1": ad["path1"], "Path 2": ad["path2"], "Final URL": ad["final_url"], "Status": "Enabled"}
        row.update({f"Headline {i}": h for i, h in enumerate(ad["headlines"], 1)})
        row.update({f"Description {i}": d for i, d in enumerate(ad["descriptions"], 1)})
        rows.append(row)
    for term in campaign["negative_keywords"]:
        rows.append({"Campaign": name, "Keyword": term, "Criterion Type": "Negative Phrase"})
    for link in campaign["sitelinks"]:
        rows.append({"Campaign": name, "Link Text": link["text"], "Final URL": link["url"]})
    for callout in campaign["callouts"]:
        rows.append({"Campaign": name, "Callout text": callout})
    return rows


def meta_rows(campaign: dict) -> list[dict]:
    rows = []
    for ad_set in campaign["ad_sets"]:
        for ad in campaign["ads"]:
            rows.append({
                "Campaign Name": campaign["name"], "Campaign Status": "PAUSED",
                "Campaign Objective": campaign["objective"], "Buying Type": "AUCTION",
                "Campaign Daily Budget": campaign["daily_budget"] if campaign["daily_budget"] is not None else "",
                "Ad Set Name": ad_set["name"], "Ad Set Run Status": "PAUSED", "Countries": ad_set["countries"],
                "Audience": ad_set["audience"], "Placements": ad_set["placements"],
                "Ad Name": ad["name"], "Ad Status": "PAUSED", "Body": ad["primary_text"], "Title": ad["headline"],
                "Link Description": ad["description"], "Link": ad["link"], "Call to Action": ad["call_to_action"],
                "Creative Brief": ad["visual"],
            })
    return rows


def tiktok_rows(campaign: dict) -> list[dict]:
    group = campaign["ad_group"]
    return [{
        "Campaign Name": campaign["name"], "Objective": campaign["objective"], "Status": "Paused",
        "Ad Group Name": group["name"], "Placement": group["placement"], "Location": group["location"],
        "Age": group["age"], "Bidding": group["bidding"],
        "Daily Budget": campaign["daily_budget"] if campaign["daily_budget"] is not None else "",
        "Ad Name": ad["name"], "Ad Text": ad["ad_text"], "Call to Action": ad["call_to_action"],
        "Landing Page URL": ad["link"], "Hook": ad["hook"], "Script": " / ".join(ad["script"]),
        "Hashtags": " ".join("#" + t for t in ad["hashtags"]),
    } for ad in campaign["ads"]]


def _write(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_campaign_files(run_dir: Path, campaigns: list[dict], currency: str) -> list[str]:
    folder = run_dir / "campaigns"
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for campaign in campaigns:
        platform = campaign["platform"]
        if platform == "google":
            _write(folder / "google_ads_editor.csv", google_rows(campaign), GOOGLE_COLUMNS)
            files.append("campaigns/google_ads_editor.csv")
        elif platform == "meta":
            _write(folder / "meta_ads_sheet.csv", meta_rows(campaign))
            files.append("campaigns/meta_ads_sheet.csv")
        elif platform == "tiktok":
            _write(folder / "tiktok_ads_sheet.csv", tiktok_rows(campaign))
            files.append("campaigns/tiktok_ads_sheet.csv")
    (folder / "campaigns.json").write_text(
        json.dumps({"currency": currency, "campaigns": campaigns}, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append("campaigns/campaigns.json")
    return files
