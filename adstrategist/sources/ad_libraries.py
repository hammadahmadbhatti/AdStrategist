"""Public ad transparency libraries: where competitor ads can be inspected for free."""
from __future__ import annotations

from urllib.parse import quote, urlencode

import httpx

META_API = "https://graph.facebook.com/v21.0/ads_archive"


def library_links(name: str, domain: str, country: str) -> dict[str, str]:
    """Deep links a person can open. Spend is not public on any of them."""
    meta_query = urlencode({"active_status": "active", "ad_type": "all", "country": country.upper(),
                            "q": name, "search_type": "keyword_unordered"})
    return {
        "meta": f"https://www.facebook.com/ads/library/?{meta_query}",
        "google": f"https://adstransparency.google.com/?region={country.upper()}&domain={quote(domain)}",
        "tiktok": "https://library.tiktok.com/ads",
    }


def meta_ads(client: httpx.Client, token: str, search_terms: str, country: str, limit: int = 25) -> dict:
    """Active ads from the Meta Ad Library API. Needs an access token from a verified Meta developer account.

    The API returns every ad delivered in the EU and UK, plus political ads worldwide.
    """
    params = {
        "access_token": token,
        "search_terms": search_terms,
        "ad_reached_countries": f'["{country.upper()}"]',
        "ad_active_status": "ACTIVE",
        "ad_type": "ALL",
        "limit": limit,
        "fields": "page_name,ad_creative_bodies,ad_creative_link_titles,ad_delivery_start_time,publisher_platforms",
    }
    try:
        resp = client.get(META_API, params=params, timeout=20.0)
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": f"Meta Ad Library request failed ({type(exc).__name__})", "ads": []}
    if resp.status_code != 200 or "error" in data:
        message = (data.get("error") or {}).get("message", f"HTTP {resp.status_code}")
        return {"error": f"Meta Ad Library: {message}", "ads": []}
    ads = []
    for row in data.get("data", []):
        ads.append({
            "page": row.get("page_name", ""),
            "text": (row.get("ad_creative_bodies") or [""])[0][:300],
            "headline": (row.get("ad_creative_link_titles") or [""])[0][:120],
            "started": (row.get("ad_delivery_start_time") or "")[:10],
            "platforms": row.get("publisher_platforms") or [],
        })
    ads.sort(key=lambda a: a["started"] or "9999")  # longest running first: usually the ones that work
    return {"error": "", "ads": ads}
