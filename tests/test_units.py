"""Unit tests for the deterministic building blocks."""
from __future__ import annotations

import pytest

from adstrategist.sources.keyword_csv import _number, parse_keyword_csv
from adstrategist.sources.search import parse_results, region_code
from adstrategist.steps import resolve
from adstrategist.steps.budget import allocate, platform_weights, project
from adstrategist.steps.keywords import classify_intent, on_topic
from adstrategist.textutil import registrable_domain, top_phrases, truncate, truncate_sentence
from adstrategist.web_fetch import parse_html

from .conftest import OWN_HTML, SEARCH_HTML


def test_truncate_never_exceeds_limit_and_cuts_at_words():
    assert truncate("Organic tea delivered monthly", 15) == "Organic tea"
    assert len(truncate("x" * 200, 30)) <= 30


def test_truncate_sentence_keeps_whole_sentences():
    text = "We ship organic tea every month. Pause any time. Cancel whenever you like, no questions asked."
    assert truncate_sentence(text, 60) == "We ship organic tea every month. Pause any time."


@pytest.mark.parametrize("url,expected", [
    ("https://www.shop.example.co.uk/a", "example.co.uk"), ("sub.example.com", "example.com"),
    ("https://example.de/", "example.de"),
])
def test_registrable_domain(url, expected):
    assert registrable_domain(url) == expected


def test_top_phrases_prefers_what_the_body_repeats_over_a_one_off_slogan():
    body = "project management for teams. " * 5 + "Loved by everyone everywhere."
    phrases = top_phrases([("Loved by everyone everywhere", 2.0)], exclude=set(), body=body)
    assert phrases[0] == "project management"


def test_parse_html_reads_facts_and_ignores_menu_items():
    page = parse_html("https://greenleaf.example/", OWN_HTML)
    assert page.title.startswith("GreenLeaf") and page.site_name == "GreenLeaf" and page.lang == "en"
    assert page.locality == "Berlin"
    assert "meta" in page.trackers and "tiktok" not in page.trackers
    assert {"free_shipping", "reviews", "pricing_shown"} <= set(page.offers)
    assert page.prices == ["€12.90"]
    assert page.images == ["https://greenleaf.example/img/hero.jpg"]
    assert not any("Pricing and plans" in b for b in page.bullets)  # nav entry, not a selling point
    assert any("organic loose leaf" in b for b in page.bullets)


def test_search_results_skip_ads_and_unwrap_redirects():
    urls = [r["url"] for r in parse_results(SEARCH_HTML)]
    assert urls == ["https://teabox.example/", "https://leafclub.example/", "https://www.wikipedia.org/wiki/Tea"]
    assert region_code("GB", "en") == "uk-en" and region_code("DE", "de") == "de-de"


@pytest.mark.parametrize("raw,value", [("1,300", 1300.0), ("1.300", 1300.0), ("0,85", 0.85), ("0.85", 0.85),
                                       ("1K – 10K", 1000.0), ("", None), ("12,345.67", 12345.67)])
def test_number_formats(raw, value):
    assert _number(raw) == value


def test_keyword_planner_export_utf16_with_preamble():
    text = ("Keyword Stats 2026-11-01 at 10_00_00\nNovember 1, 2025 - October 31, 2026\n"
            "Keyword\tCurrency\tAvg. monthly searches\tCompetition\tTop of page bid (low range)\tTop of page bid (high range)\n"
            "Organic Tea Subscription\tEUR\t1,300\tHigh\t0.45\t1.60\n"
            "tea box\tEUR\t880\tMedium\t\t0.90\n")
    data = parse_keyword_csv(text.encode("utf-16"))
    assert data["organic tea subscription"] == {"volume": 1300.0, "cpc_low": 0.45, "cpc_high": 1.6, "competition": "High"}
    assert data["tea box"] == {"volume": 880.0, "cpc_high": 0.9, "competition": "Medium"}


def test_simple_csv_and_missing_keyword_column():
    assert parse_keyword_csv(b"keyword,volume,cpc\ngreen tea,500,0.7\n")["green tea"] == {"volume": 500.0, "cpc_high": 0.7}
    with pytest.raises(ValueError):
        parse_keyword_csv(b"term;count\nx;1\n")


def test_intent_classification():
    brands = {"teabox"}
    assert classify_intent("buy organic tea", brands) == "transactional"
    assert classify_intent("bio tee kaufen", brands) == "transactional"
    assert classify_intent("best tea subscription", brands) == "commercial"
    assert classify_intent("how to brew green tea", brands) == "informational"
    assert classify_intent("teabox login", brands) == "navigational"
    assert classify_intent("tea berlin", brands, location="Berlin") == "transactional"


def test_on_topic_rejects_autocomplete_drift():
    vocabulary = {"organ", "subsc", "month", "blend", "price"}
    seed = {"organ", "subsc"}
    assert on_topic("organic tea subscription price", seed, vocabulary | {"tea"}, "")
    assert not on_topic("organic chemistry tutor", seed, vocabulary, "")
    assert not on_topic("benefits of chia seeds", {"benef"}, vocabulary, "")
    assert on_topic("anything goes", set(), set(), "")  # no vocabulary: nothing to judge by


def test_budget_weights_react_to_research_signals():
    fit = {"google": 0.4, "meta": 0.4, "tiktok": 0.2}
    weights, reasons = platform_weights(["google", "meta", "tiktok"], fit,
                                        {"buyer_keywords": 12, "competitor_platforms": {"meta": 2, "tiktok": 0},
                                         "competitors_readable": 2, "video": "no"})
    assert abs(sum(weights.values()) - 1) < 1e-9
    assert weights["google"] > 0.4 and weights["tiktok"] < 0.2 and len(reasons) == 3


def test_allocation_concentrates_small_budgets():
    weights = {"google": 0.5, "meta": 0.35, "tiktok": 0.15}
    minimums = {"google": 10, "meta": 5, "tiktok": 20}
    amounts, dropped = allocate(600, weights, minimums)
    assert dropped == ["tiktok"] and set(amounts) == {"google", "meta"}
    assert abs(sum(amounts.values()) - 600) < 1e-6
    amounts, dropped = allocate(150, weights, minimums)
    assert list(amounts) == ["google"] and amounts["google"] == 150  # one platform is always kept
    amounts, dropped = allocate(5000, weights, minimums)
    assert not dropped and len(amounts) == 3


def test_projection_arithmetic():
    out = project(1000, {"cvr": [0.02, 0.05]}, [1.0, 2.0])
    assert out["clicks"] == [500, 1000]
    assert out["conversions"] == [10.0, 50.0]
    assert out["cost_per_conversion"] == [20.0, 100.0]


def test_step_dependencies_are_added_in_pipeline_order():
    assert resolve(["campaigns"]) == ["brief", "keywords", "campaigns", "summary"]
    assert resolve(["ad_intel"]) == ["brief", "competitors", "ad_intel", "summary"]
    assert resolve([])[0] == "brief" and len(resolve([])) == 10
    with pytest.raises(ValueError):
        resolve(["nonsense"])


def test_claim_guard_blocks_invented_offers_numbers_and_superlatives():
    from adstrategist.claims import unsupported
    page = "GreenLeaf organic tea. Free shipping on every box. Rated by 2316 customers. From €12.90."
    offers = ["free_shipping", "reviews"]
    assert unsupported("Free shipping on every box", offers, page) == ""
    assert unsupported("Rated by 2316 customers", offers, page) == ""
    assert "10" in unsupported("Jetzt 10% Rabatt sichern", offers, page)
    # the page mentions discounts and the digits "20" (inside 2026), but never "20%"
    sale_page = page + " Sale: Rabatt auf ausgewählte Tees bis 2026."
    assert "20%" in unsupported("Jetzt 20% Rabatt auf Bestseller", offers + ["discount"], sale_page)
    assert unsupported("Rabatt auf ausgewählte Tees", offers + ["discount"], sale_page) == ""
    assert unsupported("Ab 12.90 pro Box", offers, page) == "" and unsupported("Ab 2.90 pro Box", offers, page)
    assert "rabatt" in unsupported("Jetzt Rabatt sichern", offers, page).lower()
    assert unsupported("Der führende Anbieter für Tee", offers, page)
    assert unsupported("Gesundheitliche Vorteile", offers, page)
    assert unsupported("Try it free for 30 days", ["reviews"], page)
    assert unsupported("Money-back guarantee", offers, page)
    assert unsupported("Organic tea delivered monthly", [], page) == ""


def test_figures_compare_across_number_formats():
    from adstrategist.claims import missing_numbers
    source = "Clicks 445 to 1727. Monthly 327.27 EUR. Share 55%. Since 2026."
    assert missing_numbers("Zwischen 445 und 1.727 Klicks, 327,27 EUR, 55 %", source) == []
    assert missing_numbers("1,727 clicks", source) == []
    assert missing_numbers("a budget of 900", "monthly: 900.0, daily: 29.61") == []
    assert missing_numbers("a budget of 900.50", "monthly: 900.0") == ["900.50"]
    assert missing_numbers("60% on Google, 20 clicks", source) == ["60%", "20"]
