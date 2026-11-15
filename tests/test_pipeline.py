"""End-to-end runs against the fake web: no network, no model unless a fake one is injected."""
from __future__ import annotations

import csv
import json
import shutil

import pytest
from fastapi.testclient import TestClient

from adstrategist import pipeline
from adstrategist.context import RunConfig
from adstrategist.pipeline import run_pipeline, validate
from adstrategist.steps.campaigns import LIMITS

from .conftest import FakeProvider

URL = "https://greenleaf.example/"


def run(settings, fetcher, tmp_path, **overrides):
    config = RunConfig(url=URL, country="US", currency="USD", video="no", **overrides)
    return run_pipeline(config, settings, fetcher=fetcher, run_dir=tmp_path / "run")


def test_full_offline_run_produces_every_section_and_file(settings, fetcher, tmp_path):
    result = run(settings, fetcher, tmp_path, budget=900.0, customer_value=60.0)
    r, run_dir = result["results"], tmp_path / "run"
    assert result["errors"] == {} and list(r) == result["meta"]["steps"]

    brief = r["brief"]
    assert brief["brand"] == "GreenLeaf" and brief["business_type"] == "ecommerce" and brief["location"] == "Berlin"
    assert brief["category"] == "Organic Tea Subscription" and "organic tea subscription" in brief["seed_keywords"]
    assert not brief["written_by_model"] and brief["warnings"]

    names = {c["domain"] for c in r["competitors"]["competitors"]}
    assert names == {"teabox.example", "leafclub.example"}  # wikipedia and the ad are excluded
    assert r["ad_intel"]["summary"] == {"google": 1, "meta": 1, "tiktok": 1}
    assert r["ad_intel"]["own_platforms"] == ["meta"]

    keywords = {k["keyword"]: k for k in r["keywords"]["keywords"]}
    assert "organic tea subscription box" in keywords
    assert keywords["organic tea subscription price"]["intent"] == "transactional"
    assert "organic tea subscription jobs" not in keywords and "jobs" in r["keywords"]["negative_keywords"]
    assert "free" in r["keywords"]["negative_keywords"]  # the page offers no free trial
    assert not r["keywords"]["has_volumes"] and not r["traffic"]["available"]

    gaps = r["gaps"]
    assert any("gift boxes" in t["topic"] for t in gaps["topics"])
    assert {g["offer"] for g in gaps["offer_gaps"]} == {"guarantee", "free trial"}
    assert any(a["offer"] == "free shipping" for a in gaps["advantages"])
    assert any(g["platform"] == "tiktok" and g["kind"] == "behind" for g in gaps["channel_gaps"])

    budget = r["budget"]
    assert budget["monthly_budget"] == 900 and not budget["budget_was_proposed"]
    assert abs(sum(row["monthly"] for row in budget["plan"]) - 900) < 0.05
    assert "tiktok" in budget["not_funded"] and all("payback" in row for row in budget["plan"])

    for name in ("report.md", "report.html", "result.json", "keywords.csv", "keywords_for_keyword_planner.txt",
                 "campaigns/google_ads_editor.csv", "campaigns/meta_ads_sheet.csv", "campaigns/tiktok_ads_sheet.csv"):
        assert (run_dir / name).stat().st_size > 0, name
    report = (run_dir / "report.md").read_text()
    for heading in ("## Summary", "## Product brief", "## Competitors", "## Keyword research", "## Market gaps",
                    "## Budget plan", "## Ad campaigns", "## Video ad"):
        assert heading in report
    assert "—" not in report
    assert "<script" not in (run_dir / "report.html").read_text()
    assert json.loads((run_dir / "result.json").read_text())["meta"]["models"]["writing"]["provider"] == "offline"


def test_campaign_copy_respects_platform_limits_and_starts_paused(settings, fetcher, tmp_path):
    result = run(settings, fetcher, tmp_path, budget=3000.0)
    campaigns = {c["platform"]: c for c in result["results"]["campaigns"]["campaigns"]}
    assert set(campaigns) == {"google", "meta", "tiktok"} and all(c["status"] == "Paused" for c in campaigns.values())
    for group in campaigns["google"]["ad_groups"]:
        assert group["ready"] and 3 <= len(group["ad"]["headlines"]) <= 15
        assert all(len(h) <= LIMITS["google"]["headline"] and "!" not in h for h in group["ad"]["headlines"])
        assert all(35 <= len(d) <= LIMITS["google"]["description"] for d in group["ad"]["descriptions"])
    assert all(len(a["headline"]) <= 40 and len(a["description"]) <= 30 for a in campaigns["meta"]["ads"])
    assert all(len(a["ad_text"]) <= 100 for a in campaigns["tiktok"]["ads"])
    assert campaigns["tiktok"]["pixel_note"] and not campaigns["meta"]["pixel_note"]

    with open(tmp_path / "run" / "campaigns" / "google_ads_editor.csv", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["Campaign Status"] == "Paused" and rows[0]["Budget type"] == "Daily"
    assert any(r["Criterion Type"] == "Exact" for r in rows) and any(r["Criterion Type"] == "Negative Phrase" for r in rows)
    ads = [r for r in rows if r["Ad type"] == "Responsive search ad"]
    assert ads and all(r["Final URL"] == URL and r["Headline 3"] for r in ads)


def test_keyword_csv_turns_on_real_volumes_bids_and_traffic(settings, fetcher, tmp_path):
    planner = tmp_path / "planner.csv"
    planner.write_text("Keyword,Avg. monthly searches,Top of page bid (low range),Top of page bid (high range)\n"
                       "organic tea subscription box,2400,0.50,1.50\norganic tea subscription price,600,0.40,1.10\n"
                       "loose leaf tea online,1900,0.30,0.90\n")
    result = run(settings, fetcher, tmp_path, budget=1200.0, keyword_csv=str(planner))
    r = result["results"]
    assert r["keywords"]["has_volumes"]
    assert r["keywords"]["keywords"][0]["keyword"] == "organic tea subscription box"  # sorted by real volume
    assert any(k["keyword"] == "loose leaf tea online" for k in r["keywords"]["keywords"])  # added from the CSV
    traffic = r["traffic"]
    assert traffic["available"] and traffic["monthly_searches"] == 4900
    assert traffic["paid_clicks"][0] < traffic["paid_clicks"][1] and traffic["paid_cost"]["cpc"][0] < 1
    google = next(row for row in r["budget"]["plan"] if row["platform"] == "google")
    assert google["cpc_source"].startswith("Keyword Planner") and google["cpc"][0] < 0.6


def test_no_budget_proposes_a_test_tier_and_selected_steps_only(settings, fetcher, tmp_path):
    result = run(settings, fetcher, tmp_path, steps=["budget"], platforms=["google", "meta"])
    assert result["meta"]["steps"] == ["brief", "budget", "summary"]
    budget = result["results"]["budget"]
    assert budget["budget_was_proposed"] and budget["monthly_budget"] == budget["tiers"][0]["monthly"]
    assert [t["name"] for t in budget["tiers"]] == ["Test", "Steady", "Scale"]
    assert budget["tiers"][0]["monthly"] < budget["tiers"][1]["monthly"] < budget["tiers"][2]["monthly"]


def test_model_output_is_used_but_overlong_copy_is_rejected(settings, fetcher, tmp_path, monkeypatch):
    fake = FakeProvider({
        "Read this product page": {"product_name": "GreenLeaf", "category": "tea subscription box",
                                   "one_liner": "Monthly organic tea for people who love a calm cup.",
                                   "business_type": "ecommerce", "location": "Atlantis",
                                   "audience": ["tea lovers"], "pains": ["boring supermarket tea"],
                                   "usps": ["Small farm organic tea"], "seed_keywords": ["organic tea subscription", "greenleaf tea"]},
        "responsive search ad": {"headlines": ["Fresh Organic Tea Monthly", "x" * 31, "Amazing Tea!!"],
                                 "descriptions": ["Small farm organic tea, shipped to your door each month.", "y" * 91]},
        "Facebook and Instagram": {"ads": [{"angle": "Calm cup", "primary_text": "Tired of dull tea?\nTry ours.",
                                            "headline": "h" * 41, "description": "Ships monthly", "visual": "A steaming cup"}]},
        "executive summary": "GreenLeaf should lead with free shipping.",
    })
    monkeypatch.setattr(pipeline, "build_provider", lambda *_a, **_k: fake)
    result = run(settings, fetcher, tmp_path, budget=2000.0, steps=["campaigns"], platforms=["google", "meta"])
    r = result["results"]
    assert r["brief"]["written_by_model"] and r["brief"]["audience"] == ["Tea lovers"]
    assert r["brief"]["location"] == "Berlin"  # the model's invented location is not on the page, so it is ignored
    assert "greenleaf tea" not in r["brief"]["seed_keywords"]  # brand terms are not seed keywords
    ad = r["campaigns"]["campaigns"][0]["ad_groups"][0]["ad"]
    assert ad["headlines"][0] == "Fresh Organic Tea Monthly" and "Amazing Tea" in ad["headlines"]
    assert all(len(h) <= 30 for h in ad["headlines"]) and all(len(d) <= 90 for d in ad["descriptions"])
    meta_ad = r["campaigns"]["campaigns"][1]["ads"][0]
    assert meta_ad["primary_text"].startswith("Tired of dull tea?") and len(meta_ad["headline"]) <= 40
    assert meta_ad["description"] == "Ships monthly"
    assert r["summary"]["text"] == "GreenLeaf should lead with free shipping." and r["summary"]["written_by_model"]
    usage = result["meta"]["model_usage"]["fake:fake-1"]
    assert usage["calls"] >= 4 and usage["failed"] == 0 and usage["input_tokens"] == 10 * usage["calls"]


def test_unreachable_page_fails_cleanly_and_skips_dependents(settings, fetcher, tmp_path):
    config = RunConfig(url="https://nowhere.example/", video="no")
    result = run_pipeline(config, settings, fetcher=fetcher, run_dir=tmp_path / "run")
    assert "HTTP 404" in result["errors"]["brief"]
    assert "Skipped" in result["errors"]["keywords"] and result["results"] == {}
    assert (tmp_path / "run" / "report.html").exists()


def test_unavailable_model_downgrades_to_rules_with_a_warning(settings, fetcher, tmp_path):
    settings.research_provider = settings.writing_provider = "ollama"
    settings.ollama_host = "http://127.0.0.1:9"  # nothing listens here
    result = run(settings, fetcher, tmp_path, steps=["budget"])
    assert result["errors"] == {} and result["meta"]["models"]["research"]["provider"] == "offline"
    assert any("not available" in e["message"] for e in result["events"])


@pytest.mark.parametrize("kwargs,message", [
    ({"url": "not a url"}, "web address"), ({"url": URL, "platforms": ["myspace"]}, "Unknown platform"),
    ({"url": URL, "platforms": []}, "at least one"), ({"url": URL, "budget": -5}, "greater than zero"),
    ({"url": URL, "video": "maybe"}, "video must be"), ({"url": URL, "steps": ["dance"]}, "Unknown step"),
])
def test_validation_messages(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate(RunConfig(**kwargs))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_video_is_rendered_as_vertical_mp4(settings, fetcher, tmp_path):
    config = RunConfig(url=URL, steps=["video"], video="yes", platforms=["tiktok"])
    result = run_pipeline(config, settings, fetcher=fetcher, run_dir=tmp_path / "run")
    video = result["results"]["video"]
    assert video["rendered"], video
    assert video["format"].startswith("1080x1920") and 8 < video["seconds"] < 40
    assert len(video["scenes"]) >= 3 and video["scenes"][-1]["cta"]
    assert (tmp_path / "run" / "video" / "ad.mp4").stat().st_size > 20_000
    assert (tmp_path / "run" / "video" / "captions.srt").read_text().startswith("1\n00:00:00,000 --> ")


def test_video_auto_mode_skips_when_no_video_platform_is_funded(settings, fetcher, tmp_path):
    config = RunConfig(url=URL, steps=["video", "budget"], video="auto", budget=400.0, country="US")
    result = run_pipeline(config, settings, fetcher=fetcher, run_dir=tmp_path / "run")
    plan = {row["platform"] for row in result["results"]["budget"]["plan"]}
    video = result["results"]["video"]
    assert video["rendered"] == bool(plan & {"meta", "tiktok"}) or "ffmpeg" in video["reason"]
    config = RunConfig(url=URL, steps=["video"], video="auto", platforms=["google"])
    video = run_pipeline(config, settings, fetcher=fetcher, run_dir=tmp_path / "run2")["results"]["video"]
    assert not video["rendered"] and "text" in video["reason"]


def test_web_api_options_validation_and_file_safety(settings, monkeypatch, tmp_path):
    from adstrategist import server
    settings.openai_api_key, settings.meta_access_token = "sk-secret-123", "meta-secret-456"
    monkeypatch.setattr(server, "_settings", lambda: settings)
    client = TestClient(server.app)
    assert "AdStrategist" in client.get("/").text
    options = client.get("/api/options").json()
    assert [s["id"] for s in options["steps"]][0] == "brief" and options["providers"][0]["id"] == "offline"
    assert "secret" not in json.dumps(options)  # keys never reach the browser
    assert client.post("/api/runs", json={"url": "nope"}).status_code == 422
    assert client.post("/api/runs", json={"url": URL, "platforms": ["x"]}).json()["detail"].startswith("Unknown platform")
    assert client.get("/api/runs/does-not-exist").status_code == 404
    (settings.output_dir / "r1").mkdir(parents=True)
    (settings.output_dir / "r1" / "report.html").write_text("ok")
    (settings.output_dir / "secret.txt").write_text("no")
    assert client.get("/runs/r1/report.html").text == "ok"
    assert client.get("/runs/r1/%2e%2e/secret.txt").status_code == 404
    assert client.get("/api/runs").json() == []


def test_research_model_removes_off_target_keywords_but_never_most_of_them(settings, fetcher, tmp_path, monkeypatch):
    from .conftest import SUGGESTIONS
    monkeypatch.setitem(SUGGESTIONS, "organic tea subscription", [
        "organic tea subscription", "organic tea subscription box", "organic tea subscription gift",
        "organic tea subscription uk", "organic tea subscription review", "organic tea subscription monthly"])
    listed = {}

    class Picky(FakeProvider):
        def complete(self, prompt, **kwargs):
            if "NOT looking for" in prompt:
                lines = [l for l in prompt.splitlines() if l[:1].isdigit()]
                listed["n"] = len(lines)
                gift = next(i for i, l in enumerate(lines) if "gift" in l)
                self.answers = {"NOT looking for": {"not_relevant": listed.get("flag") or [gift, 999]}}
            return super().complete(prompt, **kwargs)

    monkeypatch.setattr(pipeline, "build_provider", lambda *_a, **_k: Picky())
    keywords = run(settings, fetcher, tmp_path, steps=["keywords"])["results"]["keywords"]
    assert keywords["removed_by_model"] == ["organic tea subscription gift"]  # the out-of-range index is ignored
    assert "organic tea subscription gift" not in {k["keyword"] for k in keywords["keywords"]}
    assert "organic tea subscription" in {k["keyword"] for k in keywords["keywords"]}  # seeds always stay

    listed["flag"] = list(range(listed["n"]))  # a model that rejects everything is not believed
    keywords = run(settings, fetcher, tmp_path / "b", steps=["keywords"])["results"]["keywords"]
    assert keywords["removed_by_model"] == []


def test_missing_paid_writing_model_falls_back_to_the_local_research_model(settings, fetcher, tmp_path, monkeypatch):
    from adstrategist.llm import ProviderStatus
    local = FakeProvider({"executive summary": "Written locally."}, model="local")

    class NoKey(FakeProvider):
        id, label = "openai", "OpenAI-compatible API (paid)"
        def status(self):
            return ProviderStatus(False, "No API key. Set OPENAI_API_KEY in .env.")

    monkeypatch.setattr(pipeline, "build_provider",
                        lambda pid, *_a, **_k: NoKey(model="paid") if pid == "openai" else local)
    settings.research_provider, settings.writing_provider = "ollama", "openai"
    result = run(settings, fetcher, tmp_path, steps=["budget"])
    assert result["meta"]["models"]["writing"]["model"] == "local"
    assert result["results"]["summary"]["text"] == "Written locally."
    assert any("Using the research model" in e["message"] for e in result["events"])


def test_summary_figures_are_checked_against_the_data_it_was_given(settings, fetcher, tmp_path, monkeypatch):
    class Rounder(FakeProvider):
        def complete(self, prompt, **kwargs):
            if "executive summary" in prompt:
                self.answers = {"executive summary": "Put 900 USD into the plan, as given." if "previous draft" in prompt
                                else "Put 60% into Google and 40% into Meta."}
            return super().complete(prompt, **kwargs)

    model = Rounder()
    monkeypatch.setattr(pipeline, "build_provider", lambda *_a, **_k: model)
    result = run(settings, fetcher, tmp_path, steps=["budget"], budget=900.0)
    summary = result["results"]["summary"]
    assert summary["text"] == "Put 900 USD into the plan, as given." and summary["figures_corrected"]
    correction = next(p for p in model.prompts if "previous draft" in p)
    assert "60%, 40%" in correction and '"share":' in correction
    assert any("figures it was not given" in e["message"] for e in result["events"])

    stubborn = FakeProvider({"executive summary": "Put 60% into Google."})
    monkeypatch.setattr(pipeline, "build_provider", lambda *_a, **_k: stubborn)
    result = run_pipeline(RunConfig(url=URL, steps=["budget"], budget=900.0, video="no"), settings, fetcher=fetcher,
                          run_dir=tmp_path / "run2")
    summary = result["results"]["summary"]
    assert not summary["written_by_model"] and "rule based" in summary["text"] and "60%" not in summary["text"]
    assert any(c["where"] == "executive summary" for c in result["meta"]["rejected_claims"])
