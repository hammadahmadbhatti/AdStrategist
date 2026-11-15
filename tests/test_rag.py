"""Retrieval layer, cited question answering and the LangGraph engine. No network, no real models:
a deterministic bag-of-words embedder stands in for the embedding model."""
from __future__ import annotations

import hashlib
import json
import math

import pytest

pytest.importorskip("langchain_chroma")
pytest.importorskip("langgraph")

from langchain_core.embeddings import Embeddings  # noqa: E402

from adstrategist import pipeline, rag  # noqa: E402
from adstrategist import ask as ask_module  # noqa: E402
from adstrategist.ask import NOT_FOUND, ask  # noqa: E402
from adstrategist.context import RunConfig  # noqa: E402
from adstrategist.graph import diagram  # noqa: E402
from adstrategist.llm import LLMRouter  # noqa: E402
from adstrategist.pipeline import run_pipeline, validate  # noqa: E402
from adstrategist.rag import RagIndex  # noqa: E402

from .conftest import FakeProvider  # noqa: E402

URL = "https://greenleaf.example/"


class WordEmbeddings(Embeddings):
    """Texts that share words get similar vectors. Enough to test retrieval logic deterministically."""

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * 4096
        for word in "".join(c.lower() if c.isalnum() else " " for c in text).split():
            if len(word) > 2:
                vector[int(hashlib.md5(word[:6].encode()).hexdigest(), 16) % 4096] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def rag_settings(settings):
    settings.rag, settings.rag_min_score = "auto", 0.05
    return settings


def run(settings, fetcher, tmp_path, name="run", **overrides):
    config = RunConfig(url=URL, country="US", currency="USD", video="no", **overrides)
    return run_pipeline(config, settings, fetcher=fetcher, run_dir=tmp_path / name, embeddings=WordEmbeddings())


# ------------------------------------------------------------------ index

def test_index_chunks_filters_scores_and_survives_a_reopen(tmp_path, rag_settings):
    index = RagIndex.open(tmp_path / "index", rag_settings, WordEmbeddings())
    tea = ("Organic loose leaf tea from small farms. " * 6 + "Free shipping on every subscription box. " * 6 +
           "Pause or cancel the subscription any time. " * 6)
    assert index.add_text(tea, url="https://own.example/", site="own", title="Own") >= 2  # split into chunks
    index.add_text("Tea gift boxes with a thirty day money-back guarantee for every customer order.",
                   url="https://rival.example/", site="rival.example", title="Rival")
    index.add_text("Budget plan. Google Search receives sixty percent of the monthly budget.",
                   url="report#budget-plan", site="report", title="Report: Budget plan", kind="report")
    total = index.count()
    assert index.add_text(tea, url="https://own.example/", site="own") and index.count() == total  # idempotent
    assert index.add_text("too short", url="x", site="own") == 0

    best = index.search("money-back guarantee", k=3)[0]
    assert best.site == "rival.example" and 0 < best.score <= 1 and best.url == "https://rival.example/"
    assert all(p.site == "own" for p in index.search("guarantee", k=5, site="own"))
    assert all(p.site != "own" for p in index.search("tea", k=5, exclude_site="own"))
    assert [p.kind for p in index.search("monthly budget", k=5, kind="report")] == ["report"]
    assert index.search("quantum spaceship physics", k=1)[0].score < 0.05
    assert index.stats()["vector_store"] == "Chroma" and index.stats()["chunks"] == total

    reopened = RagIndex.open(tmp_path / "index", rag_settings, WordEmbeddings())
    assert reopened.count() == total and reopened.search("money-back guarantee", k=1)[0].site == "rival.example"


def test_embedding_prefixes_are_applied_per_model():
    seen = []

    class Recorder(Embeddings):
        def embed_documents(self, texts):
            seen.extend(texts)
            return [[1.0] for _ in texts]

        def embed_query(self, text):
            seen.append(text)
            return [1.0]

    wrapped = rag._prefixed(Recorder(), "embeddinggemma:latest")
    wrapped.embed_documents(["a passage"])
    wrapped.embed_query("a question")
    assert seen == ["title: none | text: a passage", "task: search result | query: a question"]
    plain = Recorder()
    assert rag._prefixed(plain, "some-other-model") is plain


def test_status_explains_what_is_missing(settings, monkeypatch):
    import httpx
    settings.rag = "off"
    assert "switched off" in rag.rag_status(settings)["detail"]
    settings.rag, settings.embed_model = "auto", "embeddinggemma"
    monkeypatch.setattr(httpx, "get", lambda url, **_: httpx.Response(
        200, json={"models": [{"name": "llama3.2:3b"}]}, request=httpx.Request("GET", url)))
    status = rag.rag_status(settings)
    assert not status["ready"] and "ollama pull embeddinggemma" in status["detail"]
    monkeypatch.setattr(httpx, "get", lambda url, **_: httpx.Response(
        200, json={"models": [{"name": "embeddinggemma:latest"}]}, request=httpx.Request("GET", url)))
    assert rag.rag_status(settings)["ready"]


# ------------------------------------------------------------------ retrieval inside the pipeline

def test_pipeline_retrieves_passages_for_prompts_and_backs_gaps_with_evidence(rag_settings, fetcher, tmp_path, monkeypatch):
    model = FakeProvider({"Read this product page": {
        "product_name": "GreenLeaf", "category": "organic tea subscription", "one_liner": "Monthly organic tea.",
        "business_type": "ecommerce", "location": "", "audience": [], "pains": [],
        "usps": ["Organic loose leaf tea"], "seed_keywords": ["organic tea subscription", "loose leaf tea box"]}})
    monkeypatch.setattr(pipeline, "build_provider", lambda *_a, **_k: model)
    result = run(rag_settings, fetcher, tmp_path, steps=["gaps"])
    meta, r = result["meta"], result["results"]

    brief_prompt = next(p for p in model.prompts if "Read this product page" in p)
    assert "Most relevant passages from the site's pages:" in brief_prompt and "Page text:" not in brief_prompt
    assert "organic" in brief_prompt.split("Most relevant passages")[1].lower()
    assert r["brief"]["retrieval"]["used"] and r["brief"]["retrieval"]["passages"] >= 1
    profile_prompt = next(p for p in model.prompts if "Describe this competitor" in p)
    assert "Most relevant passages:" in profile_prompt

    assert meta["rag"]["enabled"] and meta["rag"]["vector_store"] == "Chroma"
    assert meta["rag"]["documents"] >= 3 and meta["rag"]["retrievals"] >= 4  # own page, two competitors, report
    gift = next(t for t in r["gaps"]["topics"] if "gift boxes" in t["topic"])
    assert gift["evidence"]["site"] in {"teabox.example", "leafclub.example"}
    assert "gift boxes" in gift["evidence"]["quote"].lower() and gift["evidence"]["url"].startswith("https://")
    assert "closest_on_your_site" in gift and gift["possibly_covered"] is False
    assert r["gaps"]["evidence_attached"] >= 1
    report = (tmp_path / "run" / "report.md").read_text()
    assert "Evidence (retrieved passage)" in report and "embedded locally" in report

    index = RagIndex.open(tmp_path / "run" / "index", rag_settings, WordEmbeddings())
    assert index.search("market gaps topics competitors lead with", k=3, kind="report")  # the report is indexed too


def test_retrieval_off_or_unavailable_falls_back_to_trimmed_text(settings, fetcher, tmp_path, monkeypatch):
    model = FakeProvider()
    monkeypatch.setattr(pipeline, "build_provider", lambda *_a, **_k: model)
    result = run(settings, fetcher, tmp_path, steps=["budget"])  # settings.rag == "off"
    assert not result["meta"]["rag"]["enabled"] and not (tmp_path / "run" / "index").exists()
    assert "Page text:" in next(p for p in model.prompts if "Read this product page" in p)

    settings.rag = "auto"
    monkeypatch.setattr(rag, "rag_status", lambda _s: {"ready": False, "detail": "The embedding model x is not installed."})
    config = RunConfig(url=URL, steps=["budget"], video="no")
    result = run_pipeline(config, settings, fetcher=fetcher, run_dir=tmp_path / "run2")
    assert result["errors"] == {} and "not installed" in result["meta"]["rag"]["detail"]
    assert any("Retrieval is off for this run" in e["message"] for e in result["events"])
    config = RunConfig(url=URL, steps=["budget"], video="no", rag=False)
    assert not run_pipeline(config, settings, fetcher=fetcher, run_dir=tmp_path / "run3")["meta"]["rag"]["enabled"]


# ------------------------------------------------------------------ cited question answering

@pytest.fixture
def finished_run(rag_settings, fetcher, tmp_path):
    run(rag_settings, fetcher, tmp_path, budget=900.0)
    return tmp_path / "run"


def test_ask_answers_from_retrieved_sources_with_citations(finished_run, rag_settings):
    model = FakeProvider({"Question:": "LeafClub offers a money-back guarantee on gift boxes [1]."})
    out = ask(finished_run, "  which competitor has a   money-back guarantee on gift boxes? ", rag_settings,
              llm=LLMRouter(model, model), embeddings=WordEmbeddings())
    assert out["found"] and out["answer"].endswith("[1].") and out["question"].startswith("which competitor")
    assert [s["n"] for s in out["sources"]] == [1] and out["sources"][0]["quote"] and out["sources"][0]["score"] > 0.05
    prompt = model.prompts[0]
    assert prompt.startswith("Sources:\n[1] (") and "Question: which competitor has a money-back" in prompt
    assert "guarantee" in prompt.lower()


def test_ask_never_generates_an_answer_when_nothing_relevant_is_retrieved(finished_run, rag_settings):
    rag_settings.rag_min_score = 0.2
    model = FakeProvider({"Question:": "I will happily make something up."})
    out = ask(finished_run, "quantum spaceship physics", rag_settings, llm=LLMRouter(model, model),
              embeddings=WordEmbeddings())
    assert out == {**out, "found": False, "answer": NOT_FOUND, "sources": [], "retrieved": 0}
    assert out["trace"] == ["retrieve", "rewrite", "retrieve", "refuse"]
    assert not any("Sources:" in p for p in model.prompts)  # it tried one rewrite, but was never asked to answer


def test_ask_handles_uncited_and_refusing_answers(finished_run, rag_settings):
    uncited = FakeProvider({"Question:": "Google gets most of the budget."})
    out = ask(finished_run, "how is the monthly budget split between Google and Meta", rag_settings,
              llm=LLMRouter(uncited, uncited), embeddings=WordEmbeddings())
    assert out["found"] and "did not cite its sources" in out["answer"] and len(out["sources"]) == out["retrieved"]
    assert any(s["kind"] == "report" for s in out["sources"])  # the analysis itself is retrievable
    refusing = FakeProvider({"Question:": NOT_FOUND})
    out = ask(finished_run, "how is the monthly budget split", rag_settings, llm=LLMRouter(refusing, refusing),
              embeddings=WordEmbeddings())
    assert not out["found"] and out["sources"] == []


def test_ask_needs_an_index_a_question_and_a_model(finished_run, rag_settings, settings, fetcher, tmp_path):
    from adstrategist.llm.providers import OfflineProvider
    from adstrategist.rag import RagUnavailable
    with pytest.raises(ValueError):
        ask(finished_run, "   ", rag_settings, embeddings=WordEmbeddings())
    offline = LLMRouter(OfflineProvider(), OfflineProvider())
    with pytest.raises(RagUnavailable, match="No model"):
        ask(finished_run, "anything", rag_settings, llm=offline, embeddings=WordEmbeddings())
    settings.rag = "off"
    run_pipeline(RunConfig(url=URL, steps=["budget"], video="no"), settings, fetcher=fetcher, run_dir=tmp_path / "noindex")
    with pytest.raises(RagUnavailable, match="no retrieval index"):
        ask(tmp_path / "noindex", "anything", settings, embeddings=WordEmbeddings())


# ------------------------------------------------------------------ LangGraph engine

def test_langgraph_engine_produces_the_same_research_as_the_native_engine(settings, fetcher, tmp_path):
    native = run(settings, fetcher, tmp_path, "native", budget=900.0, engine="native")
    graph = run(settings, fetcher, tmp_path, "graph", budget=900.0, engine="langgraph")
    assert graph["errors"] == native["errors"] == {}
    assert json.dumps(graph["results"], sort_keys=True) == json.dumps(native["results"], sort_keys=True)
    assert graph["meta"]["engine"] == "langgraph" and native["meta"]["engine"] == "native"
    info = graph["meta"]["graph"]
    assert info["path"] == ["brief", "competitors", "ad_intel", "keywords", "gaps", "traffic", "budget", "campaigns",
                            "review_copy", "video", "summary"]
    assert info["copy_revisions"] == 0 and info["nodes"] == 12
    assert "review_copy" in (tmp_path / "graph" / "graph.mmd").read_text()
    assert "LangGraph state graph" in (tmp_path / "graph" / "report.md").read_text()


def test_langgraph_review_loop_sends_rejected_copy_back_once(settings, fetcher, tmp_path, monkeypatch):
    class Writer(FakeProvider):
        """Invents a discount in its first draft. Writes clean copy once the reviewer's notes arrive."""

        def complete(self, prompt, **kwargs):
            if "responsive search ad" in prompt:
                revised = "A reviewer rejected" in prompt
                self.answers = {"responsive search ad": {
                    "headlines": ["Organic Tea, Delivered Monthly"] if revised else ["50% Off Tea Today"],
                    "descriptions": ["Organic loose leaf tea from small farms, shipped to you each month."]}}
            return super().complete(prompt, **kwargs)

    model = Writer()
    monkeypatch.setattr(pipeline, "build_provider", lambda *_a, **_k: model)
    result = run(settings, fetcher, tmp_path, steps=["campaigns"], platforms=["google"], engine="langgraph")
    info = result["meta"]["graph"]
    assert info["copy_revisions"] == 1
    assert info["path"].count("campaigns") == 2 and info["path"].count("review_copy") == 2
    second_drafts = [p for p in model.prompts if "A reviewer rejected" in p]
    assert second_drafts and "50% Off Tea Today" in second_drafts[0] and "is not on the page" in second_drafts[0]
    headlines = result["results"]["campaigns"]["campaigns"][0]["ad_groups"][0]["ad"]["headlines"]
    assert "Organic Tea, Delivered Monthly" in headlines and not any("50%" in h for h in headlines)
    assert any("50% Off" in c["text"] for c in result["meta"]["rejected_claims"])  # the first draft stays on record
    assert any("second draft" in e["message"] for e in result["events"])

    native = run(settings, fetcher, tmp_path, "native", steps=["campaigns"], platforms=["google"], engine="native")
    assert not any("Organic Tea, Delivered Monthly" in h
                   for h in native["results"]["campaigns"]["campaigns"][0]["ad_groups"][0]["ad"]["headlines"])


def test_langgraph_review_loop_gives_up_after_one_revision(settings, fetcher, tmp_path, monkeypatch):
    stubborn = FakeProvider({"responsive search ad": {"headlines": ["Tea With 80% Discount"], "descriptions": []}})
    monkeypatch.setattr(pipeline, "build_provider", lambda *_a, **_k: stubborn)
    result = run(settings, fetcher, tmp_path, steps=["campaigns"], platforms=["google"], engine="langgraph")
    assert result["meta"]["graph"]["path"].count("campaigns") == 2 and result["meta"]["graph"]["copy_revisions"] == 1
    assert result["errors"] == {} and result["results"]["campaigns"]["campaigns"][0]["ad_groups"][0]["ready"]


def test_langgraph_routes_an_unreadable_page_to_abort(settings, fetcher, tmp_path):
    kwargs = dict(url="https://nowhere.example/", video="no")
    native = run_pipeline(RunConfig(**kwargs, engine="native"), settings, fetcher=fetcher, run_dir=tmp_path / "n")
    graph = run_pipeline(RunConfig(**kwargs, engine="langgraph"), settings, fetcher=fetcher, run_dir=tmp_path / "g")
    assert graph["meta"]["graph"]["path"] == ["brief", "abort"]
    assert graph["errors"] == native["errors"] and graph["results"] == {}


def test_graph_diagram_and_engine_validation():
    text = diagram()
    assert "review_copy -. &nbsp;revise&nbsp; .-> campaigns" in text and "brief -. &nbsp;page unreadable&nbsp; .-> abort" in text
    assert "campaigns" not in diagram(["brief", "budget", "summary"])
    with pytest.raises(ValueError, match="engine"):
        validate(RunConfig(url=URL, engine="crewai"))


def test_cli_and_web_api_expose_retrieval_and_engine(settings, monkeypatch, capsys):
    from fastapi.testclient import TestClient

    from adstrategist import cli, server
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: settings))
    assert cli.main(["graph"]) == 0 and "graph TD" in capsys.readouterr().out
    assert cli.main(["graph", "ask"]) == 0 and "rewrite" in capsys.readouterr().out
    assert cli.main(["ask", "anything"]) == 2 and "no finished runs" in capsys.readouterr().err
    monkeypatch.setattr(server, "_settings", lambda: settings)
    client = TestClient(server.app)
    options = client.get("/api/options").json()
    assert options["engine"] == "native" and options["rag"] == {"ready": False, "detail": "Retrieval is switched off (RAG=off in .env)."}
    (settings.output_dir / "old-run").mkdir(parents=True)
    assert client.get("/api/runs/old-run").json()["has_index"] is False
    assert client.post("/api/runs/old-run/ask", json={"question": "anything"}).status_code == 422
    assert client.post("/api/runs/missing/ask", json={"question": "anything"}).status_code == 404
    assert client.post("/api/runs", json={"url": URL, "engine": "crewai"}).status_code == 422


def test_ask_catches_misquoted_figures_and_retries_once(finished_run, rag_settings):
    question = "how is the monthly budget split between Google and Meta"

    class Rounder(FakeProvider):
        """Rounds figures the first time, quotes them exactly once it is corrected."""

        def complete(self, prompt, **kwargs):
            self.answers = {"Question:": "Google Search gets the larger share [1]." if "not in the sources" in prompt
                            else "Google gets 97% and Meta gets 3% [1]."}
            return super().complete(prompt, **kwargs)

    model = Rounder()
    out = ask(finished_run, question, rag_settings, llm=LLMRouter(model, model), embeddings=WordEmbeddings())
    assert out["figures_verified"] and "97%" not in out["answer"] and len(model.prompts) == 2
    assert "97%, 3%" in model.prompts[1]

    stubborn = FakeProvider({"Question:": "Google gets 97% of the 12345 budget [1]."})
    out = ask(finished_run, question, rag_settings, llm=LLMRouter(stubborn, stubborn), embeddings=WordEmbeddings())
    assert not out["figures_verified"] and "Check the figures: 97%, 12345" in out["answer"] and len(stubborn.prompts) == 2


def test_unverified_figures_ignores_citations_list_numbers_and_the_question():
    from adstrategist.ask import unverified_figures
    from adstrategist.rag import Passage
    passages = [Passage("Share: 55%; Monthly: 327.27 EUR. Founded 1952.", "u", "report", "t", "report", 0.5)]
    assert unverified_figures("1. Google gets 55% [1]\n2. That is 327.27 EUR [2], since 1952.", passages, "q") == []
    assert unverified_figures("Google gets 60% and Meta 40% [1].", passages, "q") == ["60%", "40%"]
    assert unverified_figures("A budget of 600 is planned.", passages, "what about a budget of 600?") == []


def test_agentic_rag_rewrites_a_failed_search_and_tries_again(finished_run, rag_settings):
    rag_settings.rag_min_score = 0.2
    model = FakeProvider({"Rewrite this question": '"money-back guarantee gift boxes"',
                          "Sources:": "LeafClub gives a money-back guarantee on gift boxes [1]."})
    out = ask(finished_run, "zzqx blorp wibble?", rag_settings, llm=LLMRouter(model, model), embeddings=WordEmbeddings())
    assert out["trace"] == ["retrieve", "rewrite", "retrieve", "generate", "verify"]
    assert out["rewritten_query"] == "money-back guarantee gift boxes" and out["found"] and out["sources"]
    assert "zzqx blorp wibble?" in model.prompts[-1]  # the answer is still to the user's own question

    direct = ask(finished_run, "money-back guarantee on gift boxes", rag_settings, llm=LLMRouter(model, model),
                 embeddings=WordEmbeddings())
    assert direct["trace"] == ["retrieve", "generate", "verify"] and direct["rewritten_query"] == ""


def test_ask_graph_diagram_shows_both_loops():
    text = ask_module.diagram()
    assert "rewrite --> retrieve" in text and "verify -. &nbsp;correct&nbsp; .-> generate" in text
    assert "retrieve -. &nbsp;nothing found&nbsp; .-> refuse" in text
