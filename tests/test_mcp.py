"""The MCP server, exercised through the SDK's in-memory client."""
from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp.server.mcpserver")

from mcp import Client  # noqa: E402

from adstrategist import ask as ask_module  # noqa: E402
from adstrategist import pipeline  # noqa: E402
from adstrategist.context import RunConfig  # noqa: E402
from adstrategist.mcp_server import build_server  # noqa: E402
from adstrategist.pipeline import run_pipeline  # noqa: E402

URL = "https://greenleaf.example/"


def call(server, tool: str, arguments: dict | None = None):
    async def go():
        async with Client(server) as client:
            return await client.call_tool(tool, arguments or {})
    result = asyncio.run(go())
    text = "".join(block.text for block in result.content if getattr(block, "text", None))
    if not result.is_error and result.structured_content is not None:
        data = result.structured_content
        return False, data["result"] if set(data) == {"result"} else data
    try:
        return result.is_error, json.loads(text)
    except ValueError:
        return result.is_error, text


@pytest.fixture
def finished(settings, fetcher):
    result = run_pipeline(RunConfig(url=URL, budget=900.0, video="no", country="US", currency="USD"), settings,
                          fetcher=fetcher, run_dir=settings.output_dir / "greenleaf-example-20260101-000000")
    return result["meta"]["run_id"]


def test_server_lists_its_tools_and_resource_template(settings):
    server = build_server(settings)

    async def go():
        async with Client(server) as client:
            tools = await client.list_tools()
            templates = await client.list_resource_templates()
            return client.instructions, tools.tools, templates
    instructions, tools, templates = asyncio.run(go())
    assert {t.name for t in tools} == {"start_research", "research_status", "list_runs", "get_report",
                                       "get_campaigns", "ask_research"}
    assert all(t.description for t in tools) and "poll research_status" in instructions
    start = next(t for t in tools if t.name == "start_research")
    assert start.input_schema["required"] == ["url"] and "monthly_budget" in start.input_schema["properties"]
    assert "adstrategist://runs/{run_id}/report" in str(templates)


def test_reports_campaigns_and_runs_are_readable_through_tools(settings, finished):
    server = build_server(settings)
    error, runs = call(server, "list_runs")
    assert not error and runs[0]["run_id"] == finished and runs[0]["can_answer_questions"] is False

    error, report = call(server, "get_report")
    assert not error and report.startswith("# Ad strategy for GreenLeaf") and "## Budget plan" in report
    error, section = call(server, "get_report", {"run_id": finished, "section": "budget plan"})
    assert not error and section.startswith("## Budget plan") and "## Ad campaigns" not in section
    error, message = call(server, "get_report", {"section": "Horoscope"})
    assert error and "Sections:" in message and "Budget plan" in message
    error, message = call(server, "get_report", {"run_id": "no-such-run"})
    assert error and "No finished run" in message

    error, campaigns = call(server, "get_campaigns", {"platform": "google"})
    assert not error and [c["platform"] for c in campaigns["campaigns"]] == ["google"]
    assert campaigns["campaigns"][0]["status"] == "Paused"
    assert call(server, "get_campaigns", {"platform": "myspace"})[0]

    async def read():
        async with Client(server) as client:
            return await client.read_resource(f"adstrategist://runs/{finished}/report")
    assert "# Ad strategy for GreenLeaf" in str(asyncio.run(read()))


def test_start_research_runs_in_the_background_and_reports_status(settings, monkeypatch):
    seen = {}

    def fake_run(config, cfg, on_event=None, run_dir=None):
        seen.update(url=config.url, budget=config.budget, steps=config.steps, platforms=config.platforms,
                    engine=config.engine)
        on_event({"t": 0.1, "level": "done", "step": "brief", "message": "Product brief"})
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text(json.dumps({"meta": {"url": config.url, "started": "2026-01-01T00:00:00"}}))
        return {"errors": {}}

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run)
    server = build_server(settings)
    error, started = call(server, "start_research", {"url": "greenleaf.example", "monthly_budget": 500,
                                                     "platforms": "google, meta", "steps": "budget",
                                                     "engine": "langgraph"})
    assert not error and started["status"] == "running" and started["steps"] == ["brief", "budget", "summary"]
    for _ in range(200):
        error, status = call(server, "research_status", {"run_id": started["run_id"]})
        if status["status"] != "running":
            break
    assert status["status"] == "done" and status["progress"] == ["[0.1s] Product brief"]
    assert seen == {"url": "https://greenleaf.example", "budget": 500, "steps": ["brief", "budget", "summary"],
                    "platforms": ["google", "meta"], "engine": "langgraph"}

    error, message = call(server, "start_research", {"url": "not a url"})
    assert error and "web address" in message
    error, message = call(server, "start_research", {"url": URL, "platforms": "myspace"})
    assert error and "Unknown platform" in message
    assert call(server, "research_status", {"run_id": "nope"})[0]


def test_ask_research_returns_cited_sources_or_a_clear_error(settings, finished, monkeypatch):
    server = build_server(settings)
    error, message = call(server, "ask_research", {"question": "anything"})
    assert error and "no retrieval index" in message  # this run was made with retrieval off

    def fake_ask(run_dir, question, cfg):
        return {"answer": "Google gets most of the budget [1].", "found": True, "figures_verified": True,
                "sources": [{"n": 1, "kind": "report", "title": "Report: Budget plan", "url": "report#budget-plan",
                             "quote": "Budget plan. " * 60, "score": 0.51}]}

    monkeypatch.setattr(ask_module, "ask", fake_ask)
    error, out = call(build_server(settings), "ask_research", {"question": "how is the budget split?", "run_id": finished})
    assert not error and out["found"] and out["answer"].endswith("[1].")
    assert out["sources"] == [{"n": 1, "where": "Report: Budget plan", "quote": ("Budget plan. " * 60)[:300],
                               "similarity": 0.51}]
