"""Model Context Protocol server: lets an MCP client (Claude Desktop, an IDE, another agent) use
AdStrategist as a set of tools.

    start_research   start a research run for a product link (returns at once, the run takes minutes)
    research_status  progress and errors of a run
    list_runs        finished runs
    get_report       the report of a run, whole or one section
    get_campaigns    the generated ad campaigns of a run
    ask_research     a question answered from the run's retrieval index, with cited sources

plus one resource template, adstrategist://runs/{run_id}/report.

Run it with `adstrategist mcp` (stdio transport). Nothing may be printed to stdout in this module:
on stdio, stdout is the protocol channel.
"""
from __future__ import annotations

import functools
import json
import re
import threading
from datetime import datetime
from pathlib import Path

from .config import PLATFORMS, Settings
from .context import RunConfig
from .steps import STEPS
from .textutil import registrable_domain, slugify

INSTRUCTIONS = (
    "AdStrategist researches a product's market and writes ad campaigns. A research run takes two to five "
    "minutes on a laptop: call start_research, then poll research_status until it is done. After that use "
    "get_report, get_campaigns and ask_research. ask_research answers only from what the run retrieved and "
    "says so when the research does not cover a question. Budget figures are benchmark ranges unless the "
    "report says real keyword data was supplied."
)


def build_server(settings: Settings | None = None):
    """Create the MCP server. `settings` is injectable for tests."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    from .ask import ask, resolve_run
    from .pipeline import run_pipeline, validate
    from .rag import RagUnavailable

    get_settings = (lambda: settings) if settings else Settings.from_env
    server = MCPServer("adstrategist", title="AdStrategist", instructions=INSTRUCTIONS,
                       description="Marketing research, budget planning and ad campaigns from one product link.")
    runs: dict[str, dict] = {}
    lock = threading.Lock()

    def readable(function):
        """The SDK hides the text of ordinary exceptions from clients. Problems the caller can fix
        (bad link, unknown run, missing index) are re-raised as ToolError so the message gets through."""
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except (ValueError, FileNotFoundError) as exc:
                raise ToolError(str(exc)) from exc
        return wrapper

    def run_folder(run_id: str) -> Path:
        try:
            return resolve_run(get_settings(), run_id)
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc

    @server.tool(description="Start a marketing research run for a product, website or app link. Returns a run_id "
                             "immediately; the run itself takes a few minutes. Poll research_status with the run_id.")
    @readable
    def start_research(url: str, monthly_budget: float | None = None, platforms: str = "google,meta,tiktok",
                       steps: str = "", country: str = "", engine: str = "",
                       competitor_urls: str = "") -> dict:
        """platforms and steps are comma separated. steps may be empty (all steps) or any of: brief, competitors,
        ad_intel, keywords, gaps, traffic, budget, campaigns, video, summary. engine: native or langgraph."""
        cfg = get_settings()
        config = RunConfig(
            url=url, budget=monthly_budget, steps=[s.strip() for s in steps.split(",") if s.strip()],
            platforms=[p.strip() for p in platforms.split(",") if p.strip()],
            country=country or cfg.default_country, currency=cfg.default_currency, engine=engine, video="no",
            competitors=[c.strip() for c in competitor_urls.split(",") if c.strip()])
        validate(config)  # raises ValueError with a readable message, which the client sees as a tool error
        with lock:
            if any(r["status"] == "running" for r in runs.values()):
                raise ValueError("A research run is already in progress. Wait for it to finish.")
            run_id = f"{slugify(registrable_domain(config.url))}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            runs[run_id] = {"status": "running", "events": [], "errors": {}, "error": ""}

        def work() -> None:
            state = runs[run_id]
            try:
                result = run_pipeline(config, cfg, on_event=state["events"].append,
                                      run_dir=cfg.output_dir / run_id)
                state["errors"] = result["errors"]
                state["status"] = "done"
            except Exception as exc:
                state["error"], state["status"] = f"{type(exc).__name__}: {exc}", "failed"

        threading.Thread(target=work, daemon=True).start()
        return {"run_id": run_id, "status": "running", "steps": config.steps,
                "next": "Call research_status with this run_id until status is done."}

    @server.tool(description="Progress of a research run started with start_research: status, the latest "
                             "progress lines, and any step that failed.")
    @readable
    def research_status(run_id: str) -> dict:
        state = runs.get(run_id)
        if state is None:
            run_folder(run_id)  # raises if it is not a finished run either
            return {"run_id": run_id, "status": "done", "progress": [], "errors": {}}
        lines = [f"[{e['t']}s] {e['message']}" for e in state["events"] if e["level"] != "info"][-8:]
        return {"run_id": run_id, "status": state["status"], "progress": lines, "errors": state["errors"],
                "error": state["error"]}

    @server.tool(description="List finished research runs, newest first.")
    @readable
    def list_runs() -> list[dict]:
        base = get_settings().output_dir
        if not base.is_dir():
            return []
        rows = []
        for folder in sorted((p for p in base.iterdir() if (p / "result.json").exists()),
                             key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
            try:
                meta = json.loads((folder / "result.json").read_text(encoding="utf-8"))["meta"]
            except (OSError, ValueError, KeyError):
                continue
            rows.append({"run_id": folder.name, "url": meta["url"], "started": meta["started"],
                         "can_answer_questions": (folder / "index").is_dir()})
        return rows

    def report_text(run_id: str, section: str = "") -> str:
        text = (run_folder(run_id) / "report.md").read_text(encoding="utf-8")
        if not section:
            return text
        parts = re.split(r"(?m)^## ", text)
        for part in parts[1:]:
            if part.lower().startswith(section.lower().strip()):
                return "## " + part.strip()
        names = [p.splitlines()[0] for p in parts[1:]]
        raise ValueError(f"No section named '{section}'. Sections: {', '.join(names)}")

    @server.tool(description="The Markdown report of a run. Leave run_id empty for the latest run. Pass a section "
                             "name (for example 'Budget plan', 'Competitors', 'Keyword research', 'Market gaps', "
                             "'Ad campaigns', 'Summary') to get only that part.")
    @readable
    def get_report(run_id: str = "", section: str = "") -> str:
        return report_text(run_id, section)[:24000]

    @server.tool(description="The ad campaigns a run generated, as structured data. Optionally one platform: "
                             "google, meta or tiktok. Campaigns are drafts with status Paused.")
    @readable
    def get_campaigns(run_id: str = "", platform: str = "") -> dict:
        path = run_folder(run_id) / "campaigns" / "campaigns.json"
        if not path.exists():
            raise ValueError("This run did not include the ad campaigns step.")
        data = json.loads(path.read_text(encoding="utf-8"))
        if platform:
            if platform not in PLATFORMS:
                raise ValueError(f"Unknown platform '{platform}'. Choose from: {', '.join(PLATFORMS)}")
            data["campaigns"] = [c for c in data["campaigns"] if c["platform"] == platform]
        return data

    @server.tool(description="Ask a question about a run. Answered only from passages retrieved from the pages "
                             "the run read and from its report, with numbered sources. Says so when the research "
                             "does not cover the question.")
    @readable
    def ask_research(question: str, run_id: str = "") -> dict:
        try:
            result = ask(run_folder(run_id), question, get_settings())
        except RagUnavailable as exc:
            raise ValueError(str(exc)) from exc
        return {"answer": result["answer"], "found": result["found"],
                "figures_verified": result["figures_verified"],
                "sources": [{"n": s["n"], "where": s["url"] if s["kind"] == "page" else s["title"],
                             "quote": s["quote"][:300], "similarity": s["score"]} for s in result["sources"]]}

    @server.resource("adstrategist://runs/{run_id}/report", mime_type="text/markdown",
                     description="The Markdown report of one research run.")
    def report_resource(run_id: str) -> str:
        return report_text(run_id)

    server.step_ids = list(STEPS)  # for introspection in tests and docs
    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
