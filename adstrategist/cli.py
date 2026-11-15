"""Command line interface: run, providers, steps, serve."""
from __future__ import annotations

import argparse
import sys

from .config import PLATFORMS, Settings
from .context import RunConfig
from .llm import PROVIDER_IDS, provider_catalog
from .steps import ALWAYS, STEPS


def _csv(value: str) -> list[str]:
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def _ask(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{question}{suffix}: ").strip()
    return answer or default


def interactive(settings: Settings, url: str | None) -> RunConfig:
    """A short question-and-answer setup for people who prefer not to remember flags."""
    print("\nAdStrategist setup. Press Enter to accept the value in brackets.\n")
    url = url or _ask("Product, website or app link")
    print("\nSteps:")
    optional = [s for s in STEPS if s not in ALWAYS]
    for i, step in enumerate(optional, 1):
        print(f"  {i}. {STEPS[step][0]}: {STEPS[step][1]}")
    picked = _ask("Which steps? Numbers separated by commas, or 'all'", "all")
    steps = optional if picked.lower() == "all" else [
        optional[int(n) - 1] for n in picked.replace(" ", "").split(",") if n.isdigit() and 0 < int(n) <= len(optional)]
    platforms = _csv(_ask(f"Ad platforms ({', '.join(PLATFORMS)})", ",".join(PLATFORMS)))
    budget = _ask("Monthly budget (leave empty and a budget is proposed)")
    print("\nModels:")
    catalog = provider_catalog(settings)
    for row in catalog:
        mark = "ready" if row["available"] else "not ready"
        print(f"  {row['id']:<10} {row['label']} [{mark}] {row['detail']}")
    research = _ask("Research model (reads and classifies)", settings.research_provider)
    writing = _ask("Writing model (ad copy and strategy)", settings.writing_provider)
    competitors = _csv(_ask("Competitor links you already know, comma separated (optional)"))
    seeds = [k.strip() for k in _ask("Search phrases to research, comma separated (optional)").split(",") if k.strip()]
    return RunConfig(
        url=url, steps=steps, platforms=platforms, budget=float(budget) if budget else None,
        currency=_ask("Currency", settings.default_currency), country=_ask("Country code", settings.default_country),
        competitors=competitors, seed_keywords=seeds, video=_ask("Video ad (auto, yes, no)", "auto"),
        media=[m.strip() for m in _ask("Your own photos or clips for the video: files or a folder, comma "
                                       "separated (optional)").split(",") if m.strip()],
        research_provider=research, writing_provider=writing,
    )


def cmd_run(args, settings: Settings) -> int:
    from .pipeline import run_pipeline

    if args.interactive:
        config = interactive(settings, args.url)
    else:
        if not args.url:
            print("Give a link, or use --interactive.", file=sys.stderr)
            return 2
        steps = [s for s in (_csv(args.steps) if args.steps else list(STEPS)) if s not in _csv(args.skip or "")]
        config = RunConfig(
            url=args.url, steps=steps, platforms=_csv(args.platforms), budget=args.budget,
            currency=args.currency or settings.default_currency, country=args.country or settings.default_country,
            language=args.language or "", competitors=args.competitor or [], max_competitors=args.max_competitors,
            seed_keywords=[k.strip() for k in (args.keywords or "").split(",") if k.strip()],
            keyword_csv=args.keyword_csv or "", customer_value=args.customer_value, video=args.video, media=args.media or [],
            engine=args.engine or "", rag=not args.no_rag,
            research_provider=args.research or args.provider or "", research_model=args.research_model or "",
            writing_provider=args.writing or args.provider or "", writing_model=args.writing_model or "",
        )

    marks = {"start": "▸", "done": "✓", "warn": "!", "error": "✗", "info": " "}

    def show(event: dict) -> None:
        print(f"[{event['t']:>6.1f}s] {marks.get(event['level'], ' ')} {event['message']}", flush=True)

    try:
        result = run_pipeline(config, settings, on_event=show)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    run_dir = result["meta"]["run_dir"]
    print(f"\nReport:  {run_dir}/report.html")
    print(f"Data:    {run_dir}/result.json")
    for step, message in result["errors"].items():
        print(f"Problem in {step}: {message}")
    if args.open:
        import webbrowser
        webbrowser.open(f"file://{run_dir}/report.html")
    return 1 if "brief" in result["errors"] else 0


def cmd_providers(_args, settings: Settings) -> int:
    print(f"Default research model: {settings.research_provider}   Default writing model: {settings.writing_provider}\n")
    from .rag import rag_status
    print(f"Retrieval (RAG): {rag_status(settings)['detail']}")
    print(f"Pipeline engine: {settings.engine}\n")
    for row in provider_catalog(settings):
        state = "READY    " if row["available"] else "NOT READY"
        kind = "local" if row["local"] else "paid "
        print(f"{state} {row['id']:<10} {kind}  {row['label']}")
        print(f"          model: {row['model'] or '(none set)'}")
        print(f"          {row['detail']}")
        if row["models"]:
            print(f"          installed: {', '.join(row['models'])}")
    return 0


def cmd_ask(args, settings: Settings) -> int:
    from .ask import ask, resolve_run
    from .rag import RagUnavailable

    try:
        run_dir = resolve_run(settings, args.run or "")
        result = ask(run_dir, args.question, settings)
    except (FileNotFoundError, RagUnavailable, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Run: {run_dir.name}\n")
    if result["rewritten_query"]:
        print(f"(Nothing found at first. Searched again for: {result['rewritten_query']})\n")
    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            where = source["url"] if source["kind"] == "page" else source["title"]
            print(f"  [{source['n']}] {where}  (similarity {source['score']:.2f})")
            print(f"      {source['quote'][:160]}")
    return 0


def cmd_graph(args, _settings: Settings) -> int:
    if args.which == "ask":
        from .ask import diagram
    else:
        from .graph import diagram
    print(diagram())
    return 0


def cmd_mcp(_args, _settings: Settings) -> int:
    try:
        from .mcp_server import main as serve_mcp
        serve_mcp()
    except ImportError:
        print('The MCP package is not installed. Run: pip install -e ".[mcp]"', file=sys.stderr)
        return 2
    return 0


def cmd_steps(_args, _settings: Settings) -> int:
    for step, (label, description, _module, deps) in STEPS.items():
        needs = f" (needs: {', '.join(deps)})" if deps else ""
        print(f"{step:<12} {label}: {description}{needs}")
    return 0


def cmd_serve(args, _settings: Settings) -> int:
    import uvicorn

    print(f"AdStrategist is running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    uvicorn.run("adstrategist.server:app", host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adstrategist", description="Marketing research and ad campaigns from one link.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="analyse a link and build the plan")
    run.add_argument("url", nargs="?", help="product, website or app link")
    run.add_argument("-i", "--interactive", action="store_true", help="answer questions instead of using flags")
    run.add_argument("--steps", help=f"comma separated, default all: {','.join(STEPS)}")
    run.add_argument("--skip", help="comma separated steps to leave out")
    run.add_argument("--platforms", default=",".join(PLATFORMS), help="comma separated: google,meta,tiktok")
    run.add_argument("--budget", type=float, help="monthly budget; omit to have one proposed")
    run.add_argument("--currency")
    run.add_argument("--country", help="two letter market, for example DE")
    run.add_argument("--language", help="force the copy language, for example de")
    run.add_argument("--competitor", action="append", help="a competitor link you already know (repeatable)")
    run.add_argument("--max-competitors", type=int, default=5)
    run.add_argument("--keywords", help="comma separated search phrases to research (overrides the automatic seeds)")
    run.add_argument("--keyword-csv", help="Google Keyword Planner export with real volumes and bids")
    run.add_argument("--customer-value", type=float, help="what one customer is worth, for payback checks")
    run.add_argument("--video", choices=["auto", "yes", "no"], default="auto")
    run.add_argument("--media", action="append", metavar="PATH",
                     help="your own photo, video clip or folder of them for the video ad (repeatable, shown in order)")
    run.add_argument("--provider", choices=PROVIDER_IDS, help="use one provider for both roles")
    run.add_argument("--research", choices=PROVIDER_IDS, help="provider for reading and classifying")
    run.add_argument("--writing", choices=PROVIDER_IDS, help="provider for ad copy and strategy")
    run.add_argument("--research-model")
    run.add_argument("--writing-model")
    run.add_argument("--engine", choices=["native", "langgraph"],
                     help="native: plain loop. langgraph: state graph that also reviews and revises ad copy")
    run.add_argument("--no-rag", action="store_true", help="do not build or use the retrieval index")
    run.add_argument("--open", action="store_true", help="open the report in the browser when done")
    run.set_defaults(func=cmd_run)

    sub.add_parser("providers", help="show which models are ready").set_defaults(func=cmd_providers)
    sub.add_parser("steps", help="list the pipeline steps").set_defaults(func=cmd_steps)
    ask = sub.add_parser("ask", help="ask a question about a finished run, answered with cited sources")
    ask.add_argument("question")
    ask.add_argument("--run", help="run id or folder; default is the most recent run")
    ask.set_defaults(func=cmd_ask)
    graph = sub.add_parser("graph", help="print a LangGraph graph as a Mermaid diagram")
    graph.add_argument("which", nargs="?", choices=["pipeline", "ask"], default="pipeline",
                       help="pipeline: the research pipeline. ask: the agentic question-answering graph")
    graph.set_defaults(func=cmd_graph)
    sub.add_parser("mcp", help="run as a Model Context Protocol server over stdio (for Claude Desktop and other "
                               "MCP clients)").set_defaults(func=cmd_mcp)
    serve = sub.add_parser("serve", help="start the local web interface")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args, Settings.from_env())


if __name__ == "__main__":
    raise SystemExit(main())
