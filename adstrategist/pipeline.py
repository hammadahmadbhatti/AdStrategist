"""Runs the selected steps in order and writes the result files."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import __version__
from .config import PLATFORMS, Settings
from .context import Context, RunConfig, StepError
from .llm import LLMRouter, build_provider
from .report import build_blocks, to_html, to_markdown
from .steps import STEPS, resolve
from .textutil import normalize_url, registrable_domain, slugify
from .web_fetch import Fetcher

Event = Callable[[dict], None]


def validate(config: RunConfig) -> RunConfig:
    config.url = normalize_url(config.url)
    if "." not in registrable_domain(config.url):
        raise ValueError(f"'{config.url}' does not look like a web address.")
    config.platforms = [p for p in dict.fromkeys(p.lower() for p in config.platforms) if p]
    unknown = set(config.platforms) - set(PLATFORMS)
    if unknown:
        raise ValueError(f"Unknown platform(s): {', '.join(sorted(unknown))}. Choose from: {', '.join(PLATFORMS)}")
    if not config.platforms:
        raise ValueError("Select at least one ad platform.")
    if config.video not in ("auto", "yes", "no"):
        raise ValueError("video must be auto, yes or no.")
    if config.engine not in ("", "native", "langgraph"):
        raise ValueError("engine must be native or langgraph.")
    if config.budget is not None and config.budget <= 0:
        raise ValueError("The budget must be greater than zero.")
    config.country = (config.country or "US").upper()[:2]
    config.currency = (config.currency or "EUR").upper()[:3]
    config.steps = resolve(config.steps)
    return config


def build_router(settings: Settings, emit: Callable[[str, str], None], research_provider: str = "",
                 research_model: str = "", writing_provider: str = "", writing_model: str = "") -> LLMRouter:
    """The two-role model router, with unavailable models downgraded and the reason reported."""
    research = build_provider(research_provider or settings.research_provider, settings, research_model)
    writing = build_provider(writing_provider or settings.writing_provider, settings, writing_model)
    same = (research.id, research.model) == (writing.id, writing.model)
    status = research.status()
    if not status.available:
        emit("warn", f"The research model ({research.label}) is not available: {status.detail} "
                     "Continuing with rule-based output for that role.")
        research = build_provider("offline", settings)
    status = status if same else writing.status()
    if not status.available:
        if research.id != "offline" and not same:
            emit("warn", f"The writing model ({writing.label}) is not available: {status.detail} "
                         f"Using the research model ({research.label}) for writing instead.")
            writing = research
        else:
            if not same:
                emit("warn", f"The writing model ({writing.label}) is not available: {status.detail} "
                             "Continuing with rule-based output for that role.")
            writing = build_provider("offline", settings)
    return LLMRouter(research, writing, emit)


def execute_step(ctx: Context, step: str, errors: dict, timings: dict, current: dict) -> None:
    """Run one step with the shared safety net. Both engines call this, so they behave the same."""
    label, _, module, deps = STEPS[step]
    current["step"] = step
    missing = [d for d in deps if d not in ctx.results]
    if missing:
        errors[step] = f"Skipped because {', '.join(STEPS[m][0] for m in missing)} did not finish."
        ctx.emit("warn", errors[step])
        return
    ctx.emit("start", label)
    step_started = time.time()
    try:
        ctx.results[step] = module.run(ctx)
        errors.pop(step, None)
        ctx.emit("done", label)
    except StepError as exc:
        errors[step] = str(exc)
        ctx.emit("error", str(exc))
    except Exception as exc:  # one broken step must not lose the rest of the run
        errors[step] = f"{type(exc).__name__}: {exc}"
        ctx.emit("error", f"{label} failed: {errors[step]}")
    timings[step] = round(timings.get(step, 0.0) + time.time() - step_started, 1)


def index_report(ctx: Context, blocks: list) -> None:
    """Put the finished report into the index too, so questions can be answered from the analysis
    (budget, keywords, gaps) and not only from the crawled pages."""
    section, lines = "Overview", []

    def flush() -> None:
        if lines:
            ctx.rag.add_text("\n".join(lines).replace("**", "").replace("`", ""), url=f"report#{slugify(section)}", site="report",
                             title=f"Report: {section}", kind="report")

    for block in blocks:
        kind = block[0]
        if kind == "h2":
            flush()
            section, lines = block[1], [block[1] + "."]
        elif kind in ("h1", "h3", "p", "note", "warn"):
            lines.append(str(block[1]))
        elif kind == "ul":
            lines += [f"- {item}" for item in block[1]]
        elif kind == "table":
            headers = block[1]
            lines += ["; ".join(f"{h}: {c}" for h, c in zip(headers, row)) for row in block[2]]
    flush()


def run_pipeline(config: RunConfig, settings: Settings | None = None, on_event: Event | None = None,
                 fetcher: Fetcher | None = None, run_dir: Path | None = None, embeddings=None) -> dict:
    settings = settings or Settings.from_env()
    config.engine = (config.engine or settings.engine or "native").lower()
    config = validate(config)
    started = time.time()
    events: list[dict] = []
    current = {"step": "setup"}

    def emit(level: str, message: str) -> None:
        event = {"t": round(time.time() - started, 1), "level": level, "step": current["step"], "message": message}
        events.append(event)
        if on_event:
            on_event(event)

    llm = build_router(settings, emit, config.research_provider, config.research_model,
                       config.writing_provider, config.writing_model)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = run_dir or settings.output_dir / f"{slugify(registrable_domain(config.url))}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    owns_fetcher = fetcher is None
    ctx = Context(config=config, settings=settings, llm=llm, fetcher=fetcher or Fetcher(), run_dir=run_dir, emit=emit)

    rag_info = {"enabled": False, "detail": "Retrieval was switched off for this run."}
    if config.rag and settings.rag != "off":
        from .rag import RagIndex, RagUnavailable
        try:
            ctx.rag = RagIndex.open(run_dir / "index", settings, embeddings)
        except RagUnavailable as exc:
            rag_info["detail"] = str(exc)
            emit("info", f"Retrieval is off for this run: {exc} Page text is trimmed instead.")
        except Exception as exc:  # a broken vector store must never block a run
            rag_info["detail"] = f"The vector index could not be opened ({type(exc).__name__})."
            emit("warn", rag_info["detail"] + " Page text is trimmed instead.")

    errors: dict[str, str] = {}
    timings: dict[str, float] = {}
    graph_info: dict = {}
    try:
        if config.engine == "langgraph":
            from .graph import run_graph
            graph_info = run_graph(ctx, config.steps, errors, timings, current)
        else:
            for step in config.steps:
                execute_step(ctx, step, errors, timings, current)
    finally:
        if owns_fetcher:
            ctx.fetcher.close()

    current["step"] = "finish"
    if ctx.rag is not None:
        rag_info = {**ctx.rag.stats(), "retrievals": ctx.retrievals,
                    "detail": f"Pages were embedded locally with {settings.embed_model} into a Chroma index."}
    result = {
        "meta": {
            "version": __version__, "url": config.url, "started": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
            "seconds": round(time.time() - started, 1), "run_dir": str(run_dir), "run_id": run_dir.name,
            "config": config.to_dict(), "models": llm.describe(), "model_usage": llm.usage(),
            "steps": config.steps, "timings": timings, "rejected_claims": ctx.rejected_claims,
            "engine": config.engine, "graph": graph_info, "rag": rag_info,
        },
        "results": ctx.results, "errors": errors, "events": events,
    }
    blocks = build_blocks(result)
    if ctx.rag is not None:
        try:
            index_report(ctx, blocks)
            result["meta"]["rag"]["chunks"] = ctx.rag.count()
        except Exception as exc:
            emit("warn", f"The report could not be added to the index ({type(exc).__name__}).")
    (run_dir / "report.md").write_text(to_markdown(blocks), encoding="utf-8")
    (run_dir / "report.html").write_text(to_html(blocks, title=f"Ad strategy: {registrable_domain(config.url)}"),
                                         encoding="utf-8")
    result["meta"]["seconds"] = round(time.time() - started, 1)
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    emit("done", f"Finished in {result['meta']['seconds']} s. Files are in {run_dir}")
    return result
