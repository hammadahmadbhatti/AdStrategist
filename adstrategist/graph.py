"""The same pipeline as a LangGraph state graph.

Each pipeline step is a node. The graph adds two things the plain loop does not have:

  * a conditional edge after the product brief: if the page could not be read, the run is routed
    to an "abort" node instead of walking through steps that cannot work;
  * an evaluator-optimizer cycle around ad copy: campaigns -> review_copy -> (revise: back to
    campaigns with the reviewer's notes | accept: move on). The reviewer is deterministic (the
    claim check and the platform limits); the reviser is the writing model.

Step logic is shared with the native engine through `pipeline.execute_step`, so both engines
produce the same research. Graph state carries the results and the control data; live resources
(HTTP client, model router, vector index) stay in the run context.
"""
from __future__ import annotations

from typing import Annotated, Callable, TypedDict

MAX_COPY_REVISIONS = 1


def _merge(left: dict, right: dict) -> dict:
    return {**left, **right}


class RunState(TypedDict, total=False):
    results: Annotated[dict, _merge]     # step id -> that step's output
    errors: Annotated[dict, _merge]      # step id -> why it failed or was skipped
    path: Annotated[list, lambda a, b: a + b]  # nodes visited, in order
    copy_attempts: int                   # drafts of the ad copy written so far
    copy_mark: int                       # where this draft's rejected claims start
    revise: bool                         # reviewer's verdict on the latest draft
    copy_revisions: int                  # revisions actually requested


def build_graph(steps: list[str], run_step: Callable[[str], dict], review: Callable[[RunState], dict],
                abort: Callable[[RunState], dict], revisable: bool = True):
    """Compile the graph for a list of steps (already in pipeline order)."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(RunState)
    for step in steps:
        graph.add_node(step, lambda state, step=step: run_step(step, state))

    def after(step: str) -> str:
        position = steps.index(step)
        return steps[position + 1] if position + 1 < len(steps) else END

    graph.add_edge(START, steps[0])
    for step in steps:
        following = after(step)
        if step == "brief" and following != END:
            graph.add_node("abort", abort)
            graph.add_conditional_edges(
                "brief", lambda state: "page read" if "brief" in state.get("results", {}) else "page unreadable",
                {"page read": following, "page unreadable": "abort"})
            graph.add_edge("abort", END)
        elif step == "campaigns" and revisable:
            graph.add_node("review_copy", review)
            graph.add_edge("campaigns", "review_copy")
            graph.add_conditional_edges("review_copy", lambda state: "revise" if state.get("revise") else "accept",
                                        {"revise": "campaigns", "accept": following})
        else:
            graph.add_edge(step, following)
    return graph.compile()


def run_graph(ctx, steps: list[str], errors: dict, timings: dict, current: dict) -> dict:
    """Run the pipeline through LangGraph. Returns facts about the run for the report."""
    from .pipeline import execute_step
    from .steps.campaigns import review_copy

    def run_step(step: str, state: RunState) -> dict:
        update: dict = {"path": [step]}
        if step == "campaigns":
            update["copy_mark"] = len(ctx.rejected_claims)
            update["copy_attempts"] = state.get("copy_attempts", 0) + 1
        execute_step(ctx, step, errors, timings, current)
        if step in ctx.results:
            update["results"] = {step: ctx.results[step]}
        if step in errors:
            update["errors"] = {step: errors[step]}
        return update

    def review(state: RunState) -> dict:
        current["step"] = "campaigns"
        feedback = review_copy(ctx, state.get("copy_mark", 0)) if "campaigns" in ctx.results else ""
        revise = bool(feedback) and state.get("copy_attempts", 0) <= MAX_COPY_REVISIONS and ctx.llm.enabled("writing")
        ctx.copy_feedback = feedback if revise else ""
        if revise:
            ctx.emit("info", "Review: the claim check removed lines from this draft. Asking the model for a "
                             "second draft with the reviewer's notes.")
        elif feedback:
            ctx.emit("info", "Review: some lines were still rejected. Keeping the checked draft.")
        return {"path": ["review_copy"], "revise": revise,
                "copy_revisions": state.get("copy_revisions", 0) + (1 if revise else 0)}

    def abort(state: RunState) -> dict:
        for step in steps[1:]:  # record every remaining step as skipped, exactly as the native engine does
            execute_step(ctx, step, errors, timings, current)
        return {"path": ["abort"], "errors": dict(errors)}

    app = build_graph(steps, run_step, review, abort, revisable="campaigns" in steps)
    final = app.invoke({"results": {}, "errors": {}, "path": [], "copy_attempts": 0, "copy_revisions": 0},
                       {"recursion_limit": 4 * len(steps) + 10})
    ctx.copy_feedback = ""
    mermaid = readable(app.get_graph().draw_mermaid())
    (ctx.run_dir / "graph.mmd").write_text(mermaid, encoding="utf-8")
    return {"nodes": len(app.get_graph().nodes) - 2, "path": final.get("path", []),
            "copy_revisions": final.get("copy_revisions", 0), "diagram": "graph.mmd"}


def readable(mermaid: str) -> str:
    """LangGraph's diagram uses pale boxes with the page's text colour, which is unreadable on a dark
    page (GitHub dark theme). Pin a dark text colour on the pale boxes."""
    return (mermaid.replace("classDef default fill:#f2f0ff,line-height:1.2",
                            "classDef default fill:#f2f0ff,color:#1c2430,stroke:#8b7fe8,line-height:1.2")
                   .replace("classDef last fill:#bfb6fc", "classDef last fill:#bfb6fc,color:#1c2430"))


def diagram(steps: list[str] | None = None) -> str:
    """Mermaid source of the graph, generated from the code (used for the README)."""
    from .steps import STEPS

    noop = lambda *_a, **_k: {}  # noqa: E731
    return readable(build_graph(steps or list(STEPS), noop, noop, noop).get_graph().draw_mermaid())
