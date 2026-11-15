"""Ask questions about a finished run: agentic RAG as a LangGraph state graph.

    retrieve -> grade --(relevant)--------------------------> generate -> verify --(figures ok)--> done
                  |                                                         |
                  |--(nothing relevant, first try)--> rewrite query -> retrieve   (figures not in sources)
                  |                                                         |
                  '--(still nothing)--> refuse                              '--> generate again, once, with the correction

The graph decides what to do next from what it finds: search again with a rewritten query, refuse,
or send a draft back because its figures are not in the passages it cites. Generation itself is a
LangChain LCEL chain (prompt | model | parser). The model is reached through the role router, so
local/paid switching and fallbacks apply here too.

No answer is ever generated from nothing: when retrieval finds no relevant passage, even after a
rewrite, the graph refuses instead of asking the model to answer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict

from .claims import missing_numbers
from .config import Settings
from .rag import Passage, RagIndex, RagUnavailable
from .textutil import LANGUAGE_NAMES

NOT_FOUND = "The research for this run does not cover that."
SYSTEM = ("You answer questions about one company's marketing research. Use only the numbered sources. "
          "After each statement, cite the source it came from like [1]. Give figures exactly as the sources "
          "give them. If the sources do not contain the answer, reply exactly: " + NOT_FOUND)
MAX_REWRITES = 1


class AskState(TypedDict, total=False):
    question: str
    query: str            # what is searched for: the question, or its rewrite
    rewrites: int
    passages: list
    answer: str
    wrong_figures: list
    corrected: bool
    trace: list           # nodes visited, for transparency and tests


def resolve_run(settings: Settings, run: str = "") -> Path:
    """A run folder from an id, a path, or nothing (the most recent run)."""
    base = settings.output_dir
    if run:
        path = Path(run).expanduser()
        path = path if path.is_dir() else base / run
        if not (path / "result.json").exists():
            raise FileNotFoundError(f"No finished run found at {run}.")
        return path
    runs = sorted((p for p in base.iterdir() if (p / "result.json").exists()), key=lambda p: p.stat().st_mtime) \
        if base.is_dir() else []
    if not runs:
        raise FileNotFoundError("There are no finished runs yet.")
    return runs[-1]


def _format_sources(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{i}] ({p.title or p.site}) {p.text}" for i, p in enumerate(passages, 1))


def unverified_figures(answer: str, passages: list[Passage], question: str) -> list[str]:
    """Figures in an answer that none of the retrieved passages contain. Citation markers, list
    numbering and figures taken from the question itself are not counted."""
    cited = [passages[int(n) - 1] for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(passages)]
    text = re.sub(r"\[\d+\]", " ", answer)
    text = re.sub(r"(?m)^\s*\d{1,2}[.)]\s", " ", text)
    text = re.sub(r"(?i)\bsources? \d+\b", " ", text)
    # When the answer cites sources, its figures must be in those, not merely somewhere in the pile.
    source = " ".join(p.text for p in (cited or passages)) + " " + question
    return [n for n in missing_numbers(text, source) if not re.fullmatch(r"\d", n)]


def answer_chain(llm):
    """LCEL chain for one draft. Input: {"sources": str, "question": str}. Output: the answer text."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Sources:\n{sources}\n\nQuestion: {question}\n\nAnswer in plain text and cite each source "
                  "you use by its number in square brackets, like [1]."),
    ])

    def generate(prompt_value) -> str:
        system, human = (m.content for m in prompt_value.to_messages())
        return llm.ask_text("answer question", human, role="writing", system=system, max_tokens=500) or ""

    return prompt | RunnableLambda(generate) | StrOutputParser()


def build_graph(index, llm, min_score: float, language: str = "en", k: int = 5):
    """Compile the agentic RAG graph. Input state: {"question": str}."""
    from langgraph.graph import END, START, StateGraph

    chain = answer_chain(llm)

    def retrieve(state: AskState) -> dict:
        query = state.get("query") or state["question"]
        passages = [p for p in index.search(query, k=k) if p.score >= min_score]
        return {"query": query, "passages": passages, "trace": state.get("trace", []) + ["retrieve"]}

    def grade(state: AskState) -> str:
        if state["passages"]:
            return "relevant"
        return "rewrite" if state.get("rewrites", 0) < MAX_REWRITES and llm.enabled("research") else "nothing found"

    def rewrite(state: AskState) -> dict:
        """The first search found nothing. Let the model restate the question the way the pages would
        phrase it (their language, their vocabulary) and search once more."""
        better = llm.ask_text(
            "rewrite question",
            f"A search over a company's marketing research found nothing for this question:\n{state['question']}\n\n"
            f"Rewrite this question as a short search query in {LANGUAGE_NAMES.get(language, 'English')}, using the "
            "words such documents would use. Keep the topic exactly the same. Reply with the query only.",
            role="research", max_tokens=60) or ""
        better = " ".join(better.strip().strip('"').split())[:200]
        return {"query": better or state["question"], "rewrites": state.get("rewrites", 0) + 1,
                "trace": state.get("trace", []) + ["rewrite"]}

    def generate(state: AskState) -> dict:
        question = state["question"]
        if state.get("wrong_figures"):
            question += ("\n(Your last answer used figures that are not in the sources: "
                         f"{', '.join(state['wrong_figures'])}. Copy every figure exactly from the sources.)")
        answer = chain.invoke({"sources": _format_sources(state["passages"]), "question": question})
        return {"answer": answer.replace("**", "").strip() or NOT_FOUND,
                "corrected": bool(state.get("wrong_figures")), "trace": state.get("trace", []) + ["generate"]}

    def verify(state: AskState) -> dict:
        wrong = [] if NOT_FOUND in state["answer"] else unverified_figures(state["answer"], state["passages"],
                                                                           state["question"])
        return {"wrong_figures": wrong, "trace": state.get("trace", []) + ["verify"]}

    def after_verify(state: AskState) -> str:
        return "correct" if state["wrong_figures"] and not state.get("corrected") else "done"

    def refuse(state: AskState) -> dict:
        return {"answer": NOT_FOUND, "wrong_figures": [], "trace": state.get("trace", []) + ["refuse"]}

    graph = StateGraph(AskState)
    for name, node in (("retrieve", retrieve), ("rewrite", rewrite), ("generate", generate), ("verify", verify),
                       ("refuse", refuse)):
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", grade, {"relevant": "generate", "rewrite": "rewrite",
                                                    "nothing found": "refuse"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", after_verify, {"correct": "generate", "done": END})
    graph.add_edge("refuse", END)
    return graph.compile()


def diagram() -> str:
    """Mermaid source of the question-answering graph, generated from the code."""
    class _NoIndex:
        def search(self, *_a, **_k):
            return []

    class _NoModel:
        def enabled(self, *_a):
            return False

        def ask_text(self, *_a, **_k):
            return ""

    from .graph import readable
    return readable(build_graph(_NoIndex(), _NoModel(), 0.0).get_graph().draw_mermaid())


def ask(run_dir: Path, question: str, settings: Settings, llm=None, embeddings=None) -> dict:
    """Answer a question from a run's index. Returns the answer, whether it was found, and the sources."""
    question = " ".join(question.split())
    if not question:
        raise ValueError("Ask a question.")
    index_dir = run_dir / "index"
    if not index_dir.is_dir():
        raise RagUnavailable("This run has no retrieval index (it was made with retrieval off). Run it again.")
    index = RagIndex.open(index_dir, settings, embeddings)
    if llm is None:
        from .pipeline import build_router
        llm = build_router(settings, lambda _level, _message: None)
    if not llm.enabled("writing"):
        raise RagUnavailable("No model is available to write the answer. Start Ollama or configure a provider.")
    try:
        language = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["results"]["brief"]["language"]
    except (OSError, ValueError, KeyError):
        language = "en"

    state = build_graph(index, llm, settings.rag_min_score, language).invoke({"question": question},
                                                                            {"recursion_limit": 20})
    passages: list[Passage] = state.get("passages", [])
    answer = state.get("answer") or NOT_FOUND
    wrong = state.get("wrong_figures", [])
    found = bool(passages) and NOT_FOUND not in answer
    if wrong and found:
        answer += (f"\n\nCheck the figures: {', '.join(wrong)} do not appear in the sources. "
                   "Read the quoted passages below.")
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(passages)})
    if found and not cited:
        # An answer that cites nothing cannot be checked. Say which passages were supplied instead.
        cited = list(range(1, len(passages) + 1))
        answer += "\n\n(The model did not cite its sources. The passages it was given are listed below.)"
    sources = [{"n": n, "title": passages[n - 1].title, "url": passages[n - 1].url, "site": passages[n - 1].site,
                "kind": passages[n - 1].kind, "quote": passages[n - 1].text, "score": passages[n - 1].score}
               for n in (cited if found else [])]
    rewritten = state.get("query") if state.get("rewrites") and state.get("query") != question else ""
    return {"question": question, "answer": answer, "found": found, "sources": sources,
            "figures_verified": not wrong, "retrieved": len(passages), "trace": state.get("trace", []),
            "rewritten_query": rewritten, "model": llm.describe()["writing"]}
