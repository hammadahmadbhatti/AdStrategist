"""Run configuration and the shared state steps read from and write to."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .claims import unsupported
from .config import Settings
from .llm import LLMRouter
from .textutil import LANGUAGE_NAMES
from .web_fetch import Fetcher, Page


class StepError(RuntimeError):
    """A step could not produce a result. The run continues with the remaining steps."""


@dataclass
class RunConfig:
    url: str
    steps: list[str] = field(default_factory=list)          # empty means every step
    platforms: list[str] = field(default_factory=lambda: ["google", "meta", "tiktok"])
    budget: float | None = None                              # monthly, in `currency`
    currency: str = "EUR"
    country: str = "DE"
    language: str = ""                                       # empty means: use the page's language
    competitors: list[str] = field(default_factory=list)     # URLs the user already knows
    max_competitors: int = 5
    seed_keywords: list[str] = field(default_factory=list)  # search phrases the user wants researched
    keyword_csv: str = ""                                    # Keyword Planner export with real volumes
    customer_value: float | None = None                      # what one customer is worth
    video: str = "auto"                                      # auto | yes | no
    media: list[str] = field(default_factory=list)           # your own photos and clips, in showing order
    engine: str = ""                                         # native | langgraph, empty means the .env default
    rag: bool = True                                         # use retrieval when it is available
    research_provider: str = ""
    research_model: str = ""
    writing_provider: str = ""
    writing_model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Context:
    config: RunConfig
    settings: Settings
    llm: LLMRouter
    fetcher: Fetcher
    run_dir: Path
    emit: Callable[[str, str], None]
    results: dict = field(default_factory=dict)
    own_page: Page | None = None
    sub_pages: list[Page] = field(default_factory=list)
    competitor_pages: dict[str, Page] = field(default_factory=dict)
    rejected_claims: list[dict] = field(default_factory=list)
    rag: object | None = None          # RagIndex when retrieval is available for this run
    copy_feedback: str = ""            # reviewer notes for a second copy draft (LangGraph engine)
    retrievals: int = 0

    def passages(self, query: str, k: int = 4, **filters) -> list:
        """Retrieved passages for a query, or [] when retrieval is off. Steps must cope with both."""
        if self.rag is None:
            return []
        self.retrievals += 1
        return self.rag.search(query, k=k, **filters)

    def index_page(self, page, site: str) -> None:
        if self.rag is not None:
            try:
                self.rag.add_page(page, site)
            except Exception as exc:  # a failed embedding call must not cost the run its page
                self.emit("warn", f"Could not index {page.url} for retrieval ({type(exc).__name__}).")

    def supported(self, text: str, where: str) -> bool:
        """False (and recorded) when a line of copy claims something the product page does not show."""
        pages = [p for p in [self.own_page, *self.sub_pages] if p]
        page_text = " ".join(f"{p.title} {p.description} {p.text}" for p in pages)
        reason = unsupported(text, self.results.get("brief", {}).get("offers") or
                             (self.own_page.offers if self.own_page else []), page_text)
        if reason:
            self.rejected_claims.append({"where": where, "text": text[:160], "reason": reason})
        return not reason

    @property
    def lang(self) -> str:
        if self.config.language:
            return self.config.language.lower()[:2]
        return (self.results.get("brief", {}).get("language") or "en")[:2]

    @property
    def language_name(self) -> str:
        return LANGUAGE_NAMES.get(self.lang, self.lang)

    @property
    def brief(self) -> dict:
        return self.results.get("brief", {})
