"""Pipeline steps. Each module exposes run(ctx) -> dict."""
from __future__ import annotations

from . import ad_intel, brief, budget, campaigns, competitors, gaps, keywords, summary, traffic, video

# id -> (label, what the user gets, module, steps it needs first)
STEPS = {
    "brief": ("Product brief", "Reads the page: what it is, who it is for, selling points.", brief, []),
    "competitors": ("Competitors", "Finds competitors through web search and reads their sites.", competitors, ["brief"]),
    "ad_intel": ("Competitor ads", "Which ad platforms competitors use, with links to their live ads.", ad_intel, ["competitors"]),
    "keywords": ("Keyword research", "Real search phrases grouped by intent, plus negatives.", keywords, ["brief"]),
    "gaps": ("Market gaps", "Topics, offers and channels competitors cover and you do not.", gaps, ["competitors"]),
    "traffic": ("Traffic potential", "Click estimates when real search volumes are supplied.", traffic, ["keywords"]),
    "budget": ("Budget plan", "Proposes or splits a monthly budget for value for money.", budget, ["brief"]),
    "campaigns": ("Ad campaigns", "Ready-to-review campaigns and import files per platform.", campaigns, ["keywords"]),
    "video": ("Video ad", "Vertical video ad rendered locally from the site's images.", video, ["brief"]),
    "summary": ("Summary", "Executive summary and next steps.", summary, ["brief"]),
}
ALWAYS = ["brief", "summary"]


def resolve(selected: list[str] | None) -> list[str]:
    """Selected steps plus whatever they depend on, in pipeline order."""
    wanted = set(selected or STEPS) | set(ALWAYS)
    unknown = wanted - set(STEPS)
    if unknown:
        raise ValueError(f"Unknown step(s): {', '.join(sorted(unknown))}. Choose from: {', '.join(STEPS)}")
    changed = True
    while changed:
        changed = False
        for step in list(wanted):
            for dep in STEPS[step][3]:
                if dep not in wanted:
                    wanted.add(dep)
                    changed = True
    return [s for s in STEPS if s in wanted]
