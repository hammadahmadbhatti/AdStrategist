"""Turns a run result into a readable report (Markdown and standalone HTML) from one block list."""
from __future__ import annotations

import html
import re

from .config import PLATFORMS
from .web_fetch import OFFER_LABELS, TRACKER_LABELS

Block = tuple  # ("h1"|"h2"|"h3"|"p"|"note"|"warn", text) | ("ul", [text]) | ("table", [headers], [[cells]])


def _rng(pair, unit: str = "") -> str:
    low, high = pair
    fmt = (lambda v: f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:,.2f}".rstrip("0").rstrip("."))
    return f"{fmt(low)} to {fmt(high)}{unit}"


def _money(value, currency: str) -> str:
    return "not set" if value is None else f"{value:,.2f} {currency}".replace(".00 ", " ")


def _join(items, empty: str = "none found") -> str:
    items = [str(i) for i in items if i]
    return ", ".join(items) if items else empty


def _evidence(finding: dict) -> str:
    evidence = finding.get("evidence")
    if not evidence:
        return "none retrieved"
    return f"“{evidence['quote']}” ([{evidence['site']}]({evidence['url']}), similarity {evidence['score']:.2f})"


def _own_coverage(finding: dict) -> str:
    own = finding.get("closest_on_your_site")
    if not own:
        return "nothing close"
    label = "possibly covered in other words" if finding.get("possibly_covered") else "not covered"
    return f"{label} (closest passage {own['score']:.2f})"


def build_blocks(result: dict) -> list[Block]:
    meta, r, errors = result["meta"], result["results"], result["errors"]
    blocks: list[Block] = []
    brief = r.get("brief", {})
    blocks.append(("h1", f"Ad strategy for {brief.get('brand') or meta['url']}"))
    models = meta["models"]
    blocks.append(("p", f"Source: {meta['url']} · Run {meta['started'].replace('T', ' ')} · {meta['seconds']} s · "
                        f"Research model: {models['research']['label']}"
                        f"{' (' + models['research']['model'] + ')' if models['research']['model'] else ''} · "
                        f"Writing model: {models['writing']['label']}"
                        f"{' (' + models['writing']['model'] + ')' if models['writing']['model'] else ''}"))
    blocks.append(("note", "Every number in this report comes from a named source: your page, a search result, "
                           "autocomplete, your keyword CSV, or a benchmark assumption that is labelled as one. "
                           "Models only write and summarise."))
    for step, message in errors.items():
        blocks.append(("warn", f"{step}: {message}"))

    if "summary" in r:
        blocks += [("h2", "Summary"), ("p", r["summary"]["text"])]
        if r["summary"]["next_steps"]:
            blocks += [("h3", "Next steps"), ("ul", r["summary"]["next_steps"])]

    if brief:
        blocks += [("h2", "Product brief"), ("p", brief.get("one_liner", ""))]
        blocks.append(("table", ["Field", "Value"], [
            ["Brand", brief.get("brand", "")], ["Category", brief.get("category", "")],
            ["Business type", brief.get("business_type", "").replace("_", " ")],
            ["Location", brief.get("location") or "not stated on the page"], ["Language", brief.get("language", "")],
            ["Audience", _join(brief.get("audience", []), "not identified")],
            ["Customer problems", _join(brief.get("pains", []), "not identified")],
            ["Prices on page", _join(brief.get("prices", []))],
            ["Offers on page", _join(OFFER_LABELS.get(o, o) for o in brief.get("offers", []))],
            ["Tracking tags", _join(TRACKER_LABELS.get(t, t) for t in brief.get("trackers", []))],
        ]))
        if brief.get("usps"):
            blocks += [("h3", "Selling points"), ("ul", brief["usps"])]
        for warning in brief.get("warnings", []):
            blocks.append(("warn", warning))

    if "competitors" in r:
        comp = r["competitors"]
        blocks.append(("h2", "Competitors"))
        if comp["competitors"]:
            blocks.append(("table", ["Competitor", "Positioning", "Prices shown", "Offers", "Found via"], [
                [f"[{c['name']}]({c['url']})", c["positioning"] if not c["error"] else f"Could not read: {c['error']}",
                 _join(c["prices"][:3], "none"), _join((OFFER_LABELS.get(o, o) for o in c["offers"]), "none"),
                 "you" if c["source"] == "you" else _join(c["found_via"][:2], "search")
                 + (" (autocomplete)" if c["source"] == "autocomplete" else "")]
                for c in comp["competitors"]]))
            claims = [f"**{c['name']}**: {_join(c['claims'] or c['usps'][:3], 'no claims read')}"
                      for c in comp["competitors"] if not c["error"]]
            if claims:
                blocks += [("h3", "What they lead with"), ("ul", claims)]
        for note in comp["notes"]:
            blocks.append(("warn", note))

    if "ad_intel" in r:
        intel = r["ad_intel"]
        blocks.append(("h2", "Competitor advertising"))
        if intel["competitors"]:
            blocks.append(("table", ["Competitor", "Ad tags on site", "See their live ads"], [
                [c["name"], _join(c["tags"], "none detected") if c["readable"] else "site not readable",
                 f"[Meta]({c['links']['meta']}) · [Google]({c['links']['google']}) · [TikTok]({c['links']['tiktok']})"]
                for c in intel["competitors"]]))
            if intel["readable"]:
                blocks.append(("p", "Set up to advertise, out of " + str(intel["readable"]) + " readable sites: " +
                               ", ".join(f"{PLATFORMS[p].split(' (')[0]} {n}" for p, n in intel["summary"].items()) + "."))
            for c in intel["competitors"]:
                ads = c.get("meta_ads")
                if ads and ads["longest_running"]:
                    blocks.append(("h3", f"{c['name']}: longest-running Meta ads ({ads['count']} active)"))
                    blocks.append(("ul", [f"Since {a['started']}: {a['headline'] or a['text'][:120]}"
                                          for a in ads["longest_running"]]))
                elif ads and ads["error"]:
                    blocks.append(("warn", ads["error"]))
        blocks.append(("ul", intel["notes"]))

    if "keywords" in r:
        kw = r["keywords"]
        blocks.append(("h2", "Keyword research"))
        blocks.append(("p", f"{len(kw['keywords'])} keywords in {len(kw['clusters'])} groups, from seeds: "
                            f"{_join(kw['seeds'])}."))
        if kw["clusters"]:
            headers = ["Group", "Keywords", "Main intent", "Relative interest"] + (["Monthly searches"] if kw["has_volumes"] else []) + ["Examples"]
            blocks.append(("table", headers, [
                [c["name"], c["count"], c["main_intent"], c["interest"]] +
                ([f"{c['volume']:,.0f}"] if kw["has_volumes"] else []) + [_join(c["top"][:4])]
                for c in kw["clusters"]]))
        top = kw["keywords"][:25]
        if top:
            headers = ["Keyword", "Intent", "Interest"] + (["Searches", "Bid range"] if kw["has_volumes"] else [])
            blocks.append(("h3", "Top keywords"))
            blocks.append(("table", headers, [
                [k["keyword"], k["intent"], k["interest"]] +
                ([f"{k['volume']:,.0f}" if k.get("volume") else "",
                  _rng((k.get("cpc_low") or k.get("cpc_high"), k.get("cpc_high") or k.get("cpc_low")))
                  if (k.get("cpc_low") or k.get("cpc_high")) else ""] if kw["has_volumes"] else [])
                for k in top]))
        if kw["negative_keywords"]:
            blocks.append(("p", f"**Negative keywords** (block these to save budget): {_join(kw['negative_keywords'])}"))
        if kw["conquest"]:
            blocks.append(("p", f"**Competitor-name ideas** (check trademark rules before bidding): {_join(kw['conquest'])}"))
        blocks.append(("ul", kw["notes"]))

    if "gaps" in r:
        gaps = r["gaps"]
        blocks.append(("h2", "Market gaps"))
        rec = gaps["recommendation"]
        if rec.get("positioning"):
            blocks.append(("p", rec["positioning"]))
        if rec.get("angles"):
            blocks += [("h3", "Ad angles"), ("ul", rec["angles"])]
        if rec.get("quick_wins"):
            blocks += [("h3", "Quick wins"), ("ul", rec["quick_wins"])]
        if gaps["topics"]:
            blocks.append(("h3", "Topics competitors lead with that your page does not mention"))
            if any(t.get("evidence") for t in gaps["topics"]):
                blocks.append(("table", ["Topic", "Used by", "Evidence (retrieved passage)", "On your site"], [
                    [t["topic"], _join(t["used_by"]), _evidence(t), _own_coverage(t)] for t in gaps["topics"]]))
            else:
                blocks.append(("table", ["Topic", "Used by"], [[t["topic"], _join(t["used_by"])] for t in gaps["topics"]]))
        facts = [f"Competitors show {g['offer']} and you do not ({_join(g['used_by'])})."
                 + (f" Evidence: {_evidence(g)}" if g.get("evidence") else "") for g in gaps["offer_gaps"]]
        facts += [f"You show {a['offer']}; {a['competitors_with_it']} of {gaps['compared_with']} competitors do." for a in gaps["advantages"]]
        facts += [g["text"] for g in gaps["channel_gaps"]]
        facts += [f"Keyword group “{g['cluster']}” ({g['keywords']} keywords): your page never uses the word(s) "
                  f"{_join(g['missing_words'])}." for g in gaps["keyword_gaps"]]
        if facts:
            blocks += [("h3", "Findings"), ("ul", facts)]
        for note in gaps["notes"]:
            blocks.append(("warn", note))

    if "traffic" in r:
        traffic = r["traffic"]
        blocks.append(("h2", "Traffic potential"))
        if traffic["available"]:
            rows = [["Monthly searches across the keyword list", f"{traffic['monthly_searches']:,}"],
                    ["Of which buyer intent", f"{traffic['buyer_intent_searches']:,}"],
                    ["Paid clicks per month", _rng(traffic["paid_clicks"])],
                    ["Organic clicks per month (ceiling)", _rng(traffic["organic_clicks_ceiling"])]]
            if traffic["paid_cost"]:
                rows.append(["Cost to capture those paid clicks", _rng(traffic["paid_cost"]["monthly"])])
            blocks.append(("table", ["Measure", "Value"], rows))
            blocks.append(("ul", list(traffic["assumptions"].values()) + traffic["notes"]))
            blocks.append(("table", ["Group", "Monthly searches", "Buyer intent"],
                           [[c["cluster"], f"{c['volume']:,}", f"{c['buyer_volume']:,}"] for c in traffic["by_cluster"]]))
        else:
            if traffic["ranking"]:
                blocks.append(("table", ["Group", "Keywords", "Main intent", "Relative interest"],
                               [[c["cluster"], c["keywords"], c["main_intent"], c["interest"]] for c in traffic["ranking"]]))
            blocks.append(("ul", traffic["notes"]))

    if "budget" in r:
        b = r["budget"]
        cur = b["currency"]
        blocks.append(("h2", "Budget plan"))
        lead = (f"No budget was given, so the plan uses the proposed Test tier: **{b['monthly_budget']:,.0f} {cur} per month**."
                if b["budget_was_proposed"] else f"Your budget: **{b['monthly_budget']:,.0f} {cur} per month**.")
        blocks.append(("p", lead))
        blocks.append(("table", ["Platform", "Share", "Monthly", "Daily", "Cost per click", "Clicks", "Conversions",
                                 "Cost per conversion"] + (["Payback"] if b["customer_value"] else []), [
            [r_["label"] + (" ★" if r_["platform"] == b["best_value"] and len(b["plan"]) > 1 else ""),
             f"{r_['share']*100:.0f}%", _money(r_["monthly"], cur), _money(r_["daily"], cur),
             f"{_rng(r_['cpc'])} ({r_['cpc_source']})", _rng(r_["clicks"]), _rng(r_["conversions"]),
             _rng(r_["cost_per_conversion"])] + ([r_.get("payback", "")] if b["customer_value"] else [])
            for r_ in b["plan"]]))
        if len(b["plan"]) > 1:
            blocks.append(("p", "★ best expected value for money (lowest cost per conversion at the midpoint)."))
        blocks.append(("h3", "Budget tiers"))
        blocks.append(("table", ["Tier", "Monthly", "Platforms", "Purpose"], [
            [t["name"], f"{t['monthly']:,} {cur}", _join(PLATFORMS[p].split(" (")[0] for p in t["platforms"]), t["purpose"]]
            for t in b["tiers"]]))
        if b["weighting_reasons"]:
            blocks += [("h3", "Why this split"), ("ul", b["weighting_reasons"])]
        for warning in b["warnings"]:
            blocks.append(("warn", warning))
        blocks += [("h3", "How to spend it"), ("ul", b["phases"]), ("ul", b["notes"])]

    if "campaigns" in r:
        camp = r["campaigns"]
        cur = r.get("budget", {}).get("currency", meta["config"]["currency"])
        blocks.append(("h2", "Ad campaigns"))
        blocks.append(("ul", camp["notes"]))
        for c in camp["campaigns"]:
            funded = "" if c.get("funded", True) else " (not funded at this budget, keep for later)"
            blocks.append(("h3", f"{PLATFORMS[c['platform']]}: {c['name']}{funded}"))
            if c["platform"] == "google":
                blocks.append(("p", f"Daily budget {_money(c['daily_budget'], cur)} · {c['bid_strategy']} · "
                                    f"{c['location']} · language {c['language']}. {c['bid_note']}"))
                for group in c["ad_groups"]:
                    ad = group["ad"]
                    blocks.append(("p", f"**Ad group: {group['name']}**{'' if group['ready'] else ' (needs more copy before launch)'}"))
                    blocks.append(("table", ["Element", "Content"], [
                        ["Keywords", _join(dict.fromkeys(k["text"] for k in group["keywords"]))],
                        ["Headlines", " | ".join(ad["headlines"])], ["Descriptions", " | ".join(ad["descriptions"])],
                        ["Landing page", ad["final_url"]]]))
                extras = []
                if c["negative_keywords"]:
                    extras.append(f"Negative keywords: {_join(c['negative_keywords'])}")
                if c["sitelinks"]:
                    extras.append(f"Sitelinks: {_join(s['text'] for s in c['sitelinks'])}")
                if c["callouts"]:
                    extras.append(f"Callouts: {_join(c['callouts'])}")
                if extras:
                    blocks.append(("ul", extras))
            elif c["platform"] == "meta":
                blocks.append(("p", f"Objective {c['objective']} · daily budget {_money(c['daily_budget'], cur)} · "
                                    f"ad sets: {_join(s['name'] for s in c['ad_sets'])}"))
                blocks.append(("table", ["Ad", "Primary text", "Headline", "Description", "Button", "Visual"], [
                    [a["angle"], a["primary_text"].replace("\n", " / "), a["headline"], a["description"],
                     a["call_to_action"], a["visual"]] for a in c["ads"]]))
                if c["pixel_note"]:
                    blocks.append(("warn", c["pixel_note"]))
            elif c["platform"] == "tiktok":
                blocks.append(("p", f"Objective {c['objective']} · daily budget {_money(c['daily_budget'], cur)} · "
                                    f"{c['ad_group']['format']}"))
                blocks.append(("table", ["Video", "Hook", "Caption", "Script", "Hashtags", "Button"], [
                    [a["name"], a["hook"], a["ad_text"], " / ".join(a["script"]),
                     " ".join("#" + t for t in a["hashtags"]), a["call_to_action"]] for a in c["ads"]]))
                if c["pixel_note"]:
                    blocks.append(("warn", c["pixel_note"]))
        blocks.append(("p", "Files: " + _join(f"`{f}`" for f in camp["files"])))

    if "video" in r:
        video = r["video"]
        blocks.append(("h2", "Video ad"))
        if video["rendered"]:
            blocks.append(("p", f"Rendered `{video['file']}`: {video['seconds']} s, {video['format']}, "
                                f"{video['pictures']}, voice: {video['voice'] or 'none (captions only)'}."))
        else:
            blocks.append(("p", f"No video was rendered. {video['reason']}"))
        if video["scenes"]:
            blocks.append(("table", ["Scene", "Picture", "On screen", "Voice"],
                           [[i, s.get("picture", ""), s["caption"] or "(no caption)", s["voiceover"]]
                            for i, s in enumerate(video["scenes"], 1)]))
        if video["notes"]:
            blocks.append(("ul", video["notes"]))

    rejected = meta.get("rejected_claims", [])
    if rejected:
        blocks.append(("h2", "Copy removed by the claim check"))
        blocks.append(("p", f"{len(rejected)} lines were removed before they reached a campaign because they "
                            "promised something your page does not show. If one of them is true, add it to your "
                            "page and run again."))
        blocks.append(("table", ["Where", "Line", "Why"], [[c["where"], c["text"], c["reason"]] for c in rejected[:20]]))

    usage = meta["model_usage"]
    blocks.append(("h2", "Run details"))
    if usage:
        blocks.append(("table", ["Model", "Calls", "Failed", "Input tokens", "Output tokens", "Seconds"], [
            [name, u["calls"], u["failed"], f"{u['input_tokens']:,}", f"{u['output_tokens']:,}", u["seconds"]]
            for name, u in usage.items()]))
    else:
        blocks.append(("p", "No model calls were made. All output is rule based."))
    rag = meta.get("rag") or {}
    if rag.get("enabled"):
        blocks.append(("p", f"Retrieval: {rag['chunks']} passages from {rag['documents']} pages, embedded locally "
                            f"with {rag['embedding_model']} into a {rag['vector_store']} index, queried "
                            f"{rag.get('retrievals', 0)} times during the run."))
    elif rag.get("detail"):
        blocks.append(("p", f"Retrieval was not used. {rag['detail']}"))
    graph = meta.get("graph") or {}
    if meta.get("engine") == "langgraph":
        blocks.append(("p", f"Engine: LangGraph state graph, {graph.get('nodes', 0)} nodes, "
                            f"{graph.get('copy_revisions', 0)} copy revision(s) requested by the review node."))
    blocks.append(("p", f"Steps run: {_join(meta['steps'])}. Files: `{meta['run_dir']}`"))
    return blocks


# --------------------------------------------------------------------------- renderers

def _md_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def to_markdown(blocks: list[Block]) -> str:
    out: list[str] = []
    for block in blocks:
        kind = block[0]
        if kind in ("h1", "h2", "h3"):
            out.append("#" * int(kind[1]) + " " + block[1])
        elif kind == "p":
            out.append(block[1])
        elif kind == "note":
            out.append("> " + block[1])
        elif kind == "warn":
            out.append("> **Note:** " + block[1])
        elif kind == "ul":
            out.append("\n".join("- " + item for item in block[1]))
        elif kind == "table":
            headers, rows = block[1], block[2]
            lines = ["| " + " | ".join(_md_cell(h) for h in headers) + " |",
                     "| " + " | ".join("---" for _ in headers) + " |"]
            lines += ["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in rows]
            out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


def _inline(text) -> str:
    """Escape, then apply the three inline marks the report uses: links, bold, code."""
    safe = html.escape(str(text), quote=False)
    safe = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
                  lambda m: f'<a href="{html.escape(html.unescape(m.group(2)), quote=True)}" target="_blank" rel="noopener">{m.group(1)}</a>', safe)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)


CSS = """
:root{--bg:#fbfaf7;--card:#fff;--ink:#1c2430;--muted:#5d6878;--line:#e3e0d8;--accent:#0f6b5c;--warn:#8a5a00;--warnbg:#fff6e0;--notebg:#eef5f3}
@media (prefers-color-scheme:dark){:root{--bg:#14181d;--card:#1b2027;--ink:#e8ecf1;--muted:#9aa5b4;--line:#2c343f;--accent:#5fcfb8;--warn:#f2c46b;--warnbg:#2b2415;--notebg:#17262a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:30px;line-height:1.2;margin:0 0 8px}h2{font-size:22px;margin:44px 0 12px;padding-top:20px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:24px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
p{margin:10px 0}a{color:var(--accent)}code{background:var(--notebg);padding:1px 6px;border-radius:4px;font-size:.9em;word-break:break-all}
ul{padding-left:22px;margin:10px 0}li{margin:5px 0}
.note,.warn{padding:12px 16px;border-radius:8px;margin:14px 0;font-size:15px}.note{background:var(--notebg)}.warn{background:var(--warnbg);color:var(--warn)}
.scroll{overflow-x:auto;margin:14px 0;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:14.5px}th,td{text-align:left;padding:10px 14px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);white-space:nowrap}tr:last-child td{border-bottom:0}
"""


def to_html(blocks: list[Block], title: str = "Ad strategy") -> str:
    out: list[str] = []
    for block in blocks:
        kind = block[0]
        if kind in ("h1", "h2", "h3"):
            out.append(f"<{kind}>{_inline(block[1])}</{kind}>")
        elif kind == "p":
            out.append(f"<p>{_inline(block[1])}</p>")
        elif kind in ("note", "warn"):
            out.append(f'<div class="{kind}">{_inline(block[1])}</div>')
        elif kind == "ul":
            out.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in block[1]) + "</ul>")
        elif kind == "table":
            head = "".join(f"<th>{_inline(h)}</th>" for h in block[1])
            body = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>" for row in block[2])
            out.append(f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>'
            f"<style>{CSS}</style></head><body><main>{''.join(out)}</main></body></html>")
