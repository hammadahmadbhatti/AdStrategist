# AdStrategist usage guide

Paste a product, website or app link. AdStrategist researches the market and hands back a budget plan and
ad campaigns for Google, Meta (Facebook and Instagram) and TikTok, plus an optional vertical video ad.

It runs on a MacBook Air M1 with 8 GB of RAM. The model is switchable: a small local model, a paid API,
or no model at all. You choose which steps run.

## What it does

| Step | What you get | Where the data comes from |
|---|---|---|
| Product brief | What the product is, audience, selling points, offers, tracking tags | Your page, read directly |
| Competitors | Up to 5 competitors with positioning, prices and offers | Google autocomplete ("brand vs ..."), web search, their homepages |
| Competitor ads | Which ad platforms each competitor is set up on, links to their live ads | Tracking tags in their page source, public ad libraries |
| Keyword research | Search phrases grouped by intent, negative keywords | Google autocomplete, optional Keyword Planner CSV |
| Market gaps | Topics, offers and channels competitors cover and you do not | Comparison of the pages above |
| Traffic potential | Click and cost ranges | Only from your Keyword Planner CSV |
| Budget plan | A proposed budget or a split of yours, with click and conversion ranges | Arithmetic over editable benchmarks and real bids |
| Ad campaigns | Search ads, Meta ads, TikTok ads, import files | Model writes, code enforces every character limit |
| Video ad | 9:16 MP4 with captions and voice | Your own photos and clips, or the site's images; local model script, Pillow, ffmpeg, macOS voice |
| Summary | Executive summary and next steps | Model, from the figures above only |

Two rules the whole design follows:

- **Models write and summarise, they never supply numbers.** Every figure comes from a named source or is a
  benchmark assumption labelled as one.
- **Ad copy may only say what your page supports.** A claim check drops any line that mentions a number,
  discount, free offer, guarantee, superlative or health benefit your page does not show. Small models
  invent these ("20% off"), so the report lists every line that was removed and why.

## Setup

```bash
cd ad-strategist
python3 -m venv .venv
.venv/bin/pip install -e ".[rag,mcp,claude,dev]"
cp .env.example .env
```

For the local model (recommended default):

```bash
brew install ollama
ollama serve
ollama pull llama3.2:3b
```

`ffmpeg` is needed for the video step (`brew install ffmpeg`).

## Use it

Web interface, with tick boxes for steps, platforms and models:

```bash
.venv/bin/adstrategist serve
```

Then open http://127.0.0.1:8765.

Command line:

```bash
.venv/bin/adstrategist run https://example.com --budget 800 --country DE --open
```

Guided questions instead of flags:

```bash
.venv/bin/adstrategist run --interactive
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--steps keywords,budget` | Run only these steps. Steps they depend on are added automatically. |
| `--skip video` | Run everything except these. |
| `--platforms google,meta` | Which ad platforms to plan for. |
| `--budget 800` | Monthly budget. Leave it out and a budget is proposed. |
| `--customer-value 120` | What one customer is worth. Adds a payback verdict per platform. |
| `--keywords "tea subscription, organic tea box"` | Your own search phrases instead of automatic ones. |
| `--competitor https://rival.com` | A competitor you already know (repeatable). |
| `--keyword-csv planner.csv` | Keyword Planner export. Switches on real volumes, bids and traffic estimates. |
| `--video auto\|yes\|no` | `auto` renders a video only if Meta or TikTok is funded. |
| `--media ~/Desktop/shots` | Your own photos and clips (file or folder, repeatable) for the video ad. |
| `--research ollama --writing anthropic` | Different models for the two roles (see below). |

`adstrategist providers` shows which models are ready. `adstrategist steps` lists the steps.

Each run writes a folder under `runs/` with `report.html`, `report.md`, `result.json`, `keywords.csv`,
`campaigns/` and `video/`.

## Switching models

Two roles are routed separately, in `.env` or per run:

- **Research model**: reads pages, classifies, extracts. Many small calls. A local 3B model is enough.
- **Writing model**: ad copy, gap analysis, executive summary. Few calls, and where a small model is
  weakest. This is the one worth pointing at a paid model.

| Provider | Value | Needs |
|---|---|---|
| No model | `offline` | Nothing. Research still runs, copy comes from templates. |
| Ollama | `ollama` | `ollama serve` and a pulled model |
| MLX | `mlx` | `mlx_lm.server --model <model>` running |
| OpenAI-compatible | `openai` | `OPENAI_API_KEY`, `OPENAI_MODEL`, optional `OPENAI_BASE_URL` (OpenAI, OpenRouter, Groq, LM Studio) |
| Claude | `anthropic` | `ANTHROPIC_API_KEY`, or `ant auth login` |

To attach an OpenAI key: open `.env`, paste it after `OPENAI_API_KEY=`, and set `OPENAI_MODEL`. Run
`adstrategist providers` to see which models the key can use. Edits to `.env` apply on the next run or
page reload, no restart needed. The key stays in that file: it is never sent to the browser or written
to a report.

Typical hybrid setup, local research and paid writing:

```bash
.venv/bin/adstrategist run https://example.com --research ollama --writing anthropic
```

If a model is unavailable or a call fails, the run does not stop: a failed paid call falls back to the local
model, and a failed local call falls back to rule-based output. The report states which model wrote what and
lists token usage.

### Models for an 8 GB Mac

| Model | Download | Notes |
|---|---|---|
| `llama3.2:3b` | 2.0 GB | Default. Fast, reliable JSON. |
| `moondream` | 1.7 GB | Optional vision model: describes your photos and clips for the video script. |
| `gemma3:4b` | 3.3 GB | Better writing, especially in German. Slower. |

Close other heavy apps while it runs. A full run with `llama3.2:3b` took 2.5 to 4 minutes on this machine
(English and German sites, all steps including the video).

## Retrieval (RAG) and questions about a run

Install the optional packages and the local embedding model once:

```bash
.venv/bin/pip install -e ".[rag]"
```

```bash
ollama pull embeddinggemma
```

From then on every run builds a vector index in its folder (`runs/<run>/index`). Pages from your site,
competitor pages and the finished report are split into passages, embedded on this computer and stored in
Chroma. The run uses the index in three ways:

- Prompts for the product brief and competitor profiles get the most relevant passages instead of the
  first part of the page, and up to five extra pages of your site are read.
- Market gap findings carry the competitor passage they came from, with its link and a similarity score.
- You can ask questions afterwards:

```bash
.venv/bin/adstrategist ask "Which competitors offer a guarantee?"
```

`--run <id or folder>` picks a run; the default is the latest. In the web form, the "Ask about this
research" box appears above the report for any run that has an index. Answers cite numbered sources and
list the quoted passages. If nothing in the index is about the question, the reply says so and no model
is called to answer. Before giving up, the question is rewritten once by the local model and searched again;
figures in an answer are checked against the passages it cites.

Settings in `.env`: `RAG=auto|off`, `EMBED_MODEL`, `RAG_MIN_SCORE` (tuned for `embeddinggemma`; another
embedding model needs a different value). `--no-rag` switches retrieval off for one run. `adstrategist
providers` shows whether retrieval is ready.

## MCP server

Install the optional package once:

```bash
.venv/bin/pip install -e ".[mcp]"
```

Then any MCP client can start the server with the command `.venv/bin/adstrategist mcp` (stdio). For Claude
Desktop, add this to its MCP configuration with the absolute path to this folder:

```json
{
  "mcpServers": {
    "adstrategist": {
      "command": "/absolute/path/to/ad-strategist/.venv/bin/adstrategist",
      "args": ["mcp"]
    }
  }
}
```

Tools: `start_research`, `research_status`, `list_runs`, `get_report`, `get_campaigns`, `ask_research`.
A research run takes minutes, so `start_research` returns a run id immediately and the client polls
`research_status`. The server uses the same `.env` settings as the command line.

## Pipeline engines

| Engine | What it is | When to use it |
|---|---|---|
| `native` | A plain loop over the steps. Default. | Fastest, fewest moving parts. |
| `langgraph` | The same steps as a LangGraph state graph, plus a review of the ad copy: lines the claim check rejects go back to the writing model once, with the reasons. | When you want the model to rewrite rejected copy instead of falling back to template lines. Costs a few more model calls. |

Choose with `--engine langgraph`, `ENGINE=langgraph` in `.env`, or the engine menu in the web form.
`adstrategist graph` prints the graph as a Mermaid diagram, and each LangGraph run saves it as `graph.mmd`.

## Your own photos and video clips

Give the video step your own material and it builds the ad from it, entirely on this computer:

```bash
.venv/bin/adstrategist run https://example.com --media ~/Desktop/shots
```

`--media` takes a photo, a clip or a folder, and can be repeated. In the web form, pick the files under
"Your own photos and video clips". Up to 8 files are used, in the order given, one scene each, followed by
a closing card in your brand colour with the call to action.

- **Photos:** JPEG, PNG, WebP and iPhone HEIC. **Clips:** MP4, MOV, M4V, WebM.
- **Clips play as footage.** Portrait clips fill the frame; landscape ones sit as a card over a blurred
  copy of themselves. Clip sound is muted in favour of the voice-over. A clip shorter than its line
  repeats, a longer one is cut.
- **The local model writes the script** around your shots (`VIDEO_SCRIPT_ROLE=research` in `.env`; set it
  to `writing` to use the writing model instead).
- **A local vision model can look at the shots first** so the script matches what is on screen:

  ```bash
  ollama pull moondream
  ```

  then `VISION_MODEL=moondream` in `.env`. Without it, the script goes by file names, so
  `grill-steak-closeup.mp4` helps and `IMG_4032.mov` does not.
- **Safeguards:** small vision models misread logos and signs, so anything the vision model claims to have
  read is stripped from its notes and blocked from the script. The claim check applies as everywhere else.
  A shot whose line is rejected keeps its place and gets a rule-based line, or plays without a caption.

With no media, the video is a slideshow of images from the site. OpenAI is not used for video: it shut
down its video generation API on 24 September 2026.

## Getting real search volumes

No free source publishes search volumes or bids, and a model must not guess them. To get them:

1. Run once. Open `keywords_for_keyword_planner.txt` in the run folder.
2. In Google Ads: Tools > Keyword Planner > Get search volume and forecasts. Paste the list.
3. Download the result as CSV.
4. Run again with `--keyword-csv that-file.csv` (or upload it under More options in the web form).

The keyword table, traffic estimate and Google cost per click then use your real numbers.

## What leaves your computer

- Requests to the page you entered and to competitor homepages (robots.txt is respected).
- Search queries to DuckDuckGo and Google autocomplete, built from the product's category and keywords.
- With a paid model selected: page text and research findings go to that provider.
- With `META_ACCESS_TOKEN` set: competitor names go to the Meta Ad Library API.

With local models only, no page content is sent to any AI service.

## Limits to know about

- **Competitor spend is not public anywhere.** You get which platforms they are set up on and links to
  their live ads, not what they pay.
- **Budget figures are ranges from benchmarks** in `adstrategist/data/benchmarks.json` until you supply a
  keyword CSV or your own account data. Edit that file to match your market.
- **Ad tags can hide inside Google Tag Manager.** The report says so when that applies.
- **Pages rendered entirely by JavaScript read as nearly empty.** The report warns when that happens.
  Pass `--keywords` and `--competitor` in that case.
- **The video is edited, not generated.** It cuts together your photos and clips (or site images) with
  captions and voice. Generating footage does not fit in 8 GB of memory.
- **Campaigns are created paused and need your review.** `google_ads_editor.csv` follows Google Ads Editor's
  import columns (Account > Import > From file, then check the preview). The Meta and TikTok files are
  campaign sheets to copy into Ads Manager, not guaranteed bulk-import formats.
- **Not verified against live services:** the Claude provider is tested against a local stand-in server
  (OpenAI text was verified live with `gpt-4o-mini`), and the Meta Ad Library call has not been run with a real token.
- **A 3B model writes plain, sometimes clumsy copy,** more so in German. Expect to edit it, or point the
  writing role at a stronger model.
- **Search can be throttled.** If competitor discovery comes back empty, wait a minute or pass
  `--competitor` links.

## Tests

```bash
.venv/bin/python -m pytest -q
```

106 tests, no network needed: a fake web serves the product page, competitors, search and autocomplete.

## Layout

```
adstrategist/
  llm/            provider interface, Ollama, OpenAI-compatible, Claude, role router
  sources/        search, autocomplete, Keyword Planner CSV, ad libraries
  steps/          one module per pipeline step
  data/benchmarks.json   editable cost and conversion assumptions
  claims.py       claim check for ad copy
  media.py        your own photos and clips for the video ad
  pipeline.py     runs the selected steps, writes result files
  report.py       Markdown and HTML report
  server.py       local web interface (FastAPI), static/index.html
  cli.py          command line
tests/
```
