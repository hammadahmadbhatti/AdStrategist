"""Local web interface. Binds to localhost only; API keys stay in .env and are never sent to the browser."""
from __future__ import annotations

import base64
import json
import re
import secrets
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .config import PACKAGE_DIR, PLATFORMS, Settings
from .context import RunConfig
from .llm import provider_catalog
from .media import IMAGE_EXT, MAX_MEDIA, VIDEO_EXT
from .rag import RagUnavailable, rag_status
from .sources.vision import vision_status
from .pipeline import run_pipeline, validate
from .steps import ALWAYS, STEPS
from .textutil import registrable_domain, slugify

app = FastAPI(title="AdStrategist", docs_url=None, redoc_url=None)
MAX_UPLOAD_BYTES = 600 * 1024 * 1024  # per file: enough for a minute of phone video
RUNS: dict[str, dict] = {}
LOCK = threading.Lock()


class AskRequest(BaseModel):
    question: str


class RunRequest(BaseModel):
    url: str
    steps: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=lambda: list(PLATFORMS))
    budget: float | None = None
    currency: str = ""
    country: str = ""
    language: str = ""
    competitors: list[str] = Field(default_factory=list)
    max_competitors: int = 5
    seed_keywords: list[str] = Field(default_factory=list)
    customer_value: float | None = None
    video: str = "auto"
    upload_id: str = ""
    engine: str = ""
    rag: bool = True
    research_provider: str = ""
    research_model: str = ""
    writing_provider: str = ""
    writing_model: str = ""
    keyword_csv_b64: str = ""


def _settings() -> Settings:
    return Settings.from_env()


def _run_dir(run_id: str) -> Path:
    base = _settings().output_dir.resolve()
    path = (base / run_id).resolve()
    if base not in path.parents or not path.is_dir():
        raise HTTPException(404, "Run not found")
    return path


def _upload_dir(upload_id: str, create: bool = False) -> Path:
    if not re.fullmatch(r"[0-9a-f]{16}", upload_id):
        raise HTTPException(404, "Upload not found")
    path = _settings().output_dir / "_uploads" / upload_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise HTTPException(404, "Upload not found")
    return path


def _uploaded_media(upload_id: str) -> list[str]:
    """Uploaded files in the order they were sent (each is stored with a numeric prefix)."""
    return [str(p) for p in sorted(_upload_dir(upload_id).iterdir()) if p.is_file()]


def _files(run_dir: Path) -> list[str]:
    return sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*")
                  if p.is_file() and p.name != "keywords_upload.csv" and "index" not in p.relative_to(run_dir).parts)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (PACKAGE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/options")
def options() -> dict:
    settings = _settings()
    return {
        "steps": [{"id": sid, "label": label, "description": description, "needs": deps, "always": sid in ALWAYS}
                  for sid, (label, description, _module, deps) in STEPS.items()],
        "platforms": [{"id": pid, "label": label} for pid, label in PLATFORMS.items()],
        "providers": provider_catalog(settings),
        "rag": rag_status(settings),
        "engine": settings.engine,
        "media": {"image_types": sorted(IMAGE_EXT), "video_types": sorted(VIDEO_EXT), "max_files": MAX_MEDIA,
                  "vision": vision_status(settings.ollama_host, settings.vision_model)},
        "defaults": {"research_provider": settings.research_provider, "writing_provider": settings.writing_provider,
                     "country": settings.default_country, "currency": settings.default_currency},
    }


@app.post("/api/uploads")
def create_upload() -> dict:
    upload_id = secrets.token_hex(8)
    _upload_dir(upload_id, create=True)
    return {"id": upload_id, "max_files": MAX_MEDIA}


@app.put("/api/uploads/{upload_id}/{name}")
async def upload_file(upload_id: str, name: str, request: Request) -> dict:
    """Receives one photo or clip as the raw request body and streams it to disk. Local use only."""
    folder = _upload_dir(upload_id)
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_EXT | VIDEO_EXT:
        raise HTTPException(415, f"{name}: only photos and video clips can be used.")
    existing = [p for p in folder.iterdir() if p.is_file()]
    if len(existing) >= MAX_MEDIA:
        raise HTTPException(413, f"At most {MAX_MEDIA} files per video.")
    stem = re.sub(r"[^\w\- ]+", "_", Path(name).stem, flags=re.UNICODE).strip(" ._")[:80] or "file"
    target = folder / f"{len(existing) + 1:02d}_{stem}{suffix}"
    size = 0
    try:
        with open(target, "wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"{name} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
                handle.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    if size == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"{name} is empty.")
    return {"name": target.name, "bytes": size}


@app.post("/api/runs")
def start_run(request: RunRequest) -> dict:
    settings = _settings()
    data = request.model_dump()
    csv_b64 = data.pop("keyword_csv_b64")
    upload_id = data.pop("upload_id")
    config = RunConfig(**data)
    if upload_id:
        config.media = _uploaded_media(upload_id)
    config.currency = config.currency or settings.default_currency
    config.country = config.country or settings.default_country
    try:
        validate(config)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with LOCK:
        if any(run["status"] == "running" for run in RUNS.values()):
            raise HTTPException(409, "A run is already in progress. Wait for it to finish.")
        run_id = f"{slugify(registrable_domain(config.url))}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_dir = settings.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        RUNS[run_id] = {"status": "running", "events": [], "error": "", "steps": config.steps}
    if csv_b64:
        try:
            upload = run_dir / "keywords_upload.csv"
            upload.write_bytes(base64.b64decode(csv_b64))
            config.keyword_csv = str(upload)
        except ValueError:
            pass

    def work() -> None:
        state = RUNS[run_id]
        try:
            result = run_pipeline(config, settings, on_event=state["events"].append, run_dir=run_dir)
            state["errors"] = result["errors"]
            state["status"] = "done"
        except Exception as exc:  # surfaced to the browser instead of dying silently in a thread
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["status"] = "error"

    threading.Thread(target=work, daemon=True).start()
    return {"id": run_id, "steps": config.steps}


@app.get("/api/runs/{run_id}")
def run_status(run_id: str, since: int = 0) -> dict:
    state = RUNS.get(run_id)
    if state is None:
        run_dir = _run_dir(run_id)  # a run from an earlier session
        return {"id": run_id, "status": "done", "events": [], "next": 0, "error": "", "errors": {},
                "steps": [], "files": _files(run_dir), "has_index": (run_dir / "index").is_dir()}
    done = state["status"] != "running"
    return {"id": run_id, "status": state["status"], "events": state["events"][since:],
            "next": len(state["events"]), "error": state["error"], "errors": state.get("errors", {}),
            "steps": state["steps"], "files": _files(_run_dir(run_id)) if done else [],
            "has_index": done and (_run_dir(run_id) / "index").is_dir()}


@app.post("/api/runs/{run_id}/ask")
def ask_run(run_id: str, request: AskRequest) -> dict:
    """Answer a question from the run's retrieval index, with the passages it was answered from."""
    from .ask import ask

    run_dir = _run_dir(run_id)
    if any(run["status"] == "running" for run in RUNS.values()):
        raise HTTPException(409, "A run is in progress. Ask again when it has finished.")
    try:
        return ask(run_dir, request.question, _settings())
    except (RagUnavailable, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/runs")
def list_runs() -> list[dict]:
    base = _settings().output_dir
    if not base.is_dir():
        return []
    rows = []
    for folder in sorted((p for p in base.iterdir() if (p / "result.json").exists()),
                         key=lambda p: p.stat().st_mtime, reverse=True)[:15]:
        try:
            meta = json.loads((folder / "result.json").read_text(encoding="utf-8"))["meta"]
            rows.append({"id": folder.name, "url": meta["url"], "started": meta["started"]})
        except (ValueError, KeyError, OSError):
            continue
    return rows


@app.get("/runs/{run_id}/{file_path:path}")
def run_file(run_id: str, file_path: str) -> FileResponse:
    run_dir = _run_dir(run_id)
    target = (run_dir / file_path).resolve()
    if run_dir not in target.parents or not target.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(target)
