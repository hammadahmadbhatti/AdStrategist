"""Optional local vision model (through Ollama) that says what a photo or clip shows.

Text models cannot see. Without this, the script is written from file names alone. With a small
vision model installed (for example `ollama pull moondream`), each picture is described first so the
script can match what is actually on screen. Everything stays on this computer.
"""
from __future__ import annotations

import base64
import io

import httpx
from PIL import Image

QUESTION = "Describe what this picture shows in one short sentence. Mention the main objects, people and setting."


def vision_status(host: str, model: str) -> dict:
    if not model:
        return {"ready": False, "detail": "No vision model is set, so scripts go by file names. "
                                          "For scripts that match the pictures: ollama pull moondream, then set "
                                          "VISION_MODEL=moondream in .env."}
    try:
        names = [m.get("name", "") for m in httpx.get(f"{host}/api/tags", timeout=3.0).json().get("models", [])]
    except (httpx.HTTPError, ValueError):
        return {"ready": False, "detail": f"Ollama is not running, so {model} cannot describe your pictures."}
    if model in names or f"{model}:latest" in names:
        return {"ready": True, "detail": f"The local vision model {model} will look at each picture first."}
    return {"ready": False, "detail": f"The vision model {model} is not installed. Run: ollama pull {model}"}


def describe(host: str, model: str, picture: Image.Image, timeout: float = 180.0) -> str:
    """One sentence about the picture, or '' when the model cannot be asked."""
    small = picture.convert("RGB")
    small.thumbnail((768, 768))
    buffer = io.BytesIO()
    small.save(buffer, "JPEG", quality=85)
    payload = {"model": model, "stream": False, "keep_alive": "2m", "options": {"temperature": 0.1, "num_predict": 60},
               "messages": [{"role": "user", "content": QUESTION,
                             "images": [base64.b64encode(buffer.getvalue()).decode()]}]}
    try:
        resp = httpx.post(f"{host}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        text = (resp.json().get("message") or {}).get("content", "")
    except (httpx.HTTPError, ValueError):
        return ""
    text = " ".join(text.split())
    first = text.split(". ")[0].strip()
    return (first if first.endswith(".") else first + ".")[:200] if first else ""
