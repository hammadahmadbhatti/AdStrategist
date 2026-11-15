"""Settings loaded from the environment (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent

PLATFORMS = {
    "google": "Google Search",
    "meta": "Meta (Facebook + Instagram)",
    "tiktok": "TikTok",
}


_FROM_FILE: set[str] = set()  # keys this process took from .env, as opposed to the real environment


def load_env(path: Path | None = None) -> None:
    """Read KEY=VALUE lines into os.environ.

    Real environment variables always win. Values that came from the file are refreshed on every
    call, so editing .env (adding an API key, switching provider) applies without a restart.
    """
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in os.environ or key in _FROM_FILE:
            os.environ[key] = value.strip('"').strip("'")
            _FROM_FILE.add(key)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    research_provider: str = "ollama"
    writing_provider: str = "ollama"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_num_ctx: int = 4096
    mlx_base_url: str = "http://localhost:8080/v1"
    mlx_model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = ""
    # Local vision model in Ollama that describes your photos and clips (for example moondream). Optional.
    vision_model: str = ""
    # Which role writes the video script: "research" keeps it on the local model.
    video_script_role: str = "research"
    # Retrieval (RAG): "auto" uses it when the packages and the embedding model are present, "off" never.
    rag: str = "auto"
    embed_model: str = "embeddinggemma"
    # Below this similarity a passage is treated as not about the question at all.
    rag_min_score: float = 0.25
    # Pipeline engine: "native" (plain Python loop) or "langgraph" (state graph with a copy review loop).
    engine: str = "native"
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "high"
    meta_access_token: str = ""
    default_country: str = "DE"
    default_currency: str = "EUR"
    output_dir: Path = field(default_factory=lambda: ROOT / "runs")

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        base = cls()
        out = Path(_env("OUTPUT_DIR", "runs"))
        return cls(
            research_provider=_env("RESEARCH_PROVIDER", base.research_provider).lower(),
            writing_provider=_env("WRITING_PROVIDER", _env("RESEARCH_PROVIDER", base.writing_provider)).lower(),
            ollama_host=_env("OLLAMA_HOST", base.ollama_host).rstrip("/"),
            ollama_model=_env("OLLAMA_MODEL", base.ollama_model),
            ollama_num_ctx=int(_env("OLLAMA_NUM_CTX", str(base.ollama_num_ctx)) or base.ollama_num_ctx),
            mlx_base_url=_env("MLX_BASE_URL", base.mlx_base_url).rstrip("/"),
            mlx_model=_env("MLX_MODEL", base.mlx_model),
            openai_base_url=_env("OPENAI_BASE_URL", base.openai_base_url).rstrip("/"),
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_model=_env("OPENAI_MODEL"),
            vision_model=_env("VISION_MODEL"),
            video_script_role=_env("VIDEO_SCRIPT_ROLE", base.video_script_role).lower(),
            rag=_env("RAG", base.rag).lower(),
            embed_model=_env("EMBED_MODEL", base.embed_model),
            rag_min_score=float(_env("RAG_MIN_SCORE", "") or base.rag_min_score),
            engine=_env("ENGINE", base.engine).lower(),
            anthropic_model=_env("ANTHROPIC_MODEL", base.anthropic_model),
            anthropic_effort=_env("ANTHROPIC_EFFORT", base.anthropic_effort).lower(),
            meta_access_token=_env("META_ACCESS_TOKEN"),
            default_country=_env("DEFAULT_COUNTRY", base.default_country).upper(),
            default_currency=_env("DEFAULT_CURRENCY", base.default_currency).upper(),
            output_dir=out if out.is_absolute() else ROOT / out,
        )
