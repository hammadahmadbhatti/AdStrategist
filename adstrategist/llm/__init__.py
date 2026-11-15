"""Switchable model layer.

Two roles are routed independently:
  research: reading pages, classifying, extracting (many small calls)
  writing:  ad copy, gap analysis, executive summary (few calls, quality matters)

Either role can point at a local model, a paid API, or no model at all.
"""
from __future__ import annotations

import json
import time
from typing import Callable

from ..config import Settings
from .base import LLMError, LLMProvider, LLMResult, ProviderStatus, coerce, extract_json, skeleton
from .providers import AnthropicProvider, OfflineProvider, OllamaProvider, OpenAICompatProvider

PROVIDER_IDS = ["offline", "ollama", "mlx", "openai", "anthropic"]

SYSTEM = (
    "You are a performance marketing analyst. Work only from the facts you are given. "
    "Never invent statistics, prices, search volumes, awards or customer numbers."
)


def build_provider(provider_id: str, settings: Settings, model: str = "") -> LLMProvider:
    provider_id = (provider_id or "offline").lower()
    if provider_id == "offline":
        return OfflineProvider()
    if provider_id == "ollama":
        return OllamaProvider(model or settings.ollama_model, settings.ollama_host, settings.ollama_num_ctx)
    if provider_id == "mlx":
        return OpenAICompatProvider("mlx", "MLX server (local)", model or settings.mlx_model,
                                    settings.mlx_base_url, is_local=True)
    if provider_id == "openai":
        local = any(h in settings.openai_base_url for h in ("localhost", "127.0.0.1"))
        label = "OpenAI-compatible (local server)" if local else "OpenAI-compatible API (paid)"
        return OpenAICompatProvider("openai", label, model or settings.openai_model,
                                    settings.openai_base_url, settings.openai_api_key, is_local=local)
    if provider_id == "anthropic":
        return AnthropicProvider(model or settings.anthropic_model, settings.anthropic_effort)
    raise ValueError(f"Unknown provider '{provider_id}'. Choose one of: {', '.join(PROVIDER_IDS)}")


def provider_catalog(settings: Settings) -> list[dict]:
    """What the CLI and the web UI show so the user can pick a provider."""
    rows = []
    for pid in PROVIDER_IDS:
        provider = build_provider(pid, settings)
        status = provider.status()
        row = {"id": pid, "label": provider.label, "local": provider.is_local, "model": provider.model,
               "available": status.available, "detail": status.detail, "models": []}
        if pid == "ollama" and isinstance(provider, OllamaProvider):
            try:
                row["models"] = provider.installed_models()
            except Exception:
                row["models"] = []
        if pid == "openai" and isinstance(provider, OpenAICompatProvider) and (provider.api_key or provider.is_local):
            row["models"] = provider.available_models()[:25]
        rows.append(row)
    return rows


class LLMRouter:
    """Sends each task to the provider chosen for its role and never lets a model failure stop a run."""

    def __init__(self, research: LLMProvider, writing: LLMProvider,
                 emit: Callable[[str, str], None] | None = None) -> None:
        self.providers = {"research": research, "writing": writing}
        self.emit = emit or (lambda level, message: None)
        self.calls: list[dict] = []

    def enabled(self, role: str = "research") -> bool:
        return self.providers[role].id != "offline"

    def describe(self) -> dict:
        return {role: {"provider": p.id, "label": p.label, "model": p.model, "local": p.is_local}
                for role, p in self.providers.items()}

    def _chain(self, role: str) -> list[LLMProvider]:
        chain = [self.providers[role]]
        other = self.providers["research"]
        if role == "writing" and other is not chain[0] and (other.id, other.model) != (chain[0].id, chain[0].model):
            chain.append(other)  # a failed paid call falls back to the local model
        return [p for p in chain if p.id != "offline"]

    def _call(self, provider: LLMProvider, task: str, prompt: str, system: str, max_tokens: int,
              schema: dict | None) -> LLMResult:
        started = time.time()
        record = {"task": task, "provider": provider.id, "model": provider.model, "ok": False,
                  "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}
        self.calls.append(record)
        try:
            result = provider.complete(prompt, system=system, max_tokens=max_tokens, json_schema=schema)
        finally:
            record["seconds"] = round(time.time() - started, 1)
        record.update(ok=True, input_tokens=result.input_tokens, output_tokens=result.output_tokens)
        return result

    def ask_json(self, task: str, prompt: str, schema: dict, *, role: str = "research", system: str = SYSTEM,
                 max_tokens: int = 700, repeat: dict | None = None) -> dict | None:
        full = (f"{prompt}\n\nReply with one JSON object only, in exactly this shape:\n"
                f"{json.dumps(skeleton(schema, repeat), ensure_ascii=False)}")
        for provider in self._chain(role):
            for _attempt in range(2):
                try:
                    result = self._call(provider, task, full, system, max_tokens, schema)
                    return coerce(extract_json(result.text), schema)
                except LLMError as exc:
                    error = exc
            self.emit("warn", f"{task}: {provider.label} failed ({error}).")
        return None

    def ask_text(self, task: str, prompt: str, *, role: str = "writing", system: str = SYSTEM,
                 max_tokens: int = 900) -> str | None:
        for provider in self._chain(role):
            try:
                return self._call(provider, task, prompt, system, max_tokens, None).text.strip()
            except LLMError as exc:
                self.emit("warn", f"{task}: {provider.label} failed ({exc}).")
        return None

    def usage(self) -> dict:
        totals: dict[str, dict] = {}
        for call in self.calls:
            key = f"{call['provider']}:{call['model']}" if call["model"] else call["provider"]
            row = totals.setdefault(key, {"calls": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0})
            row["calls"] += 1
            row["failed"] += 0 if call["ok"] else 1
            row["input_tokens"] += call["input_tokens"]
            row["output_tokens"] += call["output_tokens"]
            row["seconds"] = round(row["seconds"] + call["seconds"], 1)
        return totals


__all__ = ["LLMRouter", "LLMError", "LLMProvider", "LLMResult", "ProviderStatus", "build_provider",
           "provider_catalog", "PROVIDER_IDS", "SYSTEM"]
