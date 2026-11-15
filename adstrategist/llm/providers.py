"""Concrete providers: offline (no model), Ollama, OpenAI-compatible (also MLX), Claude."""
from __future__ import annotations

import os
from pathlib import Path

import httpx

from .base import LLMError, LLMProvider, LLMResult, ProviderStatus


class OfflineProvider(LLMProvider):
    """No model at all. Steps fall back to their rule-based output."""

    id = "offline"
    label = "No model (rule-based)"
    is_local = True

    def status(self) -> ProviderStatus:
        return ProviderStatus(True, "Always available. Research still runs, copy uses templates.")

    def complete(self, prompt, *, system="", max_tokens=1024, json_schema=None) -> LLMResult:
        raise LLMError("offline provider has no model")


class OllamaProvider(LLMProvider):
    id = "ollama"
    label = "Ollama (local)"
    is_local = True

    def __init__(self, model: str, host: str, num_ctx: int = 4096, timeout: float = 600.0) -> None:
        super().__init__(model)
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout

    def installed_models(self) -> list[str]:
        resp = httpx.get(f"{self.host}/api/tags", timeout=3.0)
        resp.raise_for_status()
        return [m.get("name", "") for m in resp.json().get("models", [])]

    def status(self) -> ProviderStatus:
        try:
            names = self.installed_models()
        except (httpx.HTTPError, ValueError):
            return ProviderStatus(False, f"Ollama is not running at {self.host}. Start it with: ollama serve")
        if self.model in names or f"{self.model}:latest" in names:
            return ProviderStatus(True, f"Running, model {self.model} is installed.")
        return ProviderStatus(False, f"Ollama is running but {self.model} is missing. Run: ollama pull {self.model}")

    def complete(self, prompt, *, system="", max_tokens=1024, json_schema=None) -> LLMResult:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.3, "num_ctx": self.num_ctx, "num_predict": max_tokens},
        }
        if json_schema:
            payload["format"] = json_schema
        try:
            resp = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Ollama error {exc.response.status_code}: {exc.response.text[:200]}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        text = (data.get("message") or {}).get("content", "")
        if not text.strip():
            raise LLMError("Ollama returned an empty response")
        return LLMResult(text, data.get("prompt_eval_count", 0) or 0, data.get("eval_count", 0) or 0)


class OpenAICompatProvider(LLMProvider):
    """Chat Completions protocol. Covers OpenAI, OpenRouter, Groq, LM Studio and `mlx_lm.server`."""

    def __init__(self, provider_id: str, label: str, model: str, base_url: str, api_key: str = "",
                 is_local: bool = False, timeout: float = 600.0) -> None:
        super().__init__(model)
        self.id = provider_id
        self.label = label
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.is_local = is_local
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def available_models(self) -> list[str]:
        """Model ids this key can use, newest first. Empty when the server cannot be asked."""
        try:
            resp = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=6.0)
            rows = resp.json().get("data", []) if resp.status_code == 200 else []
        except (httpx.HTTPError, ValueError, AttributeError):
            return []
        rows = [r for r in rows if isinstance(r, dict) and r.get("id")]
        rows.sort(key=lambda r: -(r.get("created") or 0))
        skip = ("embedding", "whisper", "tts", "dall-e", "image", "audio", "realtime", "moderation",
                "transcribe", "babbage", "davinci", "search", "sora")
        return [r["id"] for r in rows if not any(word in r["id"] for word in skip)]

    def status(self) -> ProviderStatus:
        if not self.is_local and not self.api_key:
            return ProviderStatus(False, "No API key. Set OPENAI_API_KEY in .env.")
        try:
            resp = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=6.0)
        except httpx.HTTPError:
            hint = " Start it with: mlx_lm.server --model " + self.model if self.id == "mlx" else ""
            return ProviderStatus(False, f"Nothing is answering at {self.base_url}.{hint}")
        if resp.status_code in (401, 403):
            return ProviderStatus(False, "The API key was rejected. Check OPENAI_API_KEY in .env.")
        if not self.model:
            return ProviderStatus(False, "The key works. Now set OPENAI_MODEL in .env to one of the models listed.")
        return ProviderStatus(True, f"Reachable, model {self.model}." + ("" if self.is_local else " Calls are billed."))

    def complete(self, prompt, *, system="", max_tokens=1024, json_schema=None) -> LLMResult:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload: dict = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.3}
        if json_schema:
            payload["response_format"] = {"type": "json_object"}
        # Servers disagree on optional parameters, so drop whichever one a 400 complains about.
        for _ in range(4):
            try:
                resp = httpx.post(f"{self.base_url}/chat/completions", json=payload,
                                  headers=self._headers(), timeout=self.timeout)
            except httpx.HTTPError as exc:
                raise LLMError(f"{self.label} request failed: {exc}") from exc
            if resp.status_code == 400:
                body = resp.text.lower()
                if "max_tokens" in body and "max_tokens" in payload:
                    payload["max_completion_tokens"] = max(payload.pop("max_tokens"), 4000)
                    continue
                if "temperature" in body and "temperature" in payload:
                    payload.pop("temperature")
                    continue
                if "response_format" in payload:
                    payload.pop("response_format")
                    continue
            if resp.status_code >= 400:
                raise LLMError(f"{self.label} error {resp.status_code}: {resp.text[:200]}")
            break
        else:
            raise LLMError(f"{self.label} kept rejecting the request")
        try:
            data = resp.json()
            text = data["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.label} returned an unexpected response") from exc
        if not text.strip():
            raise LLMError(f"{self.label} returned an empty response")
        usage = data.get("usage") or {}
        return LLMResult(text, usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0)


class AnthropicProvider(LLMProvider):
    """Claude through the official SDK. Credentials come from the environment or an `ant auth login` profile."""

    id = "anthropic"
    label = "Claude (Anthropic API, paid)"
    is_local = False
    # Models where a declined request can be re-run server side on a fallback model.
    FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high") -> None:
        super().__init__(model)
        self.effort = effort

    def status(self) -> ProviderStatus:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return ProviderStatus(False, "The anthropic package is not installed. Run: pip install anthropic")
        has_env = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        has_profile = (Path.home() / ".config" / "anthropic").exists()
        if not (has_env or has_profile):
            return ProviderStatus(False, "No credentials. Set ANTHROPIC_API_KEY in .env or run: ant auth login")
        return ProviderStatus(True, f"Credentials found, model {self.model}. Calls are billed.")

    def complete(self, prompt, *, system="", max_tokens=1024, json_schema=None) -> LLMResult:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("The anthropic package is not installed") from exc

        # Thinking shares the output budget, so the cap stays generous whatever the caller asked for.
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max(max_tokens, 16000),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        output_config: dict = {}
        if self.effort:
            output_config["effort"] = self.effort
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if output_config:
            kwargs["output_config"] = output_config

        try:
            client = anthropic.Anthropic()
            if self.model in self.FALLBACK_MODELS:
                message = client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                message = client.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMError("Claude rejected the API key") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError("This API key may not use the selected Claude model") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Claude model not found: {self.model}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude rate limit reached, try again shortly") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"Claude rejected the request: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Claude API") from exc
        except anthropic.AnthropicError as exc:
            raise LLMError(f"Claude client error: {exc}") from exc

        if message.stop_reason == "refusal":
            raise LLMError("Claude declined this request")
        text = "".join(block.text for block in message.content if block.type == "text")
        if not text.strip():
            raise LLMError("Claude returned an empty response")
        return LLMResult(text, message.usage.input_tokens or 0, message.usage.output_tokens or 0)
