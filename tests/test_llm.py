"""The switchable model layer: parsing, routing, fallbacks and each provider's wire format."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from adstrategist.config import Settings
from adstrategist.llm import LLMError, LLMRouter, build_provider, provider_catalog
from adstrategist.llm.base import coerce, extract_json, skeleton
from adstrategist.llm.providers import AnthropicProvider, OfflineProvider, OllamaProvider, OpenAICompatProvider

from .conftest import FakeProvider

SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "kind": {"type": "string", "enum": ["a", "b", "other"]},
    "tags": {"type": "array", "items": {"type": "string"}}, "n": {"type": "integer"}},
    "required": ["name", "kind", "tags", "n"], "additionalProperties": False}


def test_extract_json_tolerates_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}
    with pytest.raises(LLMError):
        extract_json("no json here")
    with pytest.raises(LLMError):
        extract_json("[1, 2]")


def test_coerce_forces_model_output_into_shape():
    out = coerce({"name": 5, "kind": "B", "tags": "solo", "n": "7.0", "extra": 1}, SCHEMA)
    assert out == {"name": "5", "kind": "b", "tags": ["solo"], "n": 7}
    assert coerce("garbage", SCHEMA) == {"name": "", "kind": "other", "tags": [], "n": 0}
    assert skeleton(SCHEMA)["kind"] == "a | b | other" and skeleton(SCHEMA)["tags"] == ["...", "..."]
    assert len(skeleton(SCHEMA, {"tags": 5})["tags"]) == 5  # small models copy the example's length


def test_router_returns_none_when_offline_so_steps_use_rules():
    router = LLMRouter(OfflineProvider(), OfflineProvider())
    assert router.ask_json("t", "p", SCHEMA) is None and router.ask_text("t", "p") is None
    assert not router.enabled("research") and router.calls == []


def test_router_routes_roles_to_different_providers_and_tracks_usage():
    research = FakeProvider({"classify": {"name": "r", "kind": "a", "tags": [], "n": 1}}, model="small")
    writing = FakeProvider({"write": "Polished summary."}, model="big")
    router = LLMRouter(research, writing)
    assert router.ask_json("x", "classify this", SCHEMA, role="research")["name"] == "r"
    assert router.ask_text("y", "write this", role="writing") == "Polished summary."
    assert len(research.prompts) == 1 and len(writing.prompts) == 1
    usage = router.usage()
    assert usage["fake:small"]["calls"] == 1 and usage["fake:big"]["output_tokens"] == 20


def test_writing_failure_falls_back_to_the_research_model():
    research = FakeProvider({"write": "local draft"}, model="small")
    writing = FakeProvider({"write": LLMError("paid API down")}, model="big")
    warnings = []
    router = LLMRouter(research, writing, emit=lambda level, msg: warnings.append(msg))
    assert router.ask_text("summary", "write this") == "local draft"
    assert warnings and "paid API down" in warnings[0]


def test_invalid_json_is_retried_then_given_up():
    provider = FakeProvider({"go": "not json at all"})
    router = LLMRouter(provider, provider)
    assert router.ask_json("t", "go", SCHEMA) is None
    assert len(provider.prompts) == 2 and router.usage()["fake:fake-1"]["calls"] == 2


def test_build_provider_and_catalog():
    settings = Settings(openai_api_key="", openai_model="")
    assert build_provider("ollama", settings, "custom:1b").model == "custom:1b"
    assert build_provider("mlx", settings).is_local and build_provider("mlx", settings).id == "mlx"
    assert not build_provider("openai", settings).status().available
    with pytest.raises(ValueError):
        build_provider("nope", settings)
    assert [row["id"] for row in provider_catalog(settings)][0] == "offline"


def test_ollama_request_format_and_missing_model(monkeypatch):
    sent = {}

    def fake_post(url, json=None, timeout=None, **_):
        sent.update(url=url, body=json)
        return httpx.Response(200, json={"message": {"content": '{"ok": true}'}, "prompt_eval_count": 11,
                                         "eval_count": 5}, request=httpx.Request("POST", url))

    def fake_get(url, timeout=None, **_):
        return httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    provider = OllamaProvider("llama3.2:3b", "http://localhost:11434", num_ctx=2048)
    result = provider.complete("hi", system="sys", max_tokens=99, json_schema=SCHEMA)
    assert result.text == '{"ok": true}' and (result.input_tokens, result.output_tokens) == (11, 5)
    assert sent["url"].endswith("/api/chat") and sent["body"]["format"] == SCHEMA
    assert sent["body"]["options"] == {"temperature": 0.3, "num_ctx": 2048, "num_predict": 99}
    assert sent["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert provider.status().available
    missing = OllamaProvider("gemma3:4b", "http://localhost:11434").status()
    assert not missing.available and "ollama pull gemma3:4b" in missing.detail


def test_ollama_down_is_reported_not_raised(monkeypatch):
    def refuse(*_a, **_k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "get", refuse)
    monkeypatch.setattr(httpx, "post", refuse)
    provider = OllamaProvider("m", "http://localhost:1")
    assert "ollama serve" in provider.status().detail
    with pytest.raises(LLMError):
        provider.complete("hi")


def test_openai_compatible_drops_parameters_a_server_rejects(monkeypatch):
    bodies = []

    def fake_post(url, json=None, headers=None, timeout=None, **_):
        bodies.append(dict(json))
        request = httpx.Request("POST", url)
        if "max_tokens" in json:
            return httpx.Response(400, text="Unsupported parameter: 'max_tokens'", request=request)
        if "temperature" in json:
            return httpx.Response(400, text="'temperature' does not support 0.3", request=request)
        assert headers == {"Authorization": "Bearer k"}
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 2}}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = OpenAICompatProvider("openai", "OpenAI", "some-model", "https://api.example/v1", "k")
    result = provider.complete("hi", max_tokens=100)
    assert result.text == "hello" and result.output_tokens == 2
    assert len(bodies) == 3 and "max_completion_tokens" in bodies[-1] and "temperature" not in bodies[-1]


class _FakeClaude(BaseHTTPRequestHandler):
    seen: list = []
    reply: dict = {}

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).seen.append({"path": self.path, "body": body, "beta": self.headers.get("anthropic-beta")})
        data = json.dumps(type(self).reply).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


@pytest.fixture
def fake_claude(monkeypatch):
    pytest.importorskip("anthropic")
    _FakeClaude.seen = []
    server = HTTPServer(("127.0.0.1", 0), _FakeClaude)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    yield _FakeClaude
    server.shutdown()


def _message(text: str, stop: str = "end_turn") -> dict:
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": text}],
            "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": 40, "output_tokens": 12}}


def test_claude_request_uses_structured_output_effort_and_fallbacks(fake_claude):
    fake_claude.reply = _message('{"name": "x", "kind": "a", "tags": [], "n": 1}')
    provider = AnthropicProvider("claude-opus-5-5", effort="high")
    assert provider.status().available
    result = provider.complete("extract", system="be brief", max_tokens=300, json_schema=SCHEMA)
    assert json.loads(result.text)["name"] == "x" and (result.input_tokens, result.output_tokens) == (40, 12)
    call = fake_claude.seen[-1]
    body = call["body"]
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 16000 and body["system"] == "be brief"
    assert body["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert body["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in call["beta"]
    assert "temperature" not in body and "thinking" not in body  # both are rejected by current models


def test_claude_other_models_skip_fallbacks_and_refusals_raise(fake_claude):
    fake_claude.reply = _message("fine")
    AnthropicProvider("claude-haiku-5-5", effort="").complete("hi")
    body = fake_claude.seen[-1]["body"]
    assert "fallbacks" not in body and "output_config" not in body
    fake_claude.reply = _message("", stop="refusal")
    with pytest.raises(LLMError, match="declined"):
        AnthropicProvider("claude-opus-5-5").complete("hi")


def test_claude_without_credentials_is_not_ready(monkeypatch, tmp_path):
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    status = AnthropicProvider().status()
    assert not status.available and "ANTHROPIC_API_KEY" in status.detail


def test_env_file_edits_apply_without_restart_but_real_env_wins(tmp_path, monkeypatch):
    import os

    from adstrategist import config
    monkeypatch.setattr(config, "_FROM_FILE", set())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "from-shell")
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=\nOPENAI_MODEL=from-file\n")
    config.load_env(env)
    assert os.environ["OPENAI_API_KEY"] == "" and os.environ["OPENAI_MODEL"] == "from-shell"
    env.write_text('OPENAI_API_KEY="sk-added-later"\nOPENAI_MODEL=from-file\n')
    config.load_env(env)
    assert os.environ["OPENAI_API_KEY"] == "sk-added-later" and os.environ["OPENAI_MODEL"] == "from-shell"
    monkeypatch.delenv("OPENAI_API_KEY")


def test_openai_status_guides_from_no_key_to_ready(monkeypatch):
    def fake_get(url, headers=None, timeout=None, **_):
        request = httpx.Request("GET", url)
        if headers != {"Authorization": "Bearer good"}:
            return httpx.Response(401, json={"error": "bad key"}, request=request)
        return httpx.Response(200, json={"data": [{"id": "chat-old", "created": 1}, {"id": "chat-new", "created": 9},
                                                  {"id": "text-embedding-3", "created": 5}]}, request=request)
    monkeypatch.setattr(httpx, "get", fake_get)
    make = lambda key, model: OpenAICompatProvider("openai", "OpenAI", model, "https://api.example/v1", key)
    assert "No API key" in make("", "m").status().detail
    assert "rejected" in make("bad", "m").status().detail
    waiting = make("good", "").status()
    assert not waiting.available and "OPENAI_MODEL" in waiting.detail
    assert make("good", "chat-new").status().available
    assert make("good", "").available_models() == ["chat-new", "chat-old"]  # newest first, no embedding models
    assert make("bad", "").available_models() == []
