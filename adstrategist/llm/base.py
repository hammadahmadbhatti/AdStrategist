"""Provider interface plus the JSON helpers every provider shares."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


class LLMError(RuntimeError):
    """Raised when a provider cannot return usable output."""


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ProviderStatus:
    available: bool
    detail: str


class LLMProvider:
    id = "base"
    label = "Base"
    is_local = True

    def __init__(self, model: str = "") -> None:
        self.model = model

    def status(self) -> ProviderStatus:
        raise NotImplementedError

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024,
                 json_schema: dict | None = None) -> LLMResult:
        raise NotImplementedError


def extract_json(text: str) -> dict:
    """Parse a JSON object out of model output, tolerating fences and chatter."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("model did not return JSON")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"model returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError("model returned JSON that is not an object")
    return data


def skeleton(schema: dict, repeat: dict | None = None, key: str = ""):
    """A fill-in-the-blanks example of the schema. Small models follow this better than raw JSON Schema.

    Small models also copy the example's length, so `repeat` says how many items each list should
    show, by property name (for example {"scenes": 5}). Lists default to two items.
    """
    repeat = repeat or {}
    kind = schema.get("type")
    if kind == "object":
        return {k: skeleton(v, repeat, k) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        item = schema.get("items", {"type": "string"})
        return [skeleton(item, repeat, key) for _ in range(repeat.get(key, 2))]
    if "enum" in schema:
        return " | ".join(str(v) for v in schema["enum"])
    if kind in ("number", "integer"):
        return 0
    if kind == "boolean":
        return False
    return schema.get("description", "...")


def coerce(data, schema: dict):
    """Force model output into the schema's shape so later code never crashes on it."""
    kind = schema.get("type")
    if kind == "object":
        src = data if isinstance(data, dict) else {}
        return {k: coerce(src.get(k), v) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        if isinstance(data, str):
            data = [data] if data.strip() else []
        items = data if isinstance(data, list) else []
        item_schema = schema.get("items", {"type": "string"})
        out = [coerce(i, item_schema) for i in items]
        return [i for i in out if i not in ("", None, {}, [])]
    if kind in ("number", "integer"):
        try:
            return float(data) if kind == "number" else int(float(data))
        except (TypeError, ValueError):
            return 0
    if kind == "boolean":
        return bool(data)
    value = "" if data is None or isinstance(data, (dict, list)) else str(data).strip()
    if "enum" in schema and value not in schema["enum"]:
        lowered = {str(e).lower(): e for e in schema["enum"]}
        return lowered.get(value.lower(), schema["enum"][-1])
    return value
