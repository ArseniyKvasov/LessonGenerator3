from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import Any

import httpx

from config import settings


GROQ_MODELS = (
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
)

_DEFAULT_RETRY_AFTER_SECONDS = 60.0
_MODEL_UNAVAILABLE_UNTIL: dict[str, float] = {}
_MODEL_LOCK = threading.Lock()


def _chat_completions_url() -> str:
    return f"{settings.groq_base_url.rstrip('/')}/chat/completions"


def _now() -> float:
    return time.time()


def _available_models() -> list[str]:
    current_time = _now()
    with _MODEL_LOCK:
        return [
            model
            for model in GROQ_MODELS
            if _MODEL_UNAVAILABLE_UNTIL.get(model, 0) <= current_time
        ]


def models_available() -> bool:
    return bool(settings.groq_api_key and _available_models())


def _mark_model_rate_limited(model: str, retry_after_seconds: float) -> None:
    unavailable_until = _now() + max(retry_after_seconds, 0)

    with _MODEL_LOCK:
        _MODEL_UNAVAILABLE_UNTIL[model] = max(
            _MODEL_UNAVAILABLE_UNTIL.get(model, 0),
            unavailable_until,
        )


def _parse_retry_after_seconds(response: httpx.Response) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    # Groq documents retry-after as the source of truth, but the error body
    # often also says something like "Please try again in 2.3s".
    match = re.search(
        r"(?:try again in|retry after)\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds|s|sec|seconds|m|minutes)?",
        response.text,
        flags=re.IGNORECASE,
    )
    if not match:
        return _DEFAULT_RETRY_AFTER_SECONDS

    value = float(match.group(1))
    unit = (match.group(2) or "s").lower()

    if unit in {"ms", "milliseconds"}:
        return value / 1000
    if unit in {"m", "minutes"}:
        return value * 60
    return value


def _extract_json_content(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    first_brace = content.find("{")
    last_brace = content.rfind("}")

    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return content

    try:
        return json.loads(content[first_brace : last_brace + 1])
    except json.JSONDecodeError:
        return content


def call_ai(prompt: str) -> dict[str, Any]:
    if not settings.groq_api_key:
        return {
            "status": "error",
            "error": "GROQ_API_KEY is not set.",
        }

    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }

    models = _available_models()
    if not models:
        return {
            "status": "error",
            "error": "All Groq models are currently rate limited.",
        }

    random.shuffle(models)
    last_error = None

    with httpx.Client(timeout=settings.groq_timeout_seconds) as client:
        for selected_model in models:
            payload = {
                "model": selected_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You create concise ESL lesson briefs and return only valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": settings.groq_temperature,
                "max_tokens": settings.groq_max_tokens,
                "response_format": {"type": "json_object"},
                "stream": False,
            }

            try:
                response = client.post(_chat_completions_url(), headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    retry_after_seconds = _parse_retry_after_seconds(exc.response)
                    _mark_model_rate_limited(selected_model, retry_after_seconds)
                    last_error = (
                        f"Groq model {selected_model} is rate limited. "
                        f"Retry after {retry_after_seconds:g} seconds."
                    )
                    continue

                return {
                    "status": "error",
                    "error": f"Groq API error {exc.response.status_code}: {exc.response.text}",
                    "model": selected_model,
                }
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                return {
                    "status": "error",
                    "error": f"Groq request failed: {exc}",
                    "model": selected_model,
                }

            break
        else:
            return {
                "status": "error",
                "error": last_error or "All Groq models are currently rate limited.",
            }

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return {
            "status": "error",
            "error": "Groq response has unexpected format.",
        }

    if not isinstance(content, str):
        return {
            "status": "error",
            "error": "Groq response content is not a string.",
        }

    return {
        "status": "ok",
        "response": _extract_json_content(content),
        "model": selected_model,
    }
