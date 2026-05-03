from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from config import settings


MAX_AUDIO_INPUT_LENGTH = 4096


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}

    if settings.pollinations_api_key:
        headers["Authorization"] = f"Bearer {settings.pollinations_api_key}"

    return headers


def _url(path: str) -> str:
    return f"{settings.pollinations_base_url.rstrip('/')}{path}"


def _error(message: str) -> dict[str, str]:
    return {"status": "error", "error": message}


def _script_to_text(mode: str, script: list[dict[str, str]]) -> str:
    lines = []

    for item in script:
        if not isinstance(item, dict):
            continue

        speaker = item.get("speaker", "")
        text = item.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        clean_text = text.strip()

        if mode == "dialogue" and isinstance(speaker, str) and speaker.strip():
            lines.append(f"{speaker.strip()}: {clean_text}")
        else:
            lines.append(clean_text)

    return "\n".join(lines)


def _extract_image_base64(data: Any) -> str:
    if not isinstance(data, dict):
        return ""

    images = data.get("data")

    if not isinstance(images, list) or not images:
        return ""

    first_image = images[0]

    if not isinstance(first_image, dict):
        return ""

    image_base64 = first_image.get("b64_json")

    if isinstance(image_base64, str) and image_base64.strip():
        return image_base64.strip()

    return ""


def generate_image(description: str) -> dict[str, Any]:
    if not isinstance(description, str) or not description.strip():
        return _error("Image description is required.")

    payload = {
        "model": settings.pollinations_image_model,
        "prompt": description.strip(),
        "n": 1,
        "size": settings.pollinations_image_size,
        "quality": settings.pollinations_image_quality,
        "response_format": "b64_json",
    }

    try:
        with httpx.Client(timeout=settings.pollinations_timeout_seconds) as client:
            response = client.post(
                _url("/v1/images/generations"),
                headers=_headers(),
                json=payload,
            )
            response.raise_for_status()
            image_base64 = _extract_image_base64(response.json())
    except httpx.HTTPStatusError as exc:
        return _error(f"Pollinations image API error {exc.response.status_code}: {exc.response.text}")
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return _error(f"Pollinations image request failed: {exc}")

    if not image_base64:
        return _error("Pollinations image response has unexpected format.")

    return {
        "status": "ok",
        "image_base64": image_base64,
        "model": settings.pollinations_image_model,
    }


def generate_audio(mode: str, script: list[dict[str, str]]) -> dict[str, Any]:
    if mode not in {"monologue", "dialogue"}:
        return _error("Audio mode must be monologue or dialogue.")

    text = _script_to_text(mode, script)

    if not text:
        return _error("Audio script is empty.")

    payload = {
        "model": settings.pollinations_audio_model,
        "input": text[:MAX_AUDIO_INPUT_LENGTH],
        "voice": settings.pollinations_audio_voice,
        "response_format": settings.pollinations_audio_format,
        "speed": 1,
    }

    try:
        with httpx.Client(timeout=settings.pollinations_timeout_seconds) as client:
            response = client.post(
                _url("/v1/audio/speech"),
                headers=_headers(),
                json=payload,
            )
            response.raise_for_status()
            audio_base64 = base64.b64encode(response.content).decode("ascii")
    except httpx.HTTPStatusError as exc:
        return _error(f"Pollinations audio API error {exc.response.status_code}: {exc.response.text}")
    except httpx.HTTPError as exc:
        return _error(f"Pollinations audio request failed: {exc}")

    if not audio_base64:
        return _error("Pollinations audio response is empty.")

    return {
        "status": "ok",
        "audio_base64": audio_base64,
        "model": settings.pollinations_audio_model,
    }
