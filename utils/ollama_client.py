
"""Ollama client for local Ollama or an authenticated remote proxy."""

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request


DEFAULT_MODEL = "qwen2.5vl:3b"


def _base_url() -> str:
    return os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ).strip().rstrip("/")


def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return headers


def ollama_generate(payload: dict, timeout: int = 240) -> dict:
    """Send a raw /api/generate payload to the configured Ollama service."""
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dictionary")

    if timeout <= 0:
        raise ValueError("Timeout must be positive")

    request = urllib.request.Request(
        f"{_base_url()}/api/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_headers(),
        method="POST",
    )

    start = time.monotonic()

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")

    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"Ollama HTTP {exc.code}: {details}"
        ) from exc

    except (
        urllib.error.URLError,
        TimeoutError,
        socket.timeout,
        OSError,
    ) as exc:
        elapsed = time.monotonic() - start
        raise RuntimeError(
            f"Ollama request failed after {elapsed:.1f}s: {exc}"
        ) from exc

    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid API JSON") from exc

    if not isinstance(body, dict):
        raise RuntimeError("Ollama response must be a JSON object")

    if body.get("error"):
        raise RuntimeError(f"Ollama API error: {body['error']}")

    return body


def _extract_json_object(text: str) -> dict:
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = cleaned.find("{")

        if start < 0:
            raise ValueError("Ollama response contains no JSON object")

        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Ollama returned malformed or incomplete JSON"
            ) from exc

    if not isinstance(obj, dict):
        raise ValueError("Ollama JSON root must be an object")

    return obj


def ollama_extract(
    prompt: str,
    model: str | None = None,
    timeout: int = 240,
) -> dict:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be non-empty")

    effective_model = (
        os.getenv("OLLAMA_MODEL", "").strip()
        or model
        or DEFAULT_MODEL
    )

    payload = {
        "model": effective_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "2m",
        "options": {
            "temperature": 0,
            "num_predict": 1200,
        },
    }

    start = time.monotonic()
    body = ollama_generate(payload, timeout=timeout)
    elapsed = time.monotonic() - start

    answer = body.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Ollama returned an empty response")

    print(
        f"    [ollama] Model={effective_model}; "
        f"completed in {elapsed:.1f}s; output chars={len(answer)}"
    )

    return _extract_json_object(answer)


def is_ollama_running() -> bool:
    """Check the remote proxy health or local Ollama tags endpoint."""
    base_url = _base_url()
    remote = bool(os.getenv("OLLAMA_BASE_URL", "").strip())

    url = f"{base_url}/health" if remote else f"{base_url}/api/tags"

    # The proxy's /health endpoint does not require authentication.
    headers = {} if remote else _headers()
    request = urllib.request.Request(url, headers=headers, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status == 200
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        OSError,
    ):
        return False
