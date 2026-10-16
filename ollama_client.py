"""
Ollama client — supports local Ollama OR a remote Kaggle/Colab tunnel.

Set OLLAMA_BASE_URL in your environment (Railway dashboard) to point at your
Kaggle notebook's public URL, e.g.:
    OLLAMA_BASE_URL=https://xxxx-34-x-x-x.ngrok-free.app

If not set, falls back to localhost:11434 (works when running locally).
"""

import json
import os
import re
import time
import urllib.request
import urllib.error

# ── Config ────────────────────────────────────────────────────────────────────
_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_URL       = f"{_BASE}/api/generate"
OLLAMA_HEALTH    = _BASE          # GET / → 200 means alive
DEFAULT_MODEL    = os.getenv("OLLAMA_MODEL", "mistral")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def is_ollama_running() -> bool:
    """Return True if the configured Ollama endpoint is reachable."""
    try:
        urllib.request.urlopen(OLLAMA_HEALTH, timeout=5)
        return True
    except Exception:
        return False


# ── Core call (used by national_id extractor) ─────────────────────────────────
def ollama_generate(payload: dict, timeout: int = 180) -> dict:
    """
    POST *payload* to /api/generate and return the parsed response body.
    Raises RuntimeError on network failure or non-200 status.
    """
    # Health-check first so error messages are clear
    if not is_ollama_running():
        raise RuntimeError(
            f"Ollama health check failed — endpoint unreachable: {OLLAMA_HEALTH}\n"
            "Set OLLAMA_BASE_URL env-var on Railway to your Kaggle tunnel URL."
        )

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama request failed after {time.time()-t0:.1f}s: {exc}\n"
            f"Endpoint: {OLLAMA_URL}"
        )

    return body


# ── Convenience wrapper (text-only prompts) ───────────────────────────────────
def ollama_extract(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 120) -> dict:
    """
    Send a text prompt and return parsed JSON dict.
    Raises ValueError if the response is not valid JSON.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 2048},
    }
    body = ollama_generate(payload, timeout=timeout)
    raw  = body.get("response", "")
    clean = _strip_fences(raw)

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Ollama did not return valid JSON.\nRaw:\n{raw[:500]}")
