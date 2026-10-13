"""
Ollama local LLM client.
Sends OCR text to a locally running Ollama model and parses JSON output.

Requirements:
  - Ollama installed: https://ollama.com/download
  - A model pulled, e.g.:
      ollama pull mistral          (7B, ~4GB, text only)
      ollama pull llama3.2-vision  (11B, ~8GB, supports images too)
      ollama pull qwen2.5:7b       (good Arabic support)

Start Ollama before running:  ollama serve
"""

import json
import re
import urllib.request
import urllib.error


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"   # ~4GB, faster on CPU than qwen2.5:7b


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def ollama_extract(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 120) -> dict:
    """
    Send a prompt to Ollama and return parsed JSON.
    Raises RuntimeError if Ollama is not reachable or response is not valid JSON.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic
            "num_predict": 2048,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}.\n"
            "Make sure Ollama is running: `ollama serve`\n"
            f"Original error: {e}"
        )

    raw_text = body.get("response", "")
    clean = _strip_fences(raw_text)

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try to find a JSON object inside the response
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(
            f"Ollama did not return valid JSON.\nRaw response:\n{raw_text[:500]}"
        )


def is_ollama_running() -> bool:
    """Quick check: is Ollama reachable?"""
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=3)
        return True
    except Exception:
        return False
