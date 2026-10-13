"""Local Ollama JSON extraction client."""
import json
import re
import socket
import time
import urllib.error
import urllib.request

OLLAMA_URL = 'http://localhost:11434/api/generate'
DEFAULT_MODEL = 'mistral:latest'


def _extract_json_object(text: str) -> dict:
    cleaned = re.sub(r'^\s*```(?:json)?\s*', '', text.strip(), flags=re.I)
    cleaned = re.sub(r'\s*```\s*$', '', cleaned).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        pos = cleaned.find('{')
        if pos < 0:
            raise ValueError('Ollama response contains no JSON object')
        try:
            obj, _ = decoder.raw_decode(cleaned[pos:])
        except json.JSONDecodeError as exc:
            raise ValueError('Ollama returned malformed or incomplete JSON') from exc
    if not isinstance(obj, dict):
        raise ValueError('Ollama JSON root must be an object')
    return obj


def ollama_extract(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 240) -> dict:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Prompt must be non-empty')
    if not isinstance(model, str) or not model.strip():
        raise ValueError('Model must be non-empty')
    if timeout <= 0:
        raise ValueError('Timeout must be positive')

    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'format': 'json',
        'keep_alive': '10m',
        'options': {'temperature': 0, 'num_predict': 1200},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as exc:
        details = exc.read().decode('utf-8', errors='replace')[:300]
        raise RuntimeError(f'Ollama HTTP {exc.code}: {details}') from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        elapsed = time.monotonic() - start
        raise RuntimeError(
            f'Ollama request failed after {elapsed:.1f}s: {exc}. '
            'The server may be running but generation may be too slow.'
        ) from exc
    elapsed = time.monotonic() - start
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError('Ollama returned invalid API JSON') from exc
    if not isinstance(body, dict):
        raise RuntimeError('Ollama API response must be a JSON object')
    if body.get('error'):
        raise RuntimeError(f"Ollama API error: {body['error']}")
    answer = body.get('response')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('Ollama returned an empty response')
    print(f'    [ollama] Completed in {elapsed:.1f}s; output chars={len(answer)}')
    return _extract_json_object(answer)


def is_ollama_running() -> bool:
    try:
        with urllib.request.urlopen('http://localhost:11434/api/tags', timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
