import json
import os
from collections.abc import Generator

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_CLOUD_URL = os.environ.get("OLLAMA_CLOUD_URL", "https://ollama.com").rstrip("/")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "")
# Janela de contexto local: o default do Ollama (4096) trunca o INÍCIO do
# prompt (system: identidade + fatos + exemplos) quando o contexto enche.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))


def _is_gpt_oss(model: str) -> bool:
    return "gpt-oss" in model.lower()


def get_reasoning_profile(model: str, provider: str = "local") -> str:
    if provider == "cloud-direct":
        return "cloud padrão"
    if _is_gpt_oss(model):
        return "local baixo"
    return "local desligado"


def _local_think_value(model: str) -> bool | str:
    if _is_gpt_oss(model):
        return "low"
    return False


def _parse_stream(resp: requests.Response) -> Generator[str]:
    for raw in resp.iter_lines():
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("error"):
            raise RuntimeError(str(event["error"]))
        msg = event.get("message", {})
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if content:
            yield str(content)
        if event.get("done"):
            break


def stream_chat(
    model: str,
    messages: list[dict[str, str]],
    provider: str = "local",
    api_key: str = "",
) -> Generator[str]:
    if not model:
        raise RuntimeError("Nenhum modelo selecionado.")

    payload: dict = {"model": model, "messages": messages, "stream": True}

    if provider == "cloud-direct":
        if not api_key:
            raise RuntimeError("Informe a OLLAMA_API_KEY para usar cloud.")
        resp = requests.post(
            f"{OLLAMA_CLOUD_URL}/api/chat",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            stream=True,
            timeout=(10, None),
        )
    else:
        payload["think"] = _local_think_value(model)
        payload["options"] = {"num_ctx": OLLAMA_NUM_CTX}
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=(10, None),
        )

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = resp.text[:400] if resp.text else ""
        raise RuntimeError(f"{exc}" + (f" — {detail}" if detail else "")) from exc
    yield from _parse_stream(resp)


def generate_text(
    model: str,
    prompt: str,
    timeout: int = 30,
    options: "dict | None" = None,
    keep_alive: "str | None" = None,
) -> str:
    try:
        payload: dict = {"model": model, "prompt": prompt, "stream": False}
        payload["think"] = _local_think_value(model)
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=timeout,
        )
        if resp.ok:
            return resp.json().get("response", "").strip()
    except Exception:
        pass
    return ""


def list_local_models() -> list[str]:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        if not isinstance(models, list):
            return []
        return [
            str(m["name"]) for m in models
            if isinstance(m, dict) and "name" in m
            and not m["name"].strip().lower().endswith("-cloud")
        ]
    except Exception:
        return []


def list_cloud_models(api_key: str) -> list[str]:
    if not api_key:
        raise RuntimeError("Informe a OLLAMA_API_KEY.")
    resp = requests.get(
        f"{OLLAMA_CLOUD_URL}/api/tags",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    resp.raise_for_status()
    models = resp.json().get("models", [])
    if not isinstance(models, list):
        return []
    return [str(m["name"]) for m in models if isinstance(m, dict) and "name" in m]


def get_loaded_models() -> list[str]:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        if not resp.ok:
            return []
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def is_response_truncated(text: str) -> bool:
    """Heuristic: detect responses that ended abruptly without a natural conclusion."""
    if not text or len(text) < 80:
        return False
    stripped = text.rstrip()
    # Unclosed code block
    if stripped.count("```") % 2 != 0:
        return True
    last_char = stripped[-1] if stripped else ""
    if last_char in ".!?:\"')]}":
        return False
    # Last line is long and doesn't look like a list item or heading
    last_line = stripped.split("\n")[-1].strip()
    if len(last_line) > 25 and not last_line.startswith(("#", "-", "*", ">", "|")):
        return True
    return False
