"""
Memória declarativa: fatos permanentes sobre o usuário e o contexto,
lidos de data/facts.json e injetados no system prompt em toda conversa.

Diferente do RAG (que depende da pergunta), estes fatos estão SEMPRE
disponíveis — o usuário nunca precisa re-explicar quem é ou onde trabalha.
Edite data/facts.json livremente; o arquivo é recarregado quando muda.
"""

import json
from pathlib import Path
from typing import Any

FACTS_PATH = Path(__file__).parent / "data" / "facts.json"

_cache: dict[str, Any] = {"mtime": 0.0, "block": ""}


def _flatten(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, val in value.items():
            label = key.replace("_", " ")
            if isinstance(val, (dict, list)):
                lines.extend(_flatten(val, f"{prefix}{label} > "))
            else:
                lines.append(f"- {prefix}{label}: {val}")
    elif isinstance(value, list):
        items = ", ".join(str(v) for v in value)
        lines.append(f"- {prefix.rstrip(' >')}: {items}")
    else:
        lines.append(f"- {prefix.rstrip(' >')}: {value}")
    return lines


def get_facts_block() -> str:
    """Retorna os fatos formatados em linhas curtas, ou '' se não houver arquivo.
    Cache por mtime — zero custo quando o arquivo não muda."""
    try:
        mtime = FACTS_PATH.stat().st_mtime
    except OSError:
        return ""

    if mtime == _cache["mtime"]:
        return str(_cache["block"])

    try:
        data = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
        block = "\n".join(_flatten(data)) if data else ""
    except (json.JSONDecodeError, OSError):
        block = ""

    _cache["mtime"] = mtime
    _cache["block"] = block
    return block
