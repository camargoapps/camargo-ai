"""
Few-shot injection from dataset/ into the system prompt.

Local models (small context):  1 example per category  ≈  1 600 tokens
Cloud models (large context):  all examples            ≈ 16 000 tokens
"""

import json
import math
from functools import lru_cache
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset"

# Token budget per provider tier
LOCAL_TOKEN_BUDGET = 1_600
CLOUD_TOKEN_BUDGET = 18_000

CATEGORY_LABELS: dict[str, str] = {
    "01_cordialidade":          "Cordialidade",
    "02_honestidade_limites":   "Honestidade e Limites",
    "03_correcoes":             "Correções",
    "04_explicacoes_tecnicas":  "Explicações Técnicas",
    "05_revisao_textos":        "Revisão de Textos",
    "06_emails_profissionais":  "E-mails Profissionais",
    "07_programacao":           "Programação",
    "08_atendimento_cliente":   "Atendimento ao Cliente",
    "09_planejamento":          "Planejamento",
    "10_direito_administ":      "Direito e Administração",
    "11_matematica":            "Matemática",
    "12_humor_criatividade":    "Criatividade",
    "13_resumos_comparacoes":   "Resumos e Comparações",
    "14_falta_contexto":        "Contexto Insuficiente",
    "15_multiturno":            "Diálogo Multi-turno",
    "16_atlas_voz":             "Voz e Personalidade",
}

# Categories tied to a specific persona's voice — only injected when that
# persona is active, so other personalities (technical, concise, ...) don't
# get pulled toward Atlas's tone.
PERSONA_ONLY_CATEGORIES: dict[str, str] = {
    "16_atlas_voz": "atlas",
}


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def _file_key(path: Path) -> str:
    return path.stem  # e.g. "01_cordialidade"


def _label(file_key: str) -> str:
    return CATEGORY_LABELS.get(file_key, file_key.replace("_", " ").title())


def _format_example(record: dict, compact: bool = False) -> str:
    """
    Format a single JSONL record as a readable example block.
    compact=True  →  single-line U:/A: pairs (local)
    compact=False →  labeled blocks (cloud)
    """
    msgs = record.get("messages", [])
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(msgs):
        if msgs[i]["role"] == "user":
            user_text = msgs[i]["content"].strip()
            if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
                asst_text = msgs[i + 1]["content"].strip()
                pairs.append((user_text, asst_text))
                i += 2
                continue
        i += 1

    if not pairs:
        return ""

    if compact:
        lines = []
        for u, a in pairs:
            u_short = u[:120] + "…" if len(u) > 120 else u
            lines.append(f"U: {u_short}\nA: {a}")
        return "\n\n".join(lines)
    else:
        lines = []
        for u, a in pairs:
            lines.append(f"Usuário: {u}\nAssistente: {a}")
        return "\n\n".join(lines)


@lru_cache(maxsize=2)
def _load_grouped() -> list[tuple[str, str, list[dict]]]:
    """
    Returns list of (key, label, [records]) sorted by file name.
    Cached after first load — dataset doesn't change at runtime.
    """
    result: list[tuple[str, str, list[dict]]] = []
    files = sorted(DATASET_DIR.glob("*.jsonl"))
    for fpath in files:
        key = _file_key(fpath)
        if key.startswith("00_"):       # skip negative examples
            continue
        records: list[dict] = []
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("bad_example"):
                    continue
                records.append(r)
        if records:
            result.append((key, _label(key), records))
    return result


def build_few_shot_block(is_cloud: bool = False, personality_id: str = "default") -> str:
    """
    Build the few-shot section to append to the system prompt.
    is_cloud=True  → fits more examples (larger context window)
    is_cloud=False → conservative budget (small local models)
    personality_id → excludes persona-locked categories (see PERSONA_ONLY_CATEGORIES)
                      that don't belong to the active persona.
    """
    budget = CLOUD_TOKEN_BUDGET if is_cloud else LOCAL_TOKEN_BUDGET
    compact = not is_cloud
    grouped = [
        (label, records) for key, label, records in _load_grouped()
        if PERSONA_ONLY_CATEGORIES.get(key, personality_id) == personality_id
    ]

    if not grouped:
        return ""

    sections: list[str] = []
    used_tokens = 0

    for label, records in grouped:
        section_parts: list[str] = []
        for record in records:
            block = _format_example(record, compact=compact)
            if not block:
                continue
            block_tokens = _estimate_tokens(block)
            if used_tokens + block_tokens > budget:
                break
            section_parts.append(block)
            used_tokens += block_tokens
            # Local: 1 example per category is enough
            if not is_cloud:
                break

        if section_parts:
            header = f"[{label}]" if compact else f"### {label}"
            sections.append(header + "\n" + "\n\n".join(section_parts))

        if used_tokens >= budget:
            break

    if not sections:
        return ""

    separator = "\n\n" if compact else "\n\n---\n\n"
    header = (
        "\n\n---\nExemplos de respostas esperadas:\n\n"
        if compact
        else "\n\n---\n## Exemplos de comportamento esperado\n\n"
    )
    return header + separator.join(sections)
