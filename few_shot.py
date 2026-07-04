"""
Few-shot injection from dataset/ into the system prompt.

Seleção DINÂMICA: os exemplos entram por semelhança com a pergunta atual
(BM25 + embedding nomic, mesma fusão por score do db.py), em vez de injetar
cegamente o 1º exemplo de cada categoria. Menos tokens, exemplos mais úteis.

Fallback estático (1º exemplo por categoria) quando não há prompt, o Ollama
está fora do ar ou nenhum exemplo tem sinal de semelhança.

Embeddings dos exemplos ficam cacheados em data/few_shot_embeddings.json —
computados uma vez por exemplo e reaproveitados entre restarts.
"""

import hashlib
import json
import math
import threading
import time
from functools import lru_cache
from pathlib import Path

from utils import EMBED_MODEL, bm25_scores, cosine_sim, get_embedding, tokenize

DATASET_DIR = Path(__file__).parent / "dataset"
EMB_CACHE_PATH = Path(__file__).parent / "data" / "few_shot_embeddings.json"

# Token budget per provider tier
LOCAL_TOKEN_BUDGET = 1_600
CLOUD_TOKEN_BUDGET = 18_000

# Seleção dinâmica: quantos exemplos entram (1 por categoria escolhida)
LOCAL_MAX_EXAMPLES = 4
CLOUD_MAX_EXAMPLES = 10

_REINDEX_INTERVAL = 300.0  # reteima embeddings faltantes no máx. a cada 5 min

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


# ---------------------------------------------------------------------------
# Índice de exemplos para seleção dinâmica
# ---------------------------------------------------------------------------

_index_lock = threading.Lock()
_index: "list[dict] | None" = None
_index_ts = 0.0


def _stem(token: str) -> str:
    """Stem por prefixo: em pt-BR a flexão fica no fim da palavra
    ('comparando'/'compare' → 'compa', 'revisão'/'revise' → 'revis').
    Grosseiro mas suficiente para casar pergunta ↔ exemplo."""
    return token[:5]


def _example_query_text(record: dict) -> str:
    """Texto que representa o exemplo na busca: só os turnos do usuário —
    a semelhança relevante é pergunta ↔ pergunta, não pergunta ↔ resposta."""
    users = [
        str(m.get("content", "")).strip()
        for m in record.get("messages", [])
        if m.get("role") == "user"
    ]
    return " ".join(u for u in users if u)[:800]


def _load_emb_cache() -> dict:
    try:
        data = json.loads(EMB_CACHE_PATH.read_text(encoding="utf-8"))
        if data.get("model") == EMBED_MODEL and isinstance(data.get("embeddings"), dict):
            return data["embeddings"]
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_emb_cache(embeddings: dict) -> None:
    try:
        EMB_CACHE_PATH.parent.mkdir(exist_ok=True)
        EMB_CACHE_PATH.write_text(
            json.dumps({"model": EMBED_MODEL, "embeddings": embeddings}),
            encoding="utf-8",
        )
    except OSError:
        pass


def _build_index() -> list[dict]:
    cache = _load_emb_cache()
    current: dict[str, list[float]] = {}
    computed = False
    index: list[dict] = []

    for key, label, records in _load_grouped():
        for record in records:
            qtext = _example_query_text(record)
            if not qtext:
                continue
            h = hashlib.sha1(qtext.encode("utf-8")).hexdigest()
            emb = current.get(h) or cache.get(h)
            if emb is None:
                emb = get_embedding(qtext, kind="document")
                if emb is not None:
                    computed = True
            if emb is not None:
                current[h] = emb
            toks = tokenize(qtext)
            index.append({
                "key": key,
                "label": label,
                "record": record,
                "tokens": toks,
                "stems": [_stem(t) for t in toks],
                "emb": emb,
            })

    # Persiste novos embeddings e descarta os de exemplos removidos
    if computed or set(current) != set(cache):
        _save_emb_cache(current)
    return index


def _get_index() -> list[dict]:
    """Índice em memória; retenta embeddings faltantes (Ollama fora do ar
    no boot, por exemplo) no máximo a cada _REINDEX_INTERVAL."""
    global _index, _index_ts
    now = time.time()
    with _index_lock:
        stale = _index is not None and (
            any(e["emb"] is None for e in _index) and now - _index_ts > _REINDEX_INTERVAL
        )
        if _index is None or stale:
            _index = _build_index()
            _index_ts = now
        return _index


def warm_cache() -> None:
    """Pré-computa os embeddings dos exemplos (chamar em thread no startup)."""
    try:
        _get_index()
    except Exception:
        pass


def _select_relevant(
    prompt: str, is_cloud: bool, personality_id: str
) -> "list[dict] | None":
    """Seleção em dois níveis: primeiro a CATEGORIA (BM25 sobre todos os
    exemplos dela juntos — dilui palavra incidental tipo 'dois' ou 'chefe'
    que aparece num exemplo de outro assunto), depois o melhor exemplo
    dentro de cada categoria escolhida. None → usar fallback estático."""
    candidates = [
        e for e in _get_index()
        if PERSONA_ONLY_CATEGORIES.get(e["key"], personality_id) == personality_id
    ]
    q_stems = [_stem(t) for t in tokenize(prompt)]
    if not candidates or not q_stems:
        return None

    q_emb = get_embedding(prompt[:1000], kind="query")

    by_cat: dict[str, list[dict]] = {}
    for e in candidates:
        by_cat.setdefault(e["key"], []).append(e)
    cat_keys = list(by_cat)

    # Nível 1 — score por categoria: BM25 do documento-categoria (stems de
    # todas as perguntas) + cosseno contra o centroide dos embeddings
    cat_docs = [
        [s for e in by_cat[k] for s in e["stems"]]
        for k in cat_keys
    ]
    kw = bm25_scores(q_stems, cat_docs)
    kw_max = max(kw) if kw else 0.0

    centroids: list["list[float] | None"] = []
    for k in cat_keys:
        embs = [e["emb"] for e in by_cat[k] if e["emb"]]
        if embs:
            centroids.append([sum(vals) / len(embs) for vals in zip(*embs)])
        else:
            centroids.append(None)
    cos = [
        cosine_sim(q_emb, c) if (q_emb and c) else None
        for c in centroids
    ]
    valid = [c for c in cos if c is not None and c > 0]
    c_min = min(valid) if valid else 0.0
    c_span = (max(valid) - c_min) if valid else 0.0

    # Mesma fusão por score normalizado do db.py; pesos iguais porque aqui
    # semelhança semântica importa tanto quanto termo exato (é comportamento,
    # não fato — "revisa esse texto" deve casar com qualquer pedido de revisão)
    cat_scored: list[tuple[float, str]] = []
    for i, k in enumerate(cat_keys):
        s_kw = (kw[i] / kw_max) if kw_max > 0 else 0.0
        c = cos[i]
        s_vec = ((c - c_min) / c_span) if (c is not None and c_span > 0) else 0.0
        cat_scored.append((0.5 * s_kw + 0.5 * s_vec, k))
    cat_scored.sort(key=lambda x: x[0], reverse=True)

    if all(s <= 0 for s, _ in cat_scored):
        return None

    # Nível 2 — melhor exemplo da categoria: overlap de stems + cosseno
    def _best_example(key: str) -> dict:
        def example_score(e: dict) -> tuple[float, float]:
            overlap = len(set(q_stems) & set(e["stems"]))
            c = cosine_sim(q_emb, e["emb"]) if (q_emb and e["emb"]) else 0.0
            return (float(overlap), c)
        return max(by_cat[key], key=example_score)

    max_examples = CLOUD_MAX_EXAMPLES if is_cloud else LOCAL_MAX_EXAMPLES
    selected: list[dict] = []
    used_keys: set[str] = set()

    # A categoria de voz da persona entra garantida: estilo não depende do
    # tema da pergunta, então não pode ser eliminada por baixa semelhança
    persona_keys = {k for k, p in PERSONA_ONLY_CATEGORIES.items() if p == personality_id}
    for pk in persona_keys:
        if pk in by_cat:
            selected.append(_best_example(pk))
            used_keys.add(pk)

    # Categoria com score marginal (<15% do topo) é ruído de cauda: um stem
    # incidental compartilhado, não afinidade real — melhor economizar tokens
    top_score = cat_scored[0][0]
    for s, k in cat_scored:
        if len(selected) >= max_examples:
            break
        if s <= 0 or s < 0.15 * top_score or k in used_keys:
            continue
        selected.append(_best_example(k))
        used_keys.add(k)

    return selected or None


def _render_block(selected: list[dict], is_cloud: bool, budget: int) -> str:
    compact = not is_cloud
    parts: list[str] = []
    used = 0
    for e in selected:
        block = _format_example(e["record"], compact=compact)
        if not block:
            continue
        labeled = (f"[{e['label']}]\n" if compact else f"### {e['label']}\n") + block
        block_tokens = _estimate_tokens(labeled)
        if used + block_tokens > budget:
            continue
        parts.append(labeled)
        used += block_tokens
    if not parts:
        return ""
    separator = "\n\n" if compact else "\n\n---\n\n"
    header = (
        "\n\n---\nExemplos de respostas esperadas:\n\n"
        if compact
        else "\n\n---\n## Exemplos de comportamento esperado\n\n"
    )
    return header + separator.join(parts)


# ---------------------------------------------------------------------------
# Montagem do bloco (dinâmico com fallback estático)
# ---------------------------------------------------------------------------

def _build_static_block(is_cloud: bool, personality_id: str) -> str:
    """Comportamento original: 1º exemplo por categoria até estourar o budget."""
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


def build_few_shot_block(
    is_cloud: bool = False,
    personality_id: str = "default",
    prompt: str = "",
) -> str:
    """
    Build the few-shot section to append to the system prompt.
    prompt não-vazio → seleção dinâmica por semelhança (fallback: estático)
    is_cloud=True  → mais exemplos (contexto maior)
    personality_id → exclui categorias presas a outra persona
    """
    if prompt.strip():
        selected = _select_relevant(prompt, is_cloud, personality_id)
        if selected:
            budget = CLOUD_TOKEN_BUDGET if is_cloud else LOCAL_TOKEN_BUDGET
            block = _render_block(selected, is_cloud, budget)
            if block:
                return block
    return _build_static_block(is_cloud, personality_id)
