import json
import math
from typing import Any

from ai_engine import generate_text
from db import get_db, count_user_messages, upsert_global_insight, get_global_insights
from utils import tokenize, similarity, now_iso, new_id, chunk_text

RETRIEVAL_CHAR_BUDGET = 5200


def estimate_tokens(text: str) -> int:
    # UTF-8 byte length divided by 4 is more accurate than char count for Unicode
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def remember(
    conv_id: str,
    message_id: str,
    role: str,
    content: str,
    importance: float = 1.0,
) -> None:
    chunks = chunk_text(content)
    if not chunks:
        return
    with get_db() as conn:
        ts = now_iso()
        for index, chunk in enumerate(chunks):
            tokens = tokenize(chunk)
            if not tokens:
                continue
            chunk_importance = importance * (0.92 if index else 1.0)
            conn.execute(
                "INSERT INTO memories "
                "(id, conversation_id, message_id, source_role, content, tokens, importance, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (new_id(), conv_id, message_id, role, chunk,
                 json.dumps(tokens[:450], ensure_ascii=False), chunk_importance, ts),
            )


def retrieve(
    prompt: str,
    limit: int = 10,
    char_budget: int = RETRIEVAL_CHAR_BUDGET,
) -> list[dict[str, Any]]:
    prompt_tokens = tokenize(prompt)
    # Don't retrieve memories for trivial/short messages — avoids polluting context
    if not prompt_tokens or len(prompt.strip()) < 8:
        return []

    with get_db() as conn:
        mem_rows = conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
        insight_rows = conn.execute(
            "SELECT id, conversation_id, 'insight' AS source_role, content, tokens, "
            "2.0 AS importance, 0 AS access_count, NULL AS last_accessed, created_at "
            "FROM insights ORDER BY created_at DESC LIMIT 40"
        ).fetchall()

    # Global insights: retrieved separately and given source_role="global_insight"
    global_rows_raw = get_global_insights(prompt_tokens, limit=6)
    global_rows = [
        {**r, "source_role": "global_insight", "importance": 3.0,
         "access_count": r.get("access_count", 0), "last_accessed": None}
        for r in global_rows_raw
    ]

    def score(row: Any, base_imp: float = 1.0) -> float:
        try:
            toks = [str(t) for t in json.loads(str(row["tokens"] or "[]"))]
        except (json.JSONDecodeError, TypeError):
            toks = tokenize(str(row["content"]))
        base = similarity(prompt_tokens, toks)
        if base <= 0:
            return 0.0
        imp = float(row["importance"] or base_imp)
        acc = int(row["access_count"] or 0)
        return base * imp * (1 + math.log1p(acc))

    ranked: list[tuple[float, dict[str, Any], list[str]]] = []

    for row in mem_rows:
        s = score(row)
        if s > 0:
            try:
                toks = [str(t) for t in json.loads(str(row["tokens"] or "[]"))]
            except (json.JSONDecodeError, TypeError):
                toks = tokenize(str(row["content"]))
            ranked.append((s, dict(row), toks))

    for row in insight_rows:
        s = score(row, base_imp=2.0)
        if s > 0:
            try:
                toks = [str(t) for t in json.loads(str(row["tokens"] or "[]"))]
            except (json.JSONDecodeError, TypeError):
                toks = tokenize(str(row["content"]))
            ranked.append((s, dict(row), toks))

    # Global insights: only include when there's a real query to match against
    for row in global_rows:
        s = score(row, base_imp=3.0)
        # Only apply floor when the prompt has enough substance to warrant global context
        if prompt_tokens and len(prompt_tokens) >= 2:
            s = max(s, 0.08)
        if s <= 0:
            continue
        try:
            toks = [str(t) for t in json.loads(str(row["tokens"] or "[]"))]
        except (json.JSONDecodeError, TypeError):
            toks = tokenize(str(row["content"]))
        ranked.append((s, row, toks))

    ranked.sort(key=lambda x: x[0], reverse=True)

    selected: list[dict[str, Any]] = []
    selected_tokens: list[list[str]] = []
    used_chars = 0
    seen_messages: set[str] = set()

    for raw_score, row, toks in ranked:
        content = str(row.get("content", ""))
        if not content:
            continue
        msg_id = str(row.get("message_id") or row.get("id") or "")
        diversity_penalty = 0.82 if msg_id in seen_messages else 1.0
        redundancy = max((similarity(toks, prev) for prev in selected_tokens), default=0.0)
        adjusted_score = raw_score * diversity_penalty * (1.0 - min(redundancy, 0.65))
        if adjusted_score <= 0:
            continue
        if selected and used_chars + len(content) > char_budget:
            continue
        row["rag_score"] = round(adjusted_score, 4)
        row["estimated_tokens"] = estimate_tokens(content)
        selected.append(row)
        selected_tokens.append(toks)
        used_chars += len(content)
        if msg_id:
            seen_messages.add(msg_id)
        if len(selected) >= limit:
            break

    top = selected

    # Update access counts for conversation memories (not insights or globals)
    ids = [m["id"] for m in top if m.get("source_role") not in ("insight", "global_insight")]
    if ids:
        ts = now_iso()
        with get_db() as conn:
            for mid in ids:
                conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    (ts, mid),
                )
    return top


def maybe_consolidate(conv_id: str, model: str) -> None:
    """Generate an insight from the most accessed memories every 10 user messages.
    The insight is stored per-conversation AND promoted to the global knowledge base.
    """
    count = count_user_messages(conv_id)
    if count == 0 or count % 10 != 0:
        return

    with get_db() as conn:
        mems = conn.execute(
            "SELECT id, content FROM memories WHERE conversation_id = ? "
            "ORDER BY access_count DESC, created_at DESC LIMIT 20",
            (conv_id,),
        ).fetchall()

    if len(mems) < 4:
        return

    # Skip consolidation if most memories are trivial (short greetings, random chars)
    substantial = [m for m in mems if len(str(m["content"]).strip()) >= 20]
    if len(substantial) < 3:
        return

    mem_text = "\n".join(f"- {str(m['content'])[:250]}" for m in substantial)
    # Prompt tuned for small (4B) models: direct, single instruction, fact-focused
    prompt = (
        "Leia as mensagens abaixo e escreva de 1 a 3 fatos concretos sobre o usuário "
        "(nome, profissão, documentos mencionados, preferências explícitas, tarefas recorrentes). "
        "Use frases curtas: 'O usuário é...', 'O usuário usa...', 'O usuário tem...'. "
        "Se não houver fatos concretos, escreva apenas: NADA.\n\n"
        + mem_text
    )

    insight = generate_text(model, prompt, timeout=30)
    if len(insight) < 20 or "NADA" in insight.upper()[:20]:
        return

    tokens = tokenize(insight)
    source_ids = [str(m["id"]) for m in mems]

    with get_db() as conn:
        conn.execute(
            "INSERT INTO insights (id, conversation_id, content, tokens, created_at) VALUES (?,?,?,?,?)",
            (new_id(), conv_id, insight,
             json.dumps(tokens[:200], ensure_ascii=False), now_iso()),
        )
        for mid in source_ids[:10]:
            conn.execute(
                "UPDATE memories SET importance = MIN(importance + 0.2, 3.0) WHERE id = ?",
                (mid,),
            )

    # Promote to global knowledge base (cross-conversation)
    upsert_global_insight(insight)
