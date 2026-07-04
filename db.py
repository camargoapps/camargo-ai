import json
import mimetypes
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from utils import (
    now_iso, new_id, row_to_dict,
    extract_text_preview, chunk_text, tokenize, similarity,
    get_embedding, cosine_sim, bm25_scores, rrf_fuse,
MAX_TEXT_PREVIEW, MAX_DOC_CHARS,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
AGENT_FILES_DIR = DATA_DIR / "agent_files"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
DB_PATH = DATA_DIR / "app.db"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
AGENT_FILES_DIR.mkdir(exist_ok=True)

WORKSPACE_MAX_FILE_BYTES = 5 * 1024 * 1024
WORKSPACE_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".cache", "venv", ".venv", "env", "node_modules",
    "dist", "build", ".next", "target", "data",
}


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                personality TEXT NOT NULL DEFAULT 'default',
                workspace_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attachments (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                original_name TEXT NOT NULL,
                stored_name TEXT NOT NULL,
                content_type TEXT,
                size INTEGER NOT NULL,
                text_preview TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                source_role TEXT NOT NULL,
                content TEXT NOT NULL,
                tokens TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 1.0,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                content TEXT NOT NULL,
                tokens TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS doc_chunks (
                id TEXT PRIMARY KEY,
                attachment_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                tokens TEXT NOT NULL DEFAULT '[]',
                embedding TEXT,
                filename TEXT NOT NULL DEFAULT '',
                doc_type TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (attachment_id) REFERENCES attachments(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS workspace_files (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                attachment_id TEXT NOT NULL,
                abs_path TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'indexed',
                error TEXT,
                indexed_at TEXT NOT NULL,
                UNIQUE(conversation_id, rel_path),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
                FOREIGN KEY (attachment_id) REFERENCES attachments(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS folders (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_files (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                message_id TEXT,
                folder_id TEXT,
                filename TEXT NOT NULL,
                stored_name TEXT NOT NULL,
                mime TEXT NOT NULL DEFAULT '',
                size INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS global_insights (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tokens TEXT NOT NULL DEFAULT '[]',
                access_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                tokens TEXT NOT NULL DEFAULT '[]',
                embedding TEXT,
                mtime REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
        """)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    def cols(table: str) -> set[str]:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    conv_cols = cols("conversations")
    if "personality" not in conv_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN personality TEXT NOT NULL DEFAULT 'default'")
    if "folder_id" not in conv_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN folder_id TEXT")
    if "workspace_path" not in conv_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN workspace_path TEXT")

    mem_cols = cols("memories")
    if "importance" not in mem_cols:
        conn.execute("ALTER TABLE memories ADD COLUMN importance REAL NOT NULL DEFAULT 1.0")
    if "access_count" not in mem_cols:
        conn.execute("ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0")
    if "last_accessed" not in mem_cols:
        conn.execute("ALTER TABLE memories ADD COLUMN last_accessed TEXT")

    ins_cols = cols("insights")
    if ins_cols and "tokens" not in ins_cols:
        conn.execute("ALTER TABLE insights ADD COLUMN tokens TEXT NOT NULL DEFAULT '[]'")

    # doc_chunks migration: add filename column if table existed without it
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='doc_chunks'"
    ).fetchone()
    if table_exists:
        dc_cols = cols("doc_chunks")
        if "filename" not in dc_cols:
            conn.execute("ALTER TABLE doc_chunks ADD COLUMN filename TEXT NOT NULL DEFAULT ''")
        if "embedding" not in dc_cols:
            conn.execute("ALTER TABLE doc_chunks ADD COLUMN embedding TEXT")
        if "doc_type" not in dc_cols:
            conn.execute("ALTER TABLE doc_chunks ADD COLUMN doc_type TEXT NOT NULL DEFAULT ''")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS workspace_files (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            attachment_id TEXT NOT NULL,
            abs_path TEXT NOT NULL,
            rel_path TEXT NOT NULL,
            size INTEGER NOT NULL,
            mtime REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'indexed',
            error TEXT,
            indexed_at TEXT NOT NULL,
            UNIQUE(conversation_id, rel_path),
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
            FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
            FOREIGN KEY (attachment_id) REFERENCES attachments(id) ON DELETE CASCADE
        )
    """)

    # global_insights is created via CREATE TABLE IF NOT EXISTS above — no ALTER needed


def get_folders() -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM folders ORDER BY name ASC").fetchall()
    return [row_to_dict(r) for r in rows]


def create_folder(name: str) -> dict[str, Any]:
    fid = new_id()
    ts = now_iso()
    with get_db() as conn:
        conn.execute("INSERT INTO folders (id, name, created_at) VALUES (?,?,?)", (fid, name, ts))
    return {"id": fid, "name": name, "created_at": ts}


def rename_folder(folder_id: str, name: str) -> None:
    with get_db() as conn:
        conn.execute("UPDATE folders SET name = ? WHERE id = ?", (name, folder_id))


def delete_folder(folder_id: str) -> None:
    with get_db() as conn:
        conn.execute("UPDATE conversations SET folder_id = NULL WHERE folder_id = ?", (folder_id,))
        conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))


def get_conversations() -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM conversations ORDER BY updated_at DESC").fetchall()
    return [row_to_dict(r) for r in rows]


def get_or_create_conversation(
    conv_id: str | None = None,
    title: str = "Nova conversa",
    personality: str = "atlas",
) -> dict[str, Any]:
    with get_db() as conn:
        if conv_id:
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if row:
                return row_to_dict(row)
        cid = new_id()
        ts = now_iso()
        conn.execute(
            "INSERT INTO conversations (id, title, personality, created_at, updated_at) VALUES (?,?,?,?,?)",
            (cid, title, personality, ts, ts),
        )
        return {"id": cid, "title": title, "personality": personality, "created_at": ts, "updated_at": ts}


def update_conversation(conv_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = now_iso()
    clause = ", ".join(f"{k} = ?" for k in fields)
    with get_db() as conn:
        conn.execute(f"UPDATE conversations SET {clause} WHERE id = ?", [*fields.values(), conv_id])


def delete_conversation(conv_id: str) -> None:
    with get_db() as conn:
        files = conn.execute(
            "SELECT stored_name FROM attachments WHERE message_id IN "
            "(SELECT id FROM messages WHERE conversation_id = ?)", (conv_id,)
        ).fetchall()
        conn.execute("DELETE FROM memories WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM insights WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM workspace_files WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM doc_chunks WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    for f in files:
        try:
            (UPLOAD_DIR / str(f["stored_name"])).unlink(missing_ok=True)
        except OSError:
            pass


def get_messages(conv_id: str) -> list[dict[str, Any]]:
    with get_db() as conn:
        conv = conn.execute(
            "SELECT workspace_path FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        has_workspace = bool(str(conv["workspace_path"] or "").strip()) if conv else False
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? "
            "AND role IN ('user', 'assistant') ORDER BY created_at ASC", (conv_id,)
        ).fetchall()
        messages = [row_to_dict(r) for r in rows]
        for msg in messages:
            att = conn.execute(
                "SELECT id, original_name AS name, content_type, size FROM attachments WHERE message_id = ?",
                (msg["id"],),
            ).fetchall()
            msg["attachments"] = [row_to_dict(r) for r in att]
            if msg["role"] == "assistant":
                afiles = conn.execute(
                    "SELECT id, filename, mime, size FROM agent_files WHERE message_id = ? ORDER BY created_at ASC",
                    (msg["id"],),
                ).fetchall()
                msg["agent_files"] = []
                for row in afiles:
                    item = row_to_dict(row)
                    item["save_to_workspace"] = has_workspace
                    msg["agent_files"].append(item)
    return messages


def save_agent_file_to_workspace(file_id: str) -> Path | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT af.*, c.workspace_path "
            "FROM agent_files af "
            "JOIN conversations c ON c.id = af.conversation_id "
            "WHERE af.id = ?",
            (file_id,),
        ).fetchone()

    if not row:
        return None

    workspace_path = str(row["workspace_path"] or "").strip()
    if not workspace_path:
        return None

    root = Path(workspace_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return None

    stored = str(row["stored_name"] or "")
    src = Path(stored) if Path(stored).is_absolute() else (
        AGENT_FILES_DIR / str(row["folder_id"] or "agent") / stored
    )
    if not src.exists():
        legacy_src = AGENT_FILES_DIR / str(row["folder_id"] or "agent") / (
            secure_filename(str(row["filename"] or "")) or Path(stored).name
        )
        if legacy_src.exists():
            src = legacy_src
        else:
            return None

    safe_name = secure_filename(str(row["filename"] or src.name)) or "arquivo"
    dest = root / safe_name
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        index = 1
        while True:
            candidate = root / f"{stem} {index}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            index += 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def store_agent_file(
    conv_id: str,
    message_id: str | None,
    folder_id: str | None,
    filename: str,
    stored_name: str,
    mime: str,
    size: int,
) -> str:
    fid = new_id()
    ts = now_iso()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO agent_files "
            "(id, conversation_id, message_id, folder_id, filename, stored_name, mime, size, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (fid, conv_id, message_id, folder_id, filename, stored_name, mime, size, ts),
        )
    return fid


def get_agent_files(conv_id: str | None = None, folder_id: str | None = None) -> list[dict[str, Any]]:
    with get_db() as conn:
        if conv_id:
            rows = conn.execute(
                "SELECT * FROM agent_files WHERE conversation_id = ? ORDER BY created_at DESC",
                (conv_id,),
            ).fetchall()
        elif folder_id:
            rows = conn.execute(
                "SELECT * FROM agent_files WHERE folder_id = ? ORDER BY created_at DESC",
                (folder_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM agent_files ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
    return [row_to_dict(r) for r in rows]


def add_message(conv_id: str, role: str, content: str) -> str:
    mid = new_id()
    ts = now_iso()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?,?,?,?,?)",
            (mid, conv_id, role, content, ts),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (ts, conv_id))
    return mid


# Metadados de tipo de documento: permitem priorizar chunks do tipo certo
# quando a pergunta menciona explicitamente ("qual lei...", "na tabela salarial...")
_DOC_TYPE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("parecer", ("parecer", "ementa", "relatorio", "relatório", "fundamentacao", "fundamentação")),
    ("lei", ("lei n", "lei nº", "decreto", "portaria", "resolução", "resolucao", "art.", "artigo", "inciso", "parágrafo único")),
    ("tabela", ("tabela", "planilha", ".xlsx", ".csv", "salarial", "vencimento", "referência salarial")),
    ("email", ("assunto:", "prezado", "prezada", "atenciosamente", "cordialmente", "@")),
    ("codigo", (".py", ".js", ".ts", ".java", "def ", "function ", "import ", "class ")),
]

_QUERY_TYPE_HINTS: dict[str, tuple[str, ...]] = {
    "parecer": ("parecer", "pareceres", "ementa"),
    "lei": ("lei", "leis", "decreto", "decretos", "norma", "normas", "portaria", "artigo", "legislacao", "legislação"),
    "tabela": ("tabela", "tabelas", "planilha", "salarial", "vencimentos"),
    "email": ("email", "e-mail", "mensagem"),
    "codigo": ("codigo", "código", "script", "função", "funcao"),
}


def classify_doc_type(filename: str, content: str) -> str:
    """Heurística leve: classifica o chunk pelo nome do arquivo + conteúdo."""
    haystack = (filename + " " + content[:600]).lower()
    best_type, best_hits = "", 0
    for doc_type, keywords in _DOC_TYPE_PATTERNS:
        hits = sum(1 for kw in keywords if kw in haystack)
        if hits > best_hits:
            best_type, best_hits = doc_type, hits
    return best_type if best_hits >= 2 else ""


def _wanted_doc_type(query_text: str) -> str:
    q = query_text.lower()
    for doc_type, hints in _QUERY_TYPE_HINTS.items():
        if any(h in q for h in hints):
            return doc_type
    return ""


def store_doc_chunks(
    attachment_id: str,
    conv_id: str,
    chunks: list[str],
    filename: str = "",
    with_embeddings: bool = True,
) -> None:
    if not chunks:
        return
    ts = now_iso()
    with get_db() as conn:
        for i, chunk in enumerate(chunks):
            toks = tokenize(chunk)
            emb = get_embedding(chunk) if with_embeddings else None
            conn.execute(
                "INSERT INTO doc_chunks "
                "(id, attachment_id, conversation_id, chunk_index, content, tokens, embedding, filename, doc_type, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    new_id(), attachment_id, conv_id, i, chunk,
                    json.dumps(toks[:450], ensure_ascii=False),
                    json.dumps(emb) if emb else None,
                    filename, classify_doc_type(filename, chunk), ts,
                ),
            )


def search_doc_chunks(
    conv_id: str,
    query_tokens: list[str],
    limit: int,
    char_budget: int,
    broad: bool = False,
    query_text: str = "",
) -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM doc_chunks WHERE conversation_id = ? "
            "ORDER BY attachment_id, chunk_index",
            (conv_id,),
        ).fetchall()

    if not rows:
        return []

    all_rows = [dict(r) for r in rows]

    if broad or not query_tokens:
        # Spread sampling: pick evenly distributed chunks across each document
        by_att: dict[str, list[dict[str, Any]]] = {}
        for r in all_rows:
            by_att.setdefault(str(r["attachment_id"]), []).append(r)

        selected: list[dict[str, Any]] = []
        used_chars = 0
        slots_per_att = max(1, limit // len(by_att))

        for att_chunks in by_att.values():
            n = len(att_chunks)
            step = max(1, n // slots_per_att)
            taken = 0
            i = 0
            while i < n and taken < slots_per_att and len(selected) < limit:
                chunk = att_chunks[i]
                content = str(chunk["content"])
                if used_chars + len(content) <= char_budget:
                    selected.append(chunk)
                    used_chars += len(content)
                    taken += 1
                i += step

        return selected

    # Hybrid search: BM25 (keyword) + embeddings (semântico) fundidos via RRF.
    # BM25 acerta termos exatos (siglas, nº de lei); o vetor acerta paráfrases.
    wanted_type = _wanted_doc_type(query_text) if query_text else ""
    boost = None
    if wanted_type:
        # Boost de metadado: pergunta menciona tipo ("qual lei...") → prioriza chunks do tipo
        boost = lambda r: 1.3 if str(r.get("doc_type") or "") == wanted_type else 1.0

    scored = _hybrid_rank(all_rows, query_tokens, query_text, boost=boost)
    return _select_diverse(scored, limit, char_budget)


def _hybrid_rank(
    all_rows: list[dict[str, Any]],
    query_tokens: list[str],
    query_text: str = "",
    boost: "Any | None" = None,
) -> list[tuple[float, dict[str, Any], "list[float] | None", list[str]]]:
    """Ranking híbrido BM25 + embeddings via RRF sobre linhas com colunas
    content/tokens/embedding. Retorna [(score, row, emb, toks)] ordenado."""
    query_emb: "list[float] | None" = (
        get_embedding(query_text, kind="query") if query_text else None
    )

    row_embs: list["list[float] | None"] = []
    row_toks: list[list[str]] = []
    for row in all_rows:
        row_emb: "list[float] | None" = None
        emb_raw = row.get("embedding")
        if query_emb and emb_raw:
            try:
                parsed = json.loads(emb_raw)
                if isinstance(parsed, list):
                    row_emb = parsed
            except (json.JSONDecodeError, TypeError):
                row_emb = None
        row_embs.append(row_emb)
        try:
            row_toks.append([str(t) for t in json.loads(row["tokens"] or "[]")])
        except (json.JSONDecodeError, TypeError):
            row_toks.append(tokenize(str(row["content"])))

    # Fusão por score normalizado (não por posição/RRF): preserva a MARGEM do
    # BM25 — um match 3x mais forte deve valer 3x, não "1º lugar". Necessário
    # porque o nomic-embed em pt-BR devolve cossenos achatados (~0.63-0.69 até
    # para temas sem relação) e, num ranking por posição, esse ruído plano
    # afogaria acertos exatos de keyword.
    kw = bm25_scores(query_tokens, row_toks)
    kw_max = max(kw) if kw else 0.0

    cos_list: list["float | None"] = [
        cosine_sim(query_emb, e) if (query_emb and e is not None) else None
        for e in row_embs
    ]
    valid_cos = [c for c in cos_list if c is not None and c > 0]
    cos_min = min(valid_cos) if valid_cos else 0.0
    cos_max = max(valid_cos) if valid_cos else 0.0
    cos_span = cos_max - cos_min

    W_KW, W_VEC = 0.65, 0.35

    fused: dict[int, float] = {}
    for i in range(len(all_rows)):
        s_kw = (kw[i] / kw_max) if kw_max > 0 else 0.0
        s_vec = 0.0
        c = cos_list[i]
        if c is not None and c > 0 and cos_span > 0:
            s_vec = (c - cos_min) / cos_span
        # Candidato: tem match de keyword, ou é o destaque semântico da vez
        # (permite recuperar paráfrases sem nenhum termo em comum)
        if s_kw > 0 or s_vec >= 0.95:
            fused[i] = W_KW * s_kw + W_VEC * s_vec

    if not fused:
        return []

    if boost:
        for i in list(fused):
            fused[i] *= float(boost(all_rows[i]))

    return [
        (fused[i], all_rows[i], row_embs[i], row_toks[i])
        for i in sorted(fused, key=lambda i: fused[i], reverse=True)
    ]


def _select_diverse(
    scored: list[tuple[float, dict[str, Any], "list[float] | None", list[str]]],
    limit: int,
    char_budget: int,
) -> list[dict[str, Any]]:
    """Seleção final: respeita orçamento de chars e pula chunks redundantes."""
    selected: list[dict[str, Any]] = []
    sel_embs: list["list[float] | None"] = []
    sel_toks: list[list[str]] = []
    used_chars = 0

    for _, row, emb, toks in scored:
        content = str(row["content"])

        if emb and any(e is not None for e in sel_embs):
            redundancy = max((cosine_sim(emb, e) for e in sel_embs if e is not None), default=0.0)
        elif toks and sel_toks:
            redundancy = max((similarity(toks, t) for t in sel_toks), default=0.0)
        else:
            redundancy = 0.0

        if selected and redundancy > 0.82:
            continue
        if selected and used_chars + len(content) > char_budget:
            continue

        selected.append(row)
        sel_embs.append(emb)
        sel_toks.append(toks)
        used_chars += len(content)
        if len(selected) >= limit:
            break

    return selected


# ---------------------------------------------------------------------------
# Base de conhecimento geral (knowledge/) — conteúdo de referência disponível
# em TODAS as conversas. Dá "cultura geral" a modelos pequenos via RAG.
# ---------------------------------------------------------------------------

_knowledge_lock = threading.Lock()
_knowledge_last_check = 0.0
KNOWLEDGE_SYNC_INTERVAL = 60.0  # segundos entre verificações de mtime


def _split_knowledge_sections(text: str, max_chars: int = 2200) -> list[str]:
    """Divide arquivos de referência por seção '## ' — cada seção é um chunk
    coeso (uma fórmula, uma regra). Seções longas caem no chunking padrão."""
    lines = text.strip().splitlines()
    if not lines:
        return []

    sections: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    result: list[str] = []
    for sec in sections:
        if not sec:
            continue
        if len(sec) <= max_chars:
            result.append(sec)
        else:
            result.extend(chunk_text(sec, max_chars=max_chars))
    return result


def sync_knowledge(force: bool = False) -> dict[str, int]:
    """Indexa/reindexa arquivos .md/.txt de knowledge/ cujo mtime mudou.
    Barato quando nada mudou (só stat nos arquivos); throttled por padrão."""
    global _knowledge_last_check
    now = time.time()
    if not force and now - _knowledge_last_check < KNOWLEDGE_SYNC_INTERVAL:
        return {}

    with _knowledge_lock:
        _knowledge_last_check = now
        if not KNOWLEDGE_DIR.exists():
            return {"files": 0, "chunks_indexed": 0}

        files = sorted(
            list(KNOWLEDGE_DIR.glob("*.md")) + list(KNOWLEDGE_DIR.glob("*.txt"))
        )
        indexed = 0
        with get_db() as conn:
            stored = {
                str(r["source"]): float(r["mtime"])
                for r in conn.execute(
                    "SELECT source, MAX(mtime) AS mtime FROM knowledge_chunks GROUP BY source"
                ).fetchall()
            }
            current_names = {f.name for f in files}
            for gone in set(stored) - current_names:
                conn.execute("DELETE FROM knowledge_chunks WHERE source = ?", (gone,))

            for fpath in files:
                mtime = fpath.stat().st_mtime
                if abs(stored.get(fpath.name, -1.0) - mtime) < 1e-6:
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                conn.execute("DELETE FROM knowledge_chunks WHERE source = ?", (fpath.name,))
                ts = now_iso()
                for i, section in enumerate(_split_knowledge_sections(text)):
                    toks = tokenize(section)
                    if not toks:
                        continue
                    emb = get_embedding(section)
                    conn.execute(
                        "INSERT INTO knowledge_chunks "
                        "(id, source, chunk_index, content, tokens, embedding, mtime, created_at) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (
                            new_id(), fpath.name, i, section,
                            json.dumps(toks[:450], ensure_ascii=False),
                            json.dumps(emb) if emb else None,
                            mtime, ts,
                        ),
                    )
                    indexed += 1

        return {"files": len(files), "chunks_indexed": indexed}


# Palavras de conversa cotidiana: não contam como "substância" para decidir
# se vale buscar referência (mas continuam na busca quando ela acontece)
_CHAT_NOISE = {
    "bom", "boa", "dia", "tarde", "noite", "ola", "olá", "oii", "hey",
    "tudo", "bem", "beleza", "valeu", "obrigado", "obrigada", "por", "favor",
    "ate", "até", "logo", "tchau", "abraço", "abraco", "gente", "cara",
}


def search_knowledge(
    query_tokens: list[str],
    limit: int,
    char_budget: int,
    query_text: str = "",
) -> list[dict[str, Any]]:
    """Busca híbrida na base de conhecimento geral (todas as conversas)."""
    # Saudações e conversa trivial não merecem contexto de referência.
    # 1 token substantivo basta ("crase", "PROCV") — o tokenize já descarta
    # interrogativas e auxiliares.
    substance = [t for t in query_tokens if t not in _CHAT_NOISE]
    if not substance:
        return []
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM knowledge_chunks").fetchall()
    if not rows:
        return []
    scored = _hybrid_rank([dict(r) for r in rows], query_tokens, query_text)
    return _select_diverse(scored, limit, char_budget)


def count_knowledge_chunks() -> int:
    with get_db() as conn:
        return conn.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0]


def count_doc_chunks(conv_id: str) -> int:
    with get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM doc_chunks WHERE conversation_id = ?", (conv_id,)
        ).fetchone()[0]


def _delete_workspace_file(conn: sqlite3.Connection, row: sqlite3.Row | dict[str, Any]) -> None:
    conn.execute("DELETE FROM workspace_files WHERE id = ?", (row["id"],))
    conn.execute("DELETE FROM messages WHERE id = ?", (row["message_id"],))


def _workspace_file_candidates(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in WORKSPACE_SKIP_DIRS and not d.startswith(".")
        ]
        current = Path(dirpath)
        for filename in filenames:
            path = current / filename
            if filename.startswith("."):
                continue
            try:
                if path.stat().st_size > WORKSPACE_MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(path)
    return files


def workspace_summary(conv_id: str) -> dict[str, Any]:
    with get_db() as conn:
        conv = conn.execute("SELECT workspace_path FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        indexed = conn.execute(
            "SELECT COUNT(*) FROM workspace_files WHERE conversation_id = ? AND status = 'indexed'",
            (conv_id,),
        ).fetchone()[0]
        skipped = conn.execute(
            "SELECT COUNT(*) FROM workspace_files WHERE conversation_id = ? AND status != 'indexed'",
            (conv_id,),
        ).fetchone()[0]
    return {
        "workspace_path": str(conv["workspace_path"] or "") if conv else "",
        "indexed_files": indexed,
        "skipped_files": skipped,
    }


def set_workspace_path(conv_id: str, folder_path: str) -> dict[str, Any]:
    raw = folder_path.strip()
    if not raw:
        with get_db() as conn:
            old_rows = conn.execute(
                "SELECT * FROM workspace_files WHERE conversation_id = ?", (conv_id,)
            ).fetchall()
            for row in old_rows:
                _delete_workspace_file(conn, row)
            conn.execute("UPDATE conversations SET workspace_path = ?, updated_at = ? WHERE id = ?",
                         ("", now_iso(), conv_id))
        return workspace_summary(conv_id)

    root = Path(raw).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError("Pasta não encontrada ou inválida.")

    with get_db() as conn:
        current = conn.execute(
            "SELECT workspace_path FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if current and str(current["workspace_path"] or "") != str(root):
            old_rows = conn.execute(
                "SELECT * FROM workspace_files WHERE conversation_id = ?", (conv_id,)
            ).fetchall()
            for row in old_rows:
                _delete_workspace_file(conn, row)
        conn.execute("UPDATE conversations SET workspace_path = ?, updated_at = ? WHERE id = ?",
                     (str(root), now_iso(), conv_id))
    return workspace_summary(conv_id)


def _index_workspace_file(conv_id: str, root: Path, path: Path) -> tuple[str, str]:
    rel_path = path.relative_to(root).as_posix()
    stat = path.stat()
    text = extract_text_preview(path, max_chars=MAX_DOC_CHARS)
    ts = now_iso()

    with get_db() as conn:
        existing = conn.execute(
            "SELECT * FROM workspace_files WHERE conversation_id = ? AND rel_path = ?",
            (conv_id, rel_path),
        ).fetchone()
        if existing:
            _delete_workspace_file(conn, existing)

        mid = new_id()
        aid = new_id()
        wf_id = new_id()
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?,?,?,?,?)",
            (mid, conv_id, "workspace", f"[workspace] {rel_path}", ts),
        )
        conn.execute(
            "INSERT INTO attachments "
            "(id, message_id, original_name, stored_name, content_type, size, text_preview, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                aid, mid, rel_path, f"workspace:{aid}",
                mimetypes.guess_type(path.name)[0] or "", stat.st_size,
                text[:MAX_TEXT_PREVIEW], ts,
            ),
        )
        status = "indexed" if text else "skipped"
        error = None if text else "sem texto legível"
        conn.execute(
            "INSERT INTO workspace_files "
            "(id, conversation_id, message_id, attachment_id, abs_path, rel_path, size, mtime, status, error, indexed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (wf_id, conv_id, mid, aid, str(path), rel_path, stat.st_size, stat.st_mtime, status, error, ts),
        )

    if text:
        store_doc_chunks(aid, conv_id, chunk_text(text), rel_path, with_embeddings=False)
    return rel_path, "indexed" if text else "skipped"


def sync_workspace(conv_id: str) -> dict[str, Any]:
    with get_db() as conn:
        conv = conn.execute("SELECT workspace_path FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    root_raw = str(conv["workspace_path"] or "") if conv else ""
    if not root_raw:
        return {"workspace_path": "", "indexed_files": 0, "skipped_files": 0,
                "added": 0, "updated": 0, "removed": 0, "skipped": 0}

    root = Path(root_raw).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError("A pasta vinculada não existe mais.")

    candidates = _workspace_file_candidates(root)
    current_rel_paths = {p.relative_to(root).as_posix(): p for p in candidates}

    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM workspace_files WHERE conversation_id = ?", (conv_id,)
        ).fetchall()
        known = {str(r["rel_path"]): r for r in rows}

        removed = 0
        for rel_path, row in known.items():
            if rel_path not in current_rel_paths:
                _delete_workspace_file(conn, row)
                removed += 1

    added = 0
    updated = 0
    skipped = 0
    for rel_path, path in current_rel_paths.items():
        try:
            stat = path.stat()
        except OSError:
            skipped += 1
            continue
        row = known.get(rel_path)
        unchanged = (
            row
            and int(row["size"]) == int(stat.st_size)
            and abs(float(row["mtime"]) - float(stat.st_mtime)) < 0.0001
        )
        if unchanged:
            continue
        _, status = _index_workspace_file(conv_id, root, path)
        if status != "indexed":
            skipped += 1
        if row:
            updated += 1
        else:
            added += 1

    summary = workspace_summary(conv_id)
    return {
        **summary,
        "added": added,
        "updated": updated,
        "removed": removed,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Global insights — cross-conversation curated knowledge
# ---------------------------------------------------------------------------

def upsert_global_insight(content: str) -> None:
    """Add or update a global insight, merging with an existing similar one."""
    tokens = tokenize(content)
    if not tokens or len(content) < 25:
        return
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, tokens FROM global_insights ORDER BY updated_at DESC LIMIT 80"
        ).fetchall()

    ts = now_iso()
    for row in rows:
        try:
            etoks = [str(t) for t in json.loads(row["tokens"] or "[]")]
        except (json.JSONDecodeError, TypeError):
            etoks = []
        if similarity(tokens, etoks) > 0.58:
            with get_db() as conn:
                conn.execute(
                    "UPDATE global_insights "
                    "SET content = ?, tokens = ?, access_count = access_count + 1, updated_at = ? "
                    "WHERE id = ?",
                    (content, json.dumps(tokens[:200], ensure_ascii=False), ts, row["id"]),
                )
            return

    with get_db() as conn:
        conn.execute(
            "INSERT INTO global_insights (id, content, tokens, access_count, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (new_id(), content, json.dumps(tokens[:200], ensure_ascii=False), 0, ts, ts),
        )


def get_global_insights(query_tokens: list[str], limit: int = 8) -> list[dict[str, Any]]:
    """Retrieve global insights ranked by relevance to query_tokens."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM global_insights ORDER BY access_count DESC, updated_at DESC LIMIT 120"
        ).fetchall()
    if not rows:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        try:
            toks = [str(t) for t in json.loads(row["tokens"] or "[]")]
        except (json.JSONDecodeError, TypeError):
            toks = []
        s = similarity(query_tokens, toks) if query_tokens else 0.0
        s = max(s, 0.08)  # floor so recent insights always surface somewhat
        scored.append((s, dict(row)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


def get_all_global_insights() -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM global_insights ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def store_attachments(
    message_id: str, files: list[FileStorage], conv_id: str = ""
) -> list[dict[str, Any]]:
    stored = []
    for f in files:
        if not f or not f.filename:
            continue
        safe = secure_filename(f.filename) or "arquivo"
        aid = new_id()
        stored_name = f"{aid}_{safe}"
        path = UPLOAD_DIR / stored_name
        f.save(path)
        size = path.stat().st_size

        # Extract text: short preview stored in the DB row, full text for chunk index
        preview = extract_text_preview(path, max_chars=MAX_TEXT_PREVIEW)
        full_text = extract_text_preview(path, max_chars=MAX_DOC_CHARS) if conv_id else ""

        ts = now_iso()
        with get_db() as conn:
            conn.execute(
                "INSERT INTO attachments "
                "(id, message_id, original_name, stored_name, content_type, size, text_preview, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (aid, message_id, f.filename, stored_name, f.content_type, size, preview, ts),
            )

        if conv_id and full_text:
            chunks = chunk_text(full_text)
            # Index in background so the upload response is not blocked by large docs
            threading.Thread(
                target=store_doc_chunks,
                args=(aid, conv_id, chunks, f.filename),
                daemon=True,
            ).start()

        stored.append({
            "id": aid,
            "name": f.filename,
            "content_type": f.content_type,
            "size": size,
            "text_preview": preview,
        })
    return stored


def count_user_messages(conv_id: str) -> int:
    with get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND role = 'user'", (conv_id,)
        ).fetchone()[0]


def get_memory_stats() -> dict[str, Any]:
    with get_db() as conn:
        mem = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        ins = conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
        glo = conn.execute("SELECT COUNT(*) FROM global_insights").fetchone()[0]
    return {"memory_count": mem, "insight_count": ins, "global_count": glo}
