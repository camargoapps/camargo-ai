"""
Expansão de consulta — resgate de recall da busca na base de conhecimento.

Pergunta coloquial ("como eu faço pra por a lei no sistema?") raramente
compartilha termos com o texto formal indexado ("legislação", "norma",
"indexação"). Quando a busca direta volta VAZIA, um modelo pequeno dedicado
gera sinônimos e termos técnicos e a busca roda de novo — só nesse caso a
latência extra (~segundos em CPU) é paga, e é quando ela vale a pena.

Seguro por construção: o BM25 soma matches — termo expandido que não bate
em nada não dilui o score de quem bateu. Falha silenciosa: sem modelo,
sem Ollama ou saída ruim → lista vazia e a busca segue só com a pergunta.
"""

import os
import re
import threading
import time

import ai_engine
from utils import tokenize

# Ajudante de bastidor: HELPER_MODEL troca todos de uma vez (num PC mais
# forte, use um 4b+); EXPAND_MODEL sobrepõe só este papel
HELPER_MODEL = os.environ.get("HELPER_MODEL", "qwen3.5:0.8b")
EXPAND_MODEL = os.environ.get("EXPAND_MODEL", HELPER_MODEL)
EXPAND_TIMEOUT = int(os.environ.get("EXPAND_TIMEOUT", "20"))
MAX_EXTRA_TOKENS = 10
_RECHECK_INTERVAL = 120.0  # segundos até tentar de novo quando o modelo faltou
_CACHE_MAX = 128  # expansões recentes (pergunta repetida = resposta instantânea)

_PROMPT = (
    "Liste de 3 a 6 termos de busca relacionados à pergunta abaixo: "
    "sinônimos, termos técnicos ou formais do mesmo assunto. "
    "Um termo por linha, sem numeração, sem explicação.\n\n"
    "Pergunta: {question}\n\nTermos:"
)

_lock = threading.Lock()
_available: "bool | None" = None
_checked_at = 0.0
_cache: dict[str, list[str]] = {}


def warm() -> None:
    """Pré-carrega o modelo de expansão no Ollama (chamar em thread no
    startup) para a primeira mensagem não pagar os ~10s de load."""
    if _model_available():
        ai_engine.generate_text(
            EXPAND_MODEL, "ok", timeout=60,
            options={"num_predict": 1}, keep_alive="30m",
        )


def _model_available() -> bool:
    """Confere uma vez se o modelo de expansão existe no Ollama.
    Resultado negativo é reavaliado com throttle (Ollama pode subir depois)."""
    global _available, _checked_at
    now = time.time()
    if _available is None or (not _available and now - _checked_at > _RECHECK_INTERVAL):
        with _lock:
            if _available is None or (not _available and now - _checked_at > _RECHECK_INTERVAL):
                _checked_at = now
                _available = EXPAND_MODEL in ai_engine.list_local_models()
    return bool(_available)


def expand_query(prompt: str) -> list[str]:
    """Retorna tokens extras de busca derivados da pergunta, ou [] quando
    não vale a latência (pergunta trivial) ou o modelo não está disponível."""
    base_tokens = tokenize(prompt)
    if len(base_tokens) < 2:
        return []
    if not _model_available():
        return []

    cache_key = " ".join(sorted(base_tokens))
    cached = _cache.get(cache_key)
    if cached is not None:
        return list(cached)

    raw = ai_engine.generate_text(
        EXPAND_MODEL,
        _PROMPT.format(question=prompt.strip()[:500]),
        timeout=EXPAND_TIMEOUT,
        # num_predict baixo: termos são curtos e cortar cedo reduz a latência;
        # keep_alive longo: recarregar o modelo custa ~10s, gerar custa ~2s
        options={"num_predict": 48, "temperature": 0},
        keep_alive="30m",
    )
    if not raw:
        return []

    seen = set(base_tokens)
    extra: list[str] = []
    for line in raw.splitlines():
        term = re.sub(r"^[\s\-\*\d.)]+", "", line).strip()
        # Linha longa demais é explicação do modelo, não um termo
        if not term or len(term) > 80:
            continue
        for tok in tokenize(term):
            if tok not in seen:
                seen.add(tok)
                extra.append(tok)
        if len(extra) >= MAX_EXTRA_TOKENS:
            break

    result = extra[:MAX_EXTRA_TOKENS]
    with _lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[cache_key] = result
    return list(result)
