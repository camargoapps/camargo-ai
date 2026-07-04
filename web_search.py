"""
Fase 7a: consulta à web — busca geral (DuckDuckGo) + legislação (LexML).

Ativada pelo toggle 🌐 no chat. O modelo local nunca acessa a rede:
código Python busca, baixa e extrai; o modelo só lê o texto que chega
como referência citável no prompt.

LexML: o serviço SRU oficial está fora do ar (404), então usamos a busca
HTML pública, filtrando URNs de legislação (lei, decreto, portaria...)
e descartando acervo bibliográfico (rede.virtual.bibliotecas).
"""

import html
import re
import urllib.parse

import requests

import reasoning
from utils import tokenize

# UA de navegador real: identificador de bot no UA dispara anti-robô (202)
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"}
_TIMEOUT = 12
TOTAL_CHAR_BUDGET = 4_200
PAGE_CHAR_BUDGET = 1_600

# Tipos de URN LexML que são atos normativos (não livros/artigos)
_URN_NORMATIVO = (
    ":lei:", ":lei.complementar:", ":decreto:", ":decreto.lei:", ":portaria:",
    ":instrucao.normativa:", ":medida.provisoria:", ":resolucao:",
    ":emenda.constitucional:", ":deliberacao:",
)


def _get(url: str) -> "requests.Response | None":
    try:
        resp = requests.get(url, headers=_UA, timeout=_TIMEOUT)
        return resp if resp.ok else None
    except requests.RequestException:
        return None


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|nav|header|footer|aside)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


def search_lexml(query: str, max_results: int = 4) -> list[dict]:
    url = "https://www.lexml.gov.br/busca/search?keyword=" + urllib.parse.quote(query)
    resp = _get(url)
    if not resp:
        return []
    hits: list[dict] = []
    seen: set[str] = set()
    for m in re.finditer(r'href="(/urn/urn:lex:[^"]+)"[^>]*>([^<]{8,150})', resp.text):
        urn, title = m.group(1), html.unescape(m.group(2)).strip()
        if "rede.virtual.bibliotecas" in urn or urn in seen:
            continue
        if not any(t in urn for t in _URN_NORMATIVO):
            continue
        seen.add(urn)
        hits.append({"title": title, "url": "https://www.lexml.gov.br" + urn})
        if len(hits) >= max_results:
            break
    return hits


def search_web(query: str, max_results: int = 4) -> list[dict]:
    results: list[dict] = []
    text = ""
    # html.duckduckgo.com primeiro; sob throttle (202), tenta o endpoint lite
    for base in ("https://html.duckduckgo.com/html/?q=", "https://lite.duckduckgo.com/lite/?q="):
        resp = _get(base + urllib.parse.quote(query))
        if resp and "uddg=" in resp.text:
            text = resp.text
            break
    if not text:
        return []
    for m in re.finditer(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>|<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', text
    ):
        href = m.group(1) or m.group(3) or ""
        title = _strip_html(m.group(2) or m.group(4) or "")
        # DDG embrulha o destino em //duckduckgo.com/l/?uddg=<url-codificada>
        uddg = re.search(r"[?&]uddg=([^&]+)", href)
        target = urllib.parse.unquote(uddg.group(1)) if uddg else href
        if target.startswith("//"):
            target = "https:" + target
        if not target.startswith("http") or "duckduckgo.com" in target:
            continue
        results.append({"title": title, "url": target})
        if len(results) >= max_results:
            break
    return results


def fetch_page_text(url: str, max_chars: int = PAGE_CHAR_BUDGET) -> str:
    resp = _get(url)
    if not resp or "pdf" in resp.headers.get("Content-Type", "").lower():
        return ""
    return _strip_html(resp.text)[:max_chars]


def build_web_context(prompt: str) -> str:
    """Bloco de referências da web para o prompt, ou '' se nada útil."""
    parts: list[str] = []
    used = 0

    # Pergunta inteira mata buscador — destila em palavras-chave
    # ("o que diz a lei sobre teletrabalho?" → "diz lei teletrabalho")
    query = " ".join(tokenize(prompt)[:8]) or prompt.strip()[:80]

    # Matéria oficial → LexML primeiro (títulos e links de atos normativos)
    if reasoning.is_official_matter(prompt):
        leis = search_lexml(query)
        if not leis:
            # LexML é E-lógico: verbo comum ("diz") zera a busca — retenta
            # só com os termos mais distintivos (os mais longos)
            distintivos = sorted(tokenize(prompt), key=len, reverse=True)[:3]
            if distintivos:
                leis = search_lexml(" ".join(distintivos))
        if leis:
            bloco = "(lexml.gov.br — legislação encontrada)\n" + "\n".join(
                f"- {l['title']} — {l['url']}" for l in leis
            )
            parts.append(bloco)
            used += len(bloco)

    # Busca geral: baixa e extrai as 2 primeiras páginas úteis
    fetched = 0
    for hit in search_web(query):
        if fetched >= 2 or used >= TOTAL_CHAR_BUDGET:
            break
        text = fetch_page_text(hit["url"])
        if len(text) < 200:
            continue
        domain = urllib.parse.urlparse(hit["url"]).netloc
        bloco = f"(web: {domain})\n{hit['title']}\n{text}"[: TOTAL_CHAR_BUDGET - used]
        parts.append(bloco)
        used += len(bloco)
        fetched += 1

    return "\n\n".join(parts)
