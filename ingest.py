"""
Ingestão de documentos de trabalho na base de conhecimento (knowledge/).

Converte PDF, DOCX, TXT ou MD em Markdown estruturado por seções `## ` —
o formato exato que o sync_knowledge indexa. Cada tipo de documento tem um
divisor especializado, porque chunk coeso é o que faz o RAG funcionar:

  lei / decreto / portaria / instrução normativa → divide por ARTIGO
  parecer / modelo de resposta                   → EMENTA > RELATÓRIO >
                                                   FUNDAMENTAÇÃO > CONCLUSÃO
  manual / apostila                              → capítulos, seções e
                                                   títulos numerados
  qualquer outro                                 → blocos de ~1.700 chars

Todo cabeçalho de seção repete o nome do documento: o chunk entra sozinho
no prompt, então precisa se identificar ("Lei 14.133 — Art. 75" e não só
"Art. 75").

Os arquivos gerados usam o prefixo doc_ e ficam FORA do GitHub por padrão
(knowledge/doc_*.md no .gitignore) — documento interno da PMC não deve
subir pro repositório.

Uso:
  python ingest.py caminho/lei_14133.pdf
  python ingest.py pasta_com_documentos/
  python ingest.py arquivo.pdf --tipo parecer --nome "Modelo Parecer GPDP"
  python ingest.py arquivo.pdf --dry-run     # mostra as seções sem gravar
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import db as database
from utils import chunk_text

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
SUPPORTED = {".pdf", ".docx", ".txt", ".md"}
MAX_CHARS = 300_000
TARGET_CHARS = 1_700   # alvo por seção — abaixo dos 2.200 do sync_knowledge,
                       # pra nenhuma seção cair no chunking cego sem cabeçalho


# ---------------------------------------------------------------------------
# Extração — preserva quebras de linha (o extract_text_preview do utils
# achata \s+ e destruiria a detecção de capítulos/títulos)
# ---------------------------------------------------------------------------

def _read_pdf(path: Path) -> str:
    text = ""
    pages = 0
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = len(reader.pages)
        parts = []
        total = 0
        for page in reader.pages:
            t = page.extract_text() or ""
            parts.append(t)
            total += len(t)
            if total >= MAX_CHARS:
                break
        text = "\n".join(parts)
    except Exception:
        pass

    # PDF escaneado (quase sem texto digital) → OCR do utils (texto achatado,
    # mas os divisores por regex inline continuam funcionando)
    if len(text) / max(pages, 1) < 80:
        from utils import _ocr_pdf_pages
        ocr = _ocr_pdf_pages(path, MAX_CHARS)
        if len(ocr) > len(text):
            return ocr
    return text[:MAX_CHARS]


def _read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)[:MAX_CHARS]


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = _read_pdf(path)
    elif suffix == ".docx":
        raw = _read_docx(path)
    else:
        data = path.read_bytes()
        raw = ""
        for enc in ("utf-8", "latin-1"):
            try:
                raw = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        raw = raw[:MAX_CHARS]
    # Normaliza sem achatar: colapsa espaços repetidos, preserva \n
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# Identificação do documento
# ---------------------------------------------------------------------------

def _sem_acento(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


_KIND_RE = re.compile(
    r"\b(LEI(?:\s+COMPLEMENTAR)?|DECRETO(?:-LEI)?|PORTARIA|INSTRUCAO\s+NORMATIVA|RESOLUCAO)"
    r"\s*(?:MUNICIPAL|ESTADUAL|FEDERAL)?\s*(?:N[º°.]?|No\.?)?\s*([\d][\d.]*)",
    re.IGNORECASE,
)
_ANO_RE = re.compile(r"[/\-]\s*(\d{4})\b|\bDE\s+\d{1,2}[º°]?\s+DE\s+\w+\s+DE\s+(\d{4})", re.IGNORECASE)


def detect_doc_name(text: str, path: Path) -> str:
    head = _sem_acento(text[:1500])
    m = _KIND_RE.search(head)
    if m:
        kind = " ".join(w.capitalize() for w in m.group(1).split())
        numero = m.group(2).rstrip(".")
        ano_m = _ANO_RE.search(head[m.end():m.end() + 120])
        ano = (ano_m.group(1) or ano_m.group(2)) if ano_m else ""
        return f"{kind} {numero}" + (f"/{ano}" if ano else "")
    # fallback: nome do arquivo prettificado
    stem = re.sub(r"[_\-]+", " ", path.stem).strip()
    return stem.title() if stem else "Documento"


def _article_positions(text: str) -> list[int]:
    """Posições onde começa um artigo de verdade. 'A' maiúsculo + caractere
    anterior não-letra filtram referências no meio da frase ('do art. 37')."""
    positions = []
    for m in re.finditer(r"\bArt(?:igo)?\s*\.?\s*\d+", text):
        i = m.start()
        prev = text[:i].rstrip()[-1:]
        if i == 0 or text[i - 1] == "\n" or prev in ".;:—–)]":
            positions.append(i)
    return positions


def detect_type(text: str) -> str:
    head = _sem_acento(text[:2000]).upper()
    if "INSTRUCAO NORMATIVA" in head:
        return "instrucao"
    if re.search(r"\bLEI\s+(COMPLEMENTAR\s+)?N", head):
        return "lei"
    if re.search(r"\bDECRETO\b", head):
        return "decreto"
    if re.search(r"\bPORTARIA\b", head):
        return "portaria"
    if re.search(r"\bPARECER\b", head) or "EMENTA" in head[:600]:
        return "parecer"
    if len(_article_positions(text)) >= 3:
        return "ato-normativo"
    if re.search(r"\b(MANUAL|GUIA|APOSTILA|PROCEDIMENTO|TUTORIAL)\b", head):
        return "manual"
    return "documento"


# tipo → estratégia de divisão
_NORMATIVOS = {"lei", "decreto", "portaria", "instrucao", "ato-normativo"}


# ---------------------------------------------------------------------------
# Divisores (todos retornam [(titulo_da_secao, corpo), ...])
# ---------------------------------------------------------------------------

def _emit_oversized(title: str, body: str, out: list[tuple[str, str]]) -> None:
    """Seção maior que o alvo: divide preservando o título em cada parte."""
    pieces = chunk_text(body, max_chars=TARGET_CHARS)
    if len(pieces) <= 1:
        out.append((title, body))
        return
    for n, piece in enumerate(pieces, 1):
        out.append((f"{title} (parte {n}/{len(pieces)})", piece))


def split_normativo(text: str) -> list[tuple[str, str]]:
    positions = _article_positions(text)
    if len(positions) < 2:
        return []

    sections: list[tuple[str, str]] = []
    preamble = text[:positions[0]].strip()
    if len(preamble) >= 60:
        _emit_oversized("Ementa e disposições preliminares", preamble, sections)

    bounds = positions + [len(text)]
    grupo: list[str] = []
    artigos: list[str] = []
    tamanho = 0

    def flush() -> None:
        if not grupo:
            return
        titulo = f"Art. {artigos[0]}" if len(artigos) == 1 else f"Art. {artigos[0]} a {artigos[-1]}"
        _emit_oversized(titulo, "\n".join(grupo), sections)
        grupo.clear()
        artigos.clear()

    for i in range(len(positions)):
        parte = text[bounds[i]:bounds[i + 1]].strip()
        if not parte:
            continue
        num_m = re.match(r"Art(?:igo)?\s*\.?\s*(\d+)", parte)
        numero = num_m.group(1) if num_m else "?"
        if grupo and tamanho + len(parte) > TARGET_CHARS:
            flush()
            tamanho = 0
        grupo.append(parte)
        artigos.append(numero)
        tamanho += len(parte)
    flush()
    return sections


_PARECER_SECOES = re.compile(
    r"\b(EMENTA|RELAT[ÓO]RIO|FUNDAMENTA[ÇC][ÃA]O|AN[ÁA]LISE|DO M[ÉE]RITO|"
    r"M[ÉE]RITO|CONCLUS[ÃA]O|DISPOSITIVO|VOTO)\b"
)


def split_parecer(text: str) -> list[tuple[str, str]]:
    # Cabeçalhos de parecer vêm em CAIXA ALTA — regex case-sensitive de propósito
    matches = list(_PARECER_SECOES.finditer(text))
    if len(matches) < 2:
        return []

    sections: list[tuple[str, str]] = []
    cabecalho = text[:matches[0].start()].strip()
    if len(cabecalho) >= 60:
        _emit_oversized("Cabeçalho e identificação", cabecalho, sections)

    for i, m in enumerate(matches):
        fim = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        corpo = text[m.start():fim].strip()
        if not corpo:
            continue
        titulo = m.group(1).capitalize()
        _emit_oversized(titulo, corpo, sections)
    return sections


_HEADING_ESTRUTURAL = re.compile(
    r"^(T[ÍI]TULO|CAP[ÍI]TULO|SE[ÇC][ÃA]O|SUBSE[ÇC][ÃA]O|ANEXO|AP[ÊE]NDICE)\b", re.IGNORECASE
)
_HEADING_NUMERADO = re.compile(r"^\d+(\.\d+)*[\s.)\-–]+\S")


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not (5 < len(line) < 90) or line.endswith((".", ";", ",", ":")):
        return False
    if _HEADING_ESTRUTURAL.match(line):
        return True
    if _HEADING_NUMERADO.match(line):
        return True
    # Linha inteira em caixa alta com poucas cifras = título de seção
    letras = [c for c in line if c.isalpha()]
    return len(letras) >= 5 and all(c.isupper() for c in letras)


def split_manual(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    headings = [i for i, l in enumerate(lines) if _is_heading(l)]
    if len(headings) < 2:
        return []

    sections: list[tuple[str, str]] = []
    intro = "\n".join(lines[:headings[0]]).strip()
    if len(intro) >= 60:
        _emit_oversized("Introdução", intro, sections)

    bounds = headings + [len(lines)]
    for i in range(len(headings)):
        titulo = lines[headings[i]].strip()[:70]
        corpo = "\n".join(lines[headings[i]:bounds[i + 1]]).strip()
        if len(corpo) <= len(titulo) + 10:
            continue  # título sem conteúdo (sumário, por exemplo)
        _emit_oversized(titulo, corpo, sections)
    return sections


def split_generic(text: str) -> list[tuple[str, str]]:
    pieces = chunk_text(text, max_chars=TARGET_CHARS)
    return [(f"parte {n}", p) for n, p in enumerate(pieces, 1)]


def split_document(text: str, tipo: str) -> tuple[list[tuple[str, str]], str]:
    """Aplica o divisor do tipo com fallbacks em cascata.
    Retorna (seções, estratégia_usada)."""
    if tipo in _NORMATIVOS:
        secs = split_normativo(text)
        if secs:
            return secs, "artigos"
    if tipo in ("parecer", "modelo"):
        secs = split_parecer(text)
        if secs:
            return secs, "estrutura de parecer"
    secs = split_manual(text)
    if secs:
        return secs, "títulos/capítulos"
    return split_generic(text), "blocos"


# ---------------------------------------------------------------------------
# Saída
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    s = _sem_acento(name).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "documento"


def build_markdown(name: str, tipo: str, origem: str, sections: list[tuple[str, str]]) -> str:
    parts = [f"# {name}", f"_(ingerido de {origem}; tipo: {tipo})_", ""]
    for titulo, corpo in sections:
        parts.append(f"## {name} — {titulo}")
        parts.append(corpo)
        parts.append("")
    return "\n".join(parts)


def ingest_file(path: Path, tipo_arg: str, nome_arg: str, dry_run: bool, force: bool) -> bool:
    text = read_document(path)
    if len(text) < 100:
        print(f"[PULADO] {path.name}: sem texto legível (100 chars mínimos).")
        return False

    tipo = tipo_arg if tipo_arg != "auto" else detect_type(text)
    name = nome_arg or detect_doc_name(text, path)
    sections, estrategia = split_document(text, tipo)

    out_path = KNOWLEDGE_DIR / f"doc_{slugify(name)}.md"
    tamanhos = [len(c) for _, c in sections]
    print(f"\n{path.name}")
    print(f"  nome: {name} | tipo: {tipo} | divisão: {estrategia}")
    print(f"  seções: {len(sections)} (média {sum(tamanhos)//max(len(tamanhos),1)} chars, máx {max(tamanhos, default=0)})")
    print(f"  destino: {out_path.relative_to(Path(__file__).parent)}")

    if dry_run:
        for titulo, _ in sections[:8]:
            print(f"    ## {name} — {titulo}")
        if len(sections) > 8:
            print(f"    ... +{len(sections) - 8} seções")
        return False

    if out_path.exists() and not force:
        print(f"  [PULADO] destino já existe — use --force para sobrescrever.")
        return False

    out_path.write_text(build_markdown(name, tipo, path.name, sections), encoding="utf-8")
    print("  gravado.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestão de documentos → knowledge/")
    parser.add_argument("caminhos", nargs="+", help="Arquivos ou pastas (pdf, docx, txt, md)")
    parser.add_argument("--tipo", default="auto",
                        choices=["auto", "lei", "decreto", "portaria", "instrucao",
                                 "parecer", "modelo", "manual", "documento"])
    parser.add_argument("--nome", default="", help="Nome do documento (senão detecta do texto)")
    parser.add_argument("--dry-run", action="store_true", help="Mostra as seções sem gravar")
    parser.add_argument("--force", action="store_true", help="Sobrescreve destino existente")
    parser.add_argument("--sem-index", action="store_true", help="Não reindexa ao final")
    args = parser.parse_args()

    files: list[Path] = []
    for c in args.caminhos:
        p = Path(c).expanduser()
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED))
        elif p.exists() and p.suffix.lower() in SUPPORTED:
            files.append(p)
        else:
            print(f"[AVISO] ignorando {c} (não existe ou extensão não suportada)")

    if not files:
        print("Nenhum arquivo para ingerir.")
        sys.exit(1)
    if args.nome and len(files) > 1:
        print("[ERRO] --nome só pode ser usado com um arquivo por vez.")
        sys.exit(1)

    gravados = 0
    for f in files:
        if ingest_file(f, args.tipo, args.nome, args.dry_run, args.force):
            gravados += 1

    if gravados and not args.sem_index:
        result = database.sync_knowledge(force=True)
        print(f"\nReindexado: {result.get('chunks_indexed', 0)} chunks novos "
              f"({result.get('files', 0)} arquivos na base). Já disponível nas buscas.")
    elif args.dry_run:
        print("\n(dry-run: nada gravado nem indexado)")


if __name__ == "__main__":
    main()
