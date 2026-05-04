import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from sqlite3 import Row
from typing import Any

MAX_TEXT_PREVIEW = 60_000
MAX_DOC_CHARS = 300_000   # used when indexing full documents for chunk search

CHUNK_CHARS = 1600
CHUNK_OVERLAP = 220


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return str(uuid.uuid4())


def row_to_dict(row: Row) -> dict[str, Any]:
    return dict(row)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9_]{3,}", text.lower())
    stopwords = {
        "para", "com", "uma", "que", "por", "das", "dos", "ele", "ela", "isso",
        "esse", "essa", "como", "mais", "mas", "foi", "tem", "vou", "sua", "seu",
        "the", "and", "you", "for", "with", "this", "that",
    }
    return [w for w in words if w not in stopwords]


def similarity(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


EMBED_MODEL = "nomic-embed-text"
_EMBED_URL = "http://127.0.0.1:11434/api/embed"


def get_embedding(text: str) -> "list[float] | None":
    import json as _json
    import urllib.request as _req
    payload = _json.dumps({"model": EMBED_MODEL, "input": text[:2000]}).encode()
    try:
        req = _req.Request(_EMBED_URL, data=payload, headers={"Content-Type": "application/json"})
        with _req.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())
        emb = data.get("embeddings", [None])[0]
        return emb if isinstance(emb, list) else None
    except Exception:
        return None


def cosine_sim(a: "list[float]", b: "list[float]") -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(y * y for y in b) ** 0.5
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def chunk_text(
    text: str,
    max_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            split_at = max(
                cleaned.rfind(". ", start, end),
                cleaned.rfind("? ", start, end),
                cleaned.rfind("! ", start, end),
                cleaned.rfind("\n", start, end),
            )
            if split_at > start + max_chars // 2:
                end = split_at + 1

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def summarize_title(prompt: str) -> str:
    cleaned = re.sub(r"\s+", " ", prompt).strip()
    return (cleaned[:44] + "...") if len(cleaned) > 47 else cleaned or "Conversa com arquivos"


def build_prompt(user_prompt: str, attachments: list[dict[str, Any]]) -> str:
    parts = [user_prompt.strip()]
    text_files = [
        f"Arquivo: {f['name']}\nConteúdo: {f['text_preview']}"
        for f in attachments if f.get("text_preview")
    ]
    if text_files:
        parts.append("Contexto dos arquivos anexados:\n" + "\n\n".join(text_files))
    elif attachments:
        parts.append("Arquivos sem texto legível: " + ", ".join(f["name"] for f in attachments))
    return "\n\n".join(p for p in parts if p)


def first_syllable(word: str) -> str:
    m = re.match(r"^([^aeiouáéíóúâêôãõàü]*[aeiouáéíóúâêôãõàü]+)", word, re.IGNORECASE)
    return m.group(1) if m else word[:2]


def _anon_name(value: str) -> str:
    pieces = value.split()
    if len(pieces) <= 1:
        return value
    return " ".join([pieces[0]] + [first_syllable(p) + "." for p in pieces[1:]])


def _anon_phone(m: re.Match) -> str:
    d = re.sub(r"\D", "", m.group(0))
    return f"{d[:4]}****"


def _anon_address(m: re.Match) -> str:
    kind, body = m.group(1), m.group(2)
    words = re.findall(r"[A-ZÁÉÍÓÚÂÊÔÃÕÀÜ][a-záéíóúâêôãõàü]+|\d+|\S+", body)
    out = []
    for w in words:
        if re.match(r"^\d+$", w):
            out.append("***")
        elif re.match(r"^[A-ZÁÉÍÓÚÂÊÔÃÕÀÜ]", w):
            out.append(first_syllable(w) + ".")
        else:
            out.append(w)
    return f"{kind} {' '.join(out)}"


def anonymize_for_cloud(text: str) -> str:
    text = re.sub(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", "[CPF_REMOVIDO]", text)
    text = re.sub(r"\b(?:RG|R\.G\.)\s*[:.-]?\s*[0-9A-Za-z.\-]{5,}\b", "RG [RG_REMOVIDO]", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?\d{4,5}[-\s]?\d{4}", _anon_phone, text)
    text = re.sub(
        r"\b(Rua|R\.|Avenida|Av\.|Travessa|Alameda|Estrada|Rodovia|Praça|Praca)\s+([^,\n]+)",
        _anon_address, text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b([A-ZÁÉÍÓÚÂÊÔÃÕÀÜ][a-záéíóúâêôãõàü]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÀÜ][a-záéíóúâêôãõàü]+)+)\b",
        lambda m: _anon_name(m.group(1)), text,
    )
    return text


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

_OCR_LANGS = "por+eng"  # Portuguese primary, English fallback


def _ocr_pil_image(img: "Any") -> str:
    """Run Tesseract on a PIL Image. Gracefully falls back if tesseract absent."""
    try:
        import pytesseract
        try:
            return pytesseract.image_to_string(img, lang=_OCR_LANGS, config="--psm 6")
        except pytesseract.TesseractError:
            return pytesseract.image_to_string(img, config="--psm 6")
    except Exception:
        return ""


def _ocr_pdf_pages(path: Path, max_chars: int) -> str:
    """Render each PDF page at 200 DPI and OCR. Used for scanned documents."""
    try:
        import fitz  # pymupdf
        from PIL import Image as _PILImage
        import io as _io

        doc = fitz.open(str(path))
        parts: list[str] = []
        total = 0
        mat = fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI for good OCR quality

        for page in doc:
            if total >= max_chars:
                break
            pix = page.get_pixmap(matrix=mat)
            img = _PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = _ocr_pil_image(img).strip()
            if text:
                parts.append(text)
                total += len(text)

        doc.close()
        return re.sub(r"\s+", " ", " ".join(parts)).strip()[:max_chars]
    except Exception:
        return ""


def _extract_image_ocr(path: Path, max_chars: int) -> str:
    """OCR a raster image file (JPG, PNG, TIFF, BMP, WEBP)."""
    try:
        from PIL import Image as _PILImage
        img = _PILImage.open(path)
        if img.mode not in ("RGB", "L", "RGBA"):
            img = img.convert("RGB")
        return _ocr_pil_image(img).strip()[:max_chars]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# PDF extraction — digital text first, OCR fallback for scanned docs
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path, max_chars: int) -> str:
    # Step 1: native text extraction (fast, works for digital PDFs)
    digital_text = ""
    page_count = 0
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        parts: list[str] = []
        total = 0
        for page in reader.pages:
            t = page.extract_text() or ""
            if total + len(t) > max_chars:
                parts.append(t[:max_chars - total])
                break
            parts.append(t)
            total += len(t)
            if total >= max_chars:
                break
        digital_text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    except Exception:
        pass

    # Step 2: if text is suspiciously sparse (< 80 chars/page avg), try OCR
    avg_chars_per_page = len(digital_text) / max(page_count, 1)
    if avg_chars_per_page < 80:
        ocr_text = _ocr_pdf_pages(path, max_chars)
        if len(ocr_text) > len(digital_text):
            return ocr_text

    return digital_text


def _extract_docx(path: Path, max_chars: int) -> str:
    try:
        from docx import Document  # lazy import — only needed when processing DOCX
        doc = Document(str(path))
        parts: list[str] = []
        total = 0
        for para in doc.paragraphs:
            if total >= max_chars:
                break
            t = para.text
            parts.append(t)
            total += len(t)
        return re.sub(r"\s+", " ", "\n".join(parts)).strip()[:max_chars]
    except Exception:
        return ""


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}


def extract_text_preview(path: Path, max_chars: int = MAX_TEXT_PREVIEW) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path, max_chars)

    if suffix == ".docx":
        return _extract_docx(path, max_chars)

    if suffix in _IMAGE_EXTS:
        return _extract_image_ocr(path, max_chars)

    # Plain text / CSV / JSON / etc.
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if b"\x00" in data[:1000]:
        return ""
    for enc in ("utf-8", "latin-1"):
        try:
            return re.sub(r"\s+", " ", data.decode(enc)).strip()[:max_chars]
        except UnicodeDecodeError:
            continue
    return ""
