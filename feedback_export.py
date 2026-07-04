"""
Exportador do flywheel de feedback: transforma respostas avaliadas no chat
(👍/👎) em exemplos do dataset/.

  👍  →  dataset/17_feedback.jsonl          (few-shot em runtime + treino)
  👎  →  dataset/00_feedback_negativo.jsonl (bad_example: excluído de tudo,
                                             útil só para análise e futuro DPO)

Idempotente: pares já exportados (hash pergunta+resposta) são pulados, então
pode rodar quantas vezes quiser. Depois de exportar, reinicie o app para o
few-shot dinâmico indexar os exemplos novos.

Uso:
  python feedback_export.py            # exporta
  python feedback_export.py --dry-run  # só mostra o que seria exportado
"""

import argparse
import hashlib
import json
from pathlib import Path

import db as database

DATASET_DIR = Path(__file__).parent / "dataset"
GOOD_PATH = DATASET_DIR / "17_feedback.jsonl"
BAD_PATH = DATASET_DIR / "00_feedback_negativo.jsonl"

# Resposta longa demais vira exemplo ruim de treino (estoura token limit)
# e infla o few-shot — melhor curar manualmente do que exportar cega
MAX_ANSWER_CHARS = 2500
MIN_ANSWER_CHARS = 10


def _pair_hash(user: str, assistant: str) -> str:
    return hashlib.sha1(f"{user}\x00{assistant}".encode("utf-8")).hexdigest()


def _existing_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    if not path.exists():
        return hashes
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = rec.get("messages", [])
            user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            if user and asst:
                hashes.add(_pair_hash(user, asst))
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta feedback 👍/👎 para o dataset/")
    parser.add_argument("--dry-run", action="store_true", help="Só mostra, não grava")
    args = parser.parse_args()

    pairs = database.get_feedback_pairs()
    if not pairs:
        print("Nenhuma resposta avaliada ainda. Use 👍/👎 no chat primeiro.")
        return

    seen_good = _existing_hashes(GOOD_PATH)
    seen_bad = _existing_hashes(BAD_PATH)

    new_good: list[dict] = []
    new_bad: list[dict] = []
    skipped: list[str] = []

    for p in pairs:
        user, asst = p["user"].strip(), p["assistant"].strip()
        if len(asst) < MIN_ANSWER_CHARS:
            continue
        if len(asst) > MAX_ANSWER_CHARS:
            skipped.append(f"{asst[:60]!r}... ({len(asst)} chars — longa demais, cure manualmente)")
            continue
        h = _pair_hash(user, asst)
        record = {"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": asst},
        ]}
        if p["feedback"] == "up" and h not in seen_good:
            new_good.append(record)
            seen_good.add(h)
        elif p["feedback"] == "down" and h not in seen_bad:
            record["bad_example"] = True
            new_bad.append(record)
            seen_bad.add(h)

    print(f"Avaliações no banco: {len(pairs)}")
    print(f"Novos exemplos 👍:   {len(new_good)}  → {GOOD_PATH.name}")
    print(f"Novos exemplos 👎:   {len(new_bad)}  → {BAD_PATH.name}")
    for s in skipped:
        print(f"  [PULADO] {s}")

    if args.dry_run:
        for r in new_good[:5]:
            print(f"  👍 {r['messages'][0]['content'][:70]!r}")
        print("(dry-run: nada gravado)")
        return

    for path, records in ((GOOD_PATH, new_good), (BAD_PATH, new_bad)):
        if records:
            with path.open("a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if new_good or new_bad:
        print("\nExportado. Reinicie o app para o few-shot indexar os exemplos novos.")
    else:
        print("\nNada novo para exportar.")


if __name__ == "__main__":
    main()
