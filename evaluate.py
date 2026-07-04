"""
Avaliação do pipeline completo (RAG + memória + prompt + modelo).

Roda um conjunto fixo de perguntas (eval/perguntas.jsonl) contra o MESMO
pipeline usado no chat e salva os resultados em CSV. Rode antes e depois de
cada mudança de prompt/chunking/busca e compare os CSVs.

Uso:
  python evaluate.py --model llama3.2:1b
  python evaluate.py --model gemma3:4b --personality default
  python evaluate.py --model llama3.2:1b --only q04,q06

Colunas de pontuação manual (1-5) ficam em branco no CSV:
  relevancia — usou os documentos/fatos certos?
  precisao   — as informações batem?
  formato    — estrutura adequada ao pedido?
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import ai_engine
import db as database
import memory as mem
from app import (
    build_ollama_messages,
    LOCAL_RETRIEVAL_LIMIT, LOCAL_RETRIEVAL_CHAR_BUDGET,
    LOCAL_KNOWLEDGE_LIMIT, LOCAL_KNOWLEDGE_CHAR_BUDGET,
)
from utils import tokenize

EVAL_DIR = Path(__file__).parent / "eval"
QUESTIONS_PATH = EVAL_DIR / "perguntas.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


def load_questions(only: set[str] | None = None) -> list[dict]:
    if not QUESTIONS_PATH.exists():
        print(f"[ERRO] {QUESTIONS_PATH} não encontrado.")
        sys.exit(1)
    questions = []
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if only and q.get("id") not in only:
                continue
            questions.append(q)
    return questions


def run_question(model: str, question: dict, personality: str) -> dict:
    pergunta = str(question["pergunta"])
    memories = mem.retrieve(
        pergunta,
        limit=LOCAL_RETRIEVAL_LIMIT,
        char_budget=LOCAL_RETRIEVAL_CHAR_BUDGET,
    )
    knowledge = database.search_knowledge(
        tokenize(pergunta),
        limit=LOCAL_KNOWLEDGE_LIMIT,
        char_budget=LOCAL_KNOWLEDGE_CHAR_BUDGET,
        query_text=pergunta,
    )
    messages = build_ollama_messages(
        "", pergunta, memories,
        personality_id=personality, provider="local", raw_prompt=pergunta,
        knowledge=knowledge,
    )

    start = time.time()
    try:
        answer = "".join(ai_engine.stream_chat(model, messages, provider="local"))
        error = ""
    except Exception as exc:
        answer, error = "", str(exc)
    elapsed = time.time() - start

    expected = [str(k) for k in question.get("esperado_keywords", [])]
    answer_lower = answer.lower()
    found = [k for k in expected if k.lower() in answer_lower]
    hit_rate = (len(found) / len(expected)) if expected else None

    return {
        "id": question.get("id", ""),
        "categoria": question.get("categoria", ""),
        "pergunta": pergunta,
        "resposta": answer,
        "erro": error,
        "keywords_esperadas": ", ".join(expected),
        "keywords_encontradas": ", ".join(found),
        "taxa_acerto": f"{hit_rate:.2f}" if hit_rate is not None else "",
        "memorias_usadas": len(memories),
        "referencias_usadas": len(knowledge),
        "latencia_s": f"{elapsed:.1f}",
        "relevancia_1a5": "",
        "precisao_1a5": "",
        "formato_1a5": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação Camargo AI")
    parser.add_argument("--model", required=True, help="Modelo Ollama local (ex: llama3.2:1b)")
    parser.add_argument("--personality", default="atlas")
    parser.add_argument("--only", default="", help="IDs separados por vírgula (ex: q01,q04)")
    args = parser.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    questions = load_questions(only)
    if not questions:
        print("[ERRO] Nenhuma pergunta selecionada.")
        sys.exit(1)

    sync = database.sync_knowledge(force=True)
    if sync.get("chunks_indexed"):
        print(f"Base de conhecimento reindexada: {sync['chunks_indexed']} chunks de {sync['files']} arquivos")
    print(f"Avaliando {len(questions)} perguntas com {args.model} (personalidade: {args.personality})\n")

    rows = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['pergunta'][:60]}...")
        row = run_question(args.model, q, args.personality)
        status = f"ERRO: {row['erro'][:50]}" if row["erro"] else (
            f"acerto={row['taxa_acerto'] or 'n/a'} | {row['latencia_s']}s"
        )
        print(f"        {status}")
        rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.replace(":", "-").replace("/", "-")
    out_path = RESULTS_DIR / f"eval_{stamp}_{model_slug}.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    scored = [float(r["taxa_acerto"]) for r in rows if r["taxa_acerto"]]
    avg_hit = sum(scored) / len(scored) if scored else 0.0
    latencies = [float(r["latencia_s"]) for r in rows if not r["erro"]]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    errors = sum(1 for r in rows if r["erro"])

    print(f"\n{'=' * 52}")
    print(f"Taxa média de keywords esperadas: {avg_hit:.0%}")
    print(f"Latência média:                   {avg_lat:.1f}s")
    print(f"Erros:                            {errors}/{len(rows)}")
    print(f"Resultado salvo em:               {out_path}")
    print(f"{'=' * 52}")
    print("Abra o CSV e pontue relevancia/precisao/formato (1-5) manualmente.")


if __name__ == "__main__":
    main()
