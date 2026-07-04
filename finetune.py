"""
Fine-tuning pipeline para o assistente Camargo AI.
Suporta: Unsloth (rápido, recomendado) ou HuggingFace Trainer (fallback).

Uso:
  python finetune.py --model unsloth/Phi-3-mini-4k-instruct --epochs 3
  python finetune.py --model unsloth/llama-3-8b-instruct --epochs 5 --output ./model_out
  python finetune.py --validate-only   # só valida o dataset
"""

import argparse
import json
import os
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset"
SYSTEM_PROMPT = (
    "Você é um assistente inteligente, cordial, prestativo e profissional. "
    "Responda sempre com clareza, educação e objetividade. "
    "Adapte o nível técnico ao perfil demonstrado pelo usuário. "
    "Priorize soluções práticas e resolução real dos problemas. "
    "Quando houver incerteza, sinalize claramente sem inventar. "
    "Nunca fabrique informações, datas, nomes ou dados. "
    "Ao revisar textos, entregue sempre uma versão pronta para uso. "
    "Ao explicar temas complexos, utilize exemplos e analogias. "
    "Seja direto, mas nunca seco ou distante."
)


# ---------------------------------------------------------------------------
# Dataset loading & validation
# ---------------------------------------------------------------------------

def load_dataset(exclude_bad: bool = True) -> list[dict]:
    """Load all JSONL files from dataset/, optionally skipping bad examples."""
    records = []
    files = sorted(DATASET_DIR.glob("*.jsonl"))
    if not files:
        print(f"[ERRO] Nenhum arquivo .jsonl encontrado em {DATASET_DIR}")
        sys.exit(1)

    for fpath in files:
        count = 0
        with fpath.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[AVISO] {fpath.name}:{lineno} — JSON inválido: {e}")
                    continue
                if exclude_bad and record.get("bad_example"):
                    continue
                records.append(record)
                count += 1
        print(f"  {fpath.name}: {count} exemplos carregados")

    return records


def validate_dataset(records: list[dict]) -> bool:
    """Validate dataset structure and quality."""
    errors = 0
    warnings = 0

    for i, record in enumerate(records):
        msgs = record.get("messages", [])
        if not msgs:
            print(f"[ERRO] Registro {i}: sem campo 'messages'")
            errors += 1
            continue

        roles = [m.get("role") for m in msgs]

        if roles[0] not in ("system", "user"):
            print(f"[AVISO] Registro {i}: primeira mensagem deve ser 'system' ou 'user', encontrado '{roles[0]}'")
            warnings += 1

        if roles[-1] != "assistant":
            print(f"[ERRO] Registro {i}: última mensagem deve ser 'assistant', encontrado '{roles[-1]}'")
            errors += 1

        for j, msg in enumerate(msgs):
            if not msg.get("content", "").strip():
                print(f"[AVISO] Registro {i}, mensagem {j}: conteúdo vazio")
                warnings += 1

        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        for msg in assistant_msgs:
            content = msg.get("content", "")
            if len(content) < 10:
                print(f"[AVISO] Registro {i}: resposta muito curta ({len(content)} chars)")
                warnings += 1
            if len(content) > 3000:
                print(f"[AVISO] Registro {i}: resposta muito longa ({len(content)} chars) — pode exceder token limit")
                warnings += 1

    print(f"\nValidação: {len(records)} registros | {errors} erros | {warnings} avisos")
    return errors == 0


def dataset_stats(records: list[dict]) -> None:
    """Print dataset statistics."""
    total = len(records)
    single_turn = sum(1 for r in records if len(r.get("messages", [])) <= 3)
    multi_turn = total - single_turn
    avg_assistant_len = 0
    lens = []
    for r in records:
        for m in r.get("messages", []):
            if m.get("role") == "assistant":
                lens.append(len(m.get("content", "")))
    if lens:
        avg_assistant_len = sum(lens) // len(lens)

    print(f"\n{'='*50}")
    print(f"DATASET STATISTICS")
    print(f"{'='*50}")
    print(f"Total de exemplos:        {total}")
    print(f"Single-turn:              {single_turn}")
    print(f"Multi-turn:               {multi_turn}")
    print(f"Comprimento médio (asst): {avg_assistant_len} chars")
    print(f"Comprimento mín (asst):   {min(lens) if lens else 0} chars")
    print(f"Comprimento máx (asst):   {max(lens) if lens else 0} chars")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_for_openai(records: list[dict], out_path: Path) -> None:
    """Export in OpenAI fine-tuning JSONL format (compatible with most providers)."""
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            msgs = record.get("messages", [])
            # Inject default system prompt if none present
            if msgs and msgs[0].get("role") != "system":
                msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"Dataset exportado: {out_path} ({len(records)} exemplos)")


def export_for_axolotl(records: list[dict], out_path: Path) -> None:
    """Export in Alpaca/ShareGPT format for Axolotl."""
    converted = []
    for record in records:
        msgs = record.get("messages", [])
        if msgs and msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
        converted.append({"conversations": [{"from": m["role"], "value": m["content"]} for m in msgs]})
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"Dataset Axolotl exportado: {out_path} ({len(converted)} exemplos)")


# ---------------------------------------------------------------------------
# Training (Unsloth)
# ---------------------------------------------------------------------------

def train_unsloth(records: list[dict], model_name: str, output_dir: str, epochs: int, batch_size: int) -> None:
    try:
        import torch
    except ImportError:
        print("[ERRO] PyTorch não instalado. Execute: pip install torch")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[ERRO] Nenhuma GPU NVIDIA (CUDA) detectada nesta máquina.")
        print("Unsloth exige GPU CUDA — não roda em CPU nem em GPU integrada (Intel/AMD).")
        print("Instalar unsloth/trl aqui não resolve, pois o treino não vai rodar de qualquer forma.\n")
        print("Alternativas:")
        print("  1. Treinar em ambiente com GPU NVIDIA (ex: Google Colab com T4 grátis, RunPod, Lambda),")
        print("     usando o dataset exportado por este script:")
        print("       python finetune.py --export-axolotl dataset_axolotl.json")
        print("       python finetune.py --export-openai dataset_openai.jsonl")
        print("  2. Continuar melhorando via prompt/few-shot (personality.py / few_shot.py / dataset/),")
        print("     que não depende de GPU.")
        sys.exit(1)

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError:
        print("[ERRO] Unsloth ou TRL não instalado. Execute:")
        print("  pip install unsloth trl datasets transformers accelerate")
        sys.exit(1)

    print(f"\nCarregando modelo: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing=True,
    )

    def format_example(record: dict) -> str:
        msgs = record.get("messages", [])
        if msgs and msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    texts = [format_example(r) for r in records]
    dataset = Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            num_train_epochs=epochs,
            learning_rate=1e-5,
            fp16=True,
            logging_steps=10,
            output_dir=output_dir,
            save_strategy="epoch",
            report_to="none",
        ),
    )

    print(f"\nIniciando treinamento: {epochs} épocas, batch={batch_size}")
    trainer.train()

    print(f"\nSalvando modelo em {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Treinamento concluído!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning Camargo AI")
    parser.add_argument("--model", default="unsloth/Phi-3-mini-4k-instruct",
                        help="Modelo base (HuggingFace ou Unsloth)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="./camargo_finetuned",
                        help="Diretório de saída do modelo treinado")
    parser.add_argument("--validate-only", action="store_true",
                        help="Apenas valida o dataset sem treinar")
    parser.add_argument("--export-openai", metavar="PATH",
                        help="Exporta dataset no formato OpenAI JSONL")
    parser.add_argument("--export-axolotl", metavar="PATH",
                        help="Exporta dataset no formato Axolotl/ShareGPT")
    parser.add_argument("--include-bad", action="store_true",
                        help="Inclui exemplos marcados como bad_example (não recomendado para treino)")
    args = parser.parse_args()

    print(f"\nCarregando dataset de {DATASET_DIR}...")
    records = load_dataset(exclude_bad=not args.include_bad)
    print(f"\nTotal: {len(records)} exemplos carregados\n")

    dataset_stats(records)
    valid = validate_dataset(records)

    if args.export_openai:
        export_for_openai(records, Path(args.export_openai))

    if args.export_axolotl:
        export_for_axolotl(records, Path(args.export_axolotl))

    if args.validate_only:
        sys.exit(0 if valid else 1)

    if not valid:
        print("\n[ERRO] Dataset contém erros. Corrija antes de treinar.")
        sys.exit(1)

    train_unsloth(
        records=records,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
