#!/usr/bin/env bash
# Backup dos dados privados — tudo que o git NÃO versiona e não é regenerável:
#   data/app.db            conversas, memórias, insights, índice da base
#   data/facts.json        fatos permanentes sobre o usuário
#   knowledge/doc_*.md     documentos de trabalho ingeridos
#   knowledge/perfil_*.md  perfil pessoal
#   dataset/*feedback*     exemplos aprovados/reprovados no chat
#
# Uso:
#   ./backup.sh                  # gera ~/camargo_backup_<data>.tar.gz
#   ./backup.sh /media/pendrive  # gera no destino indicado
#
# Restauração no PC novo (depois de git clone + venv):
#   tar -xzf camargo_backup_<data>.tar.gz -C /caminho/do/camargo-ai
set -euo pipefail
cd "$(dirname "$0")"

DEST="${1:-$HOME}"
OUT="$DEST/camargo_backup_$(date +%Y%m%d_%H%M%S).tar.gz"

FILES=()
for f in data/app.db data/facts.json \
         knowledge/perfil_marcelo_camargo.md \
         dataset/17_feedback.jsonl dataset/00_feedback_negativo.jsonl \
         knowledge/doc_*.md; do
  [ -e "$f" ] && FILES+=("$f")
done

if [ ${#FILES[@]} -eq 0 ]; then
  echo "Nada para copiar (nenhum dado privado encontrado)."
  exit 1
fi

tar -czf "$OUT" "${FILES[@]}"
echo "Backup criado: $OUT"
echo "Conteúdo:"
tar -tzf "$OUT" | sed 's/^/  /'
