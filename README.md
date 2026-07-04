# Marcellus (camargo-ai)

Assistente de IA **100% local** construído para fazer um modelo pequeno (4B)
render como um grande — via camada de sistema, não via hardware. Flask +
Ollama + SQLite, sem serviços pagos, com foco em trabalho administrativo
público (pareceres, legislação, redação oficial).

> A tese do projeto: entre um modelo mediano bem servido de contexto e um
> modelo forte às cegas, o mediano ganha. Toda a engenharia aqui serve
> contexto certo, na hora certa, no formato que um 4B consegue usar.

## Recursos

| Recurso | Como funciona |
|---|---|
| **RAG híbrido** | Base de conhecimento (`knowledge/`) indexada por seções `##`, busca BM25 + embeddings com fusão por score |
| **Memória entre conversas** | Tudo que é conversado vira memória recuperável por significado em qualquer conversa futura; insights consolidados a cada 5 mensagens |
| **Few-shot dinâmico** | Exemplos do `dataset/` entram por semelhança com a pergunta (não injeção cega) — menos tokens, mais pertinência |
| **Expansão de consulta** | Ajudante 0.8b gera sinônimos quando a busca local volta vazia |
| **Ingestão de documentos** | `ingest.py` converte PDF/DOCX/TXT em seções coesas: leis por artigo, pareceres por EMENTA→CONCLUSÃO, manuais por capítulo |
| **Flywheel de feedback** | 👍/👎 no chat → `feedback_export.py` → exemplos de few-shot hoje e de fine-tune amanhã |
| **Cálculo verificado** | Pergunta de cálculo → ajudante traduz em expressão → Python executa → modelo apresenta o resultado exato |
| **Modo Rigoroso** | Segundo passe que confere a resposta contra as referências antes de entregar |
| **Consulta web opcional** | 🌐 liga LexML (legislação oficial) + busca geral; desligado, nada sai da máquina |
| **Citação condicional** | Fonte citada só em matéria oficial (lei, decreto, parecer...) |
| **Busca global 🔍** | Encontra em qual conversa algo foi dito |
| **Fine-tuning** | `finetune.py` (validação/export) + `colab_finetune.ipynb` (LoRA no Colab T4, export GGUF pro Ollama) |

## Requisitos

- Python 3.10+, [Ollama](https://ollama.com) rodando localmente
- Modelos: um de chat + os dois ajudantes fixos

```bash
ollama pull qwen3.5:4b          # chat (ou o modelo que preferir)
ollama pull qwen3.5:0.8b        # ajudante (expansão, cálculo)
ollama pull nomic-embed-text    # embeddings
```

## Como rodar

```bash
git clone https://github.com/camargoapps/camargo-ai.git
cd camargo-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py                   # abre em http://127.0.0.1:5000
```

## Estrutura

```
app.py            Flask + frontend embutido + orquestração do chat
db.py             SQLite: conversas, memórias, base de conhecimento, busca híbrida
memory.py         memória entre conversas (gravação, resgate, consolidação)
few_shot.py       seleção dinâmica de exemplos do dataset/
query_expand.py   expansão de consulta (resgate quando a busca falha)
math_tool.py      cálculo verificado por execução
web_search.py     LexML + busca web (atrás do toggle 🌐)
reasoning.py      andaime de raciocínio + parâmetros por tarefa
ingest.py         documentos → knowledge/ estruturado
feedback_export.py  👍/👎 → dataset/
finetune.py       validação de dataset + treino (Unsloth)
evaluate.py       harness de avaliação/regressão
backup.sh         backup dos dados privados (fora do git)
knowledge/        base de conhecimento (.md por seções ##)
dataset/          exemplos comportamentais (few-shot + treino)
data/             runtime do usuário (banco, facts.json) — fora do git
```

## Configuração (variáveis de ambiente)

| Variável | Padrão | Para quê |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | endpoint do Ollama |
| `OLLAMA_NUM_CTX` | `8192` | janela de contexto local |
| `HELPER_MODEL` | `qwen3.5:0.8b` | modelo dos ajudantes de bastidor |
| `EXPAND_MODEL` / `MATH_MODEL` | = `HELPER_MODEL` | sobrepor papel individual |

## Alimentando a base de conhecimento

```bash
python ingest.py lei_14133.pdf              # detecta tipo/nome, divide, indexa
python ingest.py pasta_de_documentos/       # lote
python ingest.py doc.pdf --dry-run          # espiar as seções antes
```

Ou crie `.md` à mão em `knowledge/` com seções `##` autocontidas (~2.000
chars) — a indexação é automática. Fatos permanentes sobre o usuário vão em
`data/facts.json` (modelo em `data/facts.example.json`).

## Privacidade

O `.gitignore` garante que **dados pessoais e de trabalho nunca sobem**:
`data/` inteira, `knowledge/doc_*.md` (documentos ingeridos), perfil pessoal
e os arquivos de feedback. A consulta web é opt-in por mensagem; todo o
restante roda offline.

## Backup e migração de máquina

O git versiona só o código. Os dados privados migram via:

```bash
./backup.sh                    # gera camargo_backup_<data>.tar.gz em ~
./backup.sh /media/pendrive    # ou direto no pendrive
```

No PC novo: `git clone` + venv + modelos do Ollama, depois
`tar -xzf camargo_backup_<data>.tar.gz -C camargo-ai/` — e o sistema acorda
com as memórias, base e feedback intactos. Caches de embeddings se
regeneram sozinhos.
