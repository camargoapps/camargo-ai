# IA e Tecnologia — RAG, LLM e Fine-tuning

## RAG — Retrieval-Augmented Generation (Conceito)
- RAG é uma arquitetura que combina recuperação de documentos com geração de texto por LLM.
- Fluxo básico: pergunta do usuário → embedding da pergunta → busca vetorial no banco → chunks relevantes retornados → LLM gera resposta usando os chunks como contexto.
- Vantagem principal: o modelo não precisa memorizar os dados — eles ficam no banco e são recuperados sob demanda.
- Quando usar RAG: bases de conhecimento que mudam com o tempo, documentos proprietários, volume que não cabe no contexto do modelo.
- Quando NÃO usar RAG: conhecimento estático e pequeno que cabe em um system prompt; tarefas que exigem raciocínio, não recuperação.

## ChromaDB — Banco Vetorial Local
- ChromaDB é um banco de vetores open-source, embutível em Python, sem necessidade de servidor externo na versão padrão.
- Instalação: `pip install chromadb`.
- Operações principais: `client.create_collection()`, `collection.add(documents, embeddings, ids)`, `collection.query(query_embeddings, n_results)`.
- Persistência: `chromadb.PersistentClient(path="./chroma_db")` — grava em disco; sem isso, dados somem ao reiniciar.
- Distância padrão: cosseno (cosine similarity) — adequado para textos; L2 (euclidiana) disponível como alternativa.

## Embeddings — Conceito e Modelos
- Embedding é uma representação vetorial de texto que captura significado semântico — textos similares ficam próximos no espaço vetorial.
- Dimensão típica: 768 (BERT-base), 1536 (text-embedding-ada-002 OpenAI), 768 (nomic-embed-text via Ollama).
- nomic-embed-text: modelo de embedding local gratuito, roda via Ollama (`ollama pull nomic-embed-text`), dimensão 768, contexto de 8192 tokens.
- Regra de ouro: use sempre o mesmo modelo para indexar e para buscar — trocar o modelo exige reindexar tudo.
- Embedding não é geração de texto — é uma função que transforma texto em números; não responde perguntas, só calcula proximidade.

## Ollama — LLMs Locais
- Ollama é uma ferramenta para baixar e rodar modelos LLM localmente, sem depender de API externa.
- Instalação Linux: `curl -fsSL https://ollama.com/install.sh | sh`.
- Comandos principais: `ollama pull llama3`, `ollama run llama3`, `ollama list`, `ollama rm modelo`.
- Formatos suportados: GGUF (quantizados), GGML legado.
- API REST local: `http://localhost:11434/api/generate` e `/api/chat` — compatível com padrão OpenAI via `/v1/chat/completions`.
- Quantização Q4_K_M: boa relação qualidade/velocidade para uso em CPU/GPU consumidor; perde ~2–4% de qualidade vs. FP16.

## Fine-tuning vs RAG — Quando Usar Cada Um
- RAG: ideal para injetar conhecimento factual, documentos, bases de dados — o modelo não precisa "aprender", só recuperar.
- Fine-tuning: ideal para mudar o estilo, tom, formato de resposta, ou para especializar o modelo em um domínio com padrões linguísticos específicos.
- LoRA (Low-Rank Adaptation): técnica de fine-tuning eficiente — congela os pesos originais e treina apenas matrizes de baixa dimensão adicionadas às camadas de atenção.
- QLoRA: LoRA + quantização do modelo base (4-bit) — treina modelos grandes (7B, 13B) em GPUs com 12–16 GB de VRAM.
- Dataset mínimo para fine-tuning viável: ~500–2.000 exemplos no formato `{"instruction": "...", "input": "...", "output": "..."}` (Alpaca format).

## LoRA — Parâmetros Principais
- `r` (rank): dimensão das matrizes LoRA — valores típicos: 8, 16, 32. Maior r = mais capacidade, mais memória.
- `lora_alpha`: escala de aprendizado — regra prática: `lora_alpha = 2 * r` (ex.: r=16, alpha=32).
- `lora_dropout`: regularização — valor típico 0.05 a 0.1.
- `target_modules`: camadas onde LoRA é aplicado — geralmente `["q_proj", "v_proj"]` para modelos LLaMA-based.
- Resultado do fine-tuning: um arquivo de adaptador (`.safetensors` ou pasta) que se combina com o modelo base em tempo de inferência.

## Flask — Backend para RAG Local
- Flask é um microframework Python para criar APIs HTTP — usado como backend do Camargo AI.
- Rota básica: `@app.route('/chat', methods=['POST'])` — recebe JSON, processa, retorna JSON.
- Integração com ChromaDB + Ollama: busca vetorial retorna chunks → chunks são concatenados ao prompt → prompt enviado ao Ollama via `requests.post()`.
- Contexto de conversa: Flask não mantém estado entre requests — histórico deve ser enviado pelo cliente ou armazenado em sessão/banco.
- Debug local: `app.run(debug=True, port=5000)` — nunca usar `debug=True` em produção.

## Chunking — Estratégias de Divisão de Texto
- Chunk é a unidade mínima indexada no banco vetorial — tamanho influencia diretamente a qualidade da recuperação.
- Chunk muito grande: captura muito contexto, mas dilui a relevância semântica — busca fica imprecisa.
- Chunk muito pequeno: busca precisa, mas chunk recuperado pode não ter informação suficiente para o LLM responder.
- Tamanho recomendado para documentos técnicos: 300–600 tokens com overlap de 50–100 tokens.
- Separadores naturais: quebra por `##` (Markdown), por parágrafo (`\n\n`), ou por frase — preferir quebra semântica à quebra por número fixo de caracteres.

## VRAM e Hardware para LLMs Locais
- Regra básica: modelo FP16 ocupa ~2 GB de VRAM por bilhão de parâmetros (ex.: 7B = ~14 GB).
- Quantizado Q4_K_M: ~0,5 GB por bilhão (ex.: 7B = ~4–5 GB; 13B = ~8 GB).
- RTX 3060 12 GB GDDR6: roda confortavelmente modelos até 13B em Q4_K_M; modelos 7B em Q8 com folga.
- CPU offload (llama.cpp/Ollama): camadas que não cabem na VRAM são processadas na RAM — muito mais lento, mas funcional.
- Para fine-tuning QLoRA de 7B: mínimo 8 GB VRAM; 12 GB permite batch_size=2–4 com gradiente acumulado.
