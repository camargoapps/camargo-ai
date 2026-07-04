# Excel e informática prática

## Fórmulas essenciais do Excel (versão pt-BR usa ponto e vírgula)
- =SOMA(A1:A10) soma o intervalo; =MÉDIA(A1:A10) calcula a média.
- =SE(condição; valor_se_verdadeiro; valor_se_falso). Ex.: =SE(B2>=7;"Aprovado";"Reprovado").
- =CONT.SE(A1:A100;"critério") conta células que atendem ao critério; =CONT.VALORES conta as não vazias.
- =SOMASE(intervalo_do_critério; critério; intervalo_da_soma). Com vários critérios: =SOMASES e =CONT.SES.
- =PROCV(valor; tabela; nº_da_coluna; FALSO) busca vertical exata; no Excel mais novo prefira =PROCX(valor; coluna_de_busca; coluna_de_retorno).
- Texto: =CONCAT(A1;" ";B1) ou A1&" "&B1; =MAIÚSCULA, =MINÚSCULA, =ARRUMAR (remove espaços extras).
- Datas: =HOJE() dá a data atual; =DATADIF(início;fim;"d") diferença em dias ("m" meses, "y" anos); somar 30 dias: A1+30.
- =ARRED(valor; casas) arredonda; =SEERRO(fórmula;"") evita mostrar erro na célula.
- Referência fixa com $: $A$1 não muda ao arrastar a fórmula (a tecla F4 alterna entre as formas).

## Atalhos úteis no Windows
- Ctrl+C copiar, Ctrl+V colar, Ctrl+X recortar, Ctrl+Z desfazer, Ctrl+Y refazer, Ctrl+F localizar.
- Excel: Ctrl+Shift+L liga/desliga filtros; Ctrl+Setas pula para a borda dos dados; F2 edita a célula.
- Windows+Shift+S captura um trecho da tela; Alt+Tab alterna janelas; Ctrl+P imprimir.

## Arquivos e formatos
- PDF preserva o layout — use para versão final e assinatura; DOCX/ODT para documentos em edição.
- XLSX/ODS: planilhas; CSV: texto separado por vírgula ou ponto e vírgula, ideal para importar/exportar dados entre sistemas.
- ZIP compacta vários arquivos em um; JPG para fotos; PNG para capturas de tela.
- Nomeação recomendada: datas no padrão AAAA-MM-DD para ordenar corretamente (ex.: 2026-07-04_parecer_estagio.pdf) e evitar acentos e espaços em nomes de arquivo para sistemas antigos.

## Boas práticas com planilhas de dados
- Uma linha por registro, uma coluna por campo, cabeçalho na primeira linha, sem células mescladas na área de dados.
- Use Dados > Validação de Dados para criar listas suspensas e evitar erros de digitação.
- Congele o cabeçalho (Exibir > Congelar Painéis) e use Página Inicial > Formatar como Tabela para ganhar filtros automáticos.

## Conceitos de IA no dia a dia
- LLM: modelo que gera texto prevendo a próxima palavra; não consulta uma base de fatos, por isso pode "alucinar" informações.
- RAG: recupera trechos de documentos reais e os entrega ao modelo junto com a pergunta — reduz alucinação e permite citar a fonte.
- Token: pedaço de palavra (cerca de 4 caracteres); os modelos têm limite de contexto medido em tokens.
- Fine-tuning/LoRA: treina o estilo e o comportamento com exemplos; RAG fornece conhecimento, fine-tuning fornece comportamento.
