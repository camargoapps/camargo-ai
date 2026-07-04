"""
Chain-of-thought forçado para perguntas analíticas.

Modelos pequenos (1B-4B) melhoram muito quando gerar a resposta passa por
etapas explícitas — o "pensar" deles acontece nos próprios tokens gerados.
Detectamos perguntas que exigem análise e anexamos um andaime de raciocínio
JUNTO da pergunta (fim do contexto), onde a atenção do modelo é mais forte.
"""

import re

# Sinais de que a pergunta exige análise, não recall simples
_ANALYTICAL_HINTS = (
    "analis", "análise", "analise", "avalie", "avaliar", "compare", "compara",
    "diferença", "diferenca", "por que", "porque", "por quê", "justifi",
    "explique", "explica como", "como funciona", "vantagens", "desvantagens",
    "prós", "pros e contras", "melhor opção", "melhor opcao", "devo ",
    "vale a pena", "parecer", "desvio de função", "desvio de funcao",
    "estágio probatório", "estagio probatorio", "acumulação", "acumulacao",
    "legalidade", "é legal", "e legal", "pode ser", "é possível", "e possivel",
    "calcul", "quanto", "estime", "impacto", "consequência", "consequencia",
    "risco", "decidir", "decisão", "decisao", "estratégia", "estrategia",
)

# Recall simples / conversa — nunca forçar CoT nesses casos
_SIMPLE_HINTS = (
    "oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "obrigad",
    "traduz", "resuma em uma frase", "que horas",
)

# Parâmetros de geração por tipo de tarefa: factual/jurídico pede precisão
# (temperatura baixa reduz alucinação); criativo pede variedade
_FACTUAL_HINTS = (
    "lei", "artigo", "decreto", "portaria", "norma", "estatuto", "parecer",
    "jurídic", "juridic", "legal", "prazo", "direito", "quanto", "calcul",
    "estágio probatório", "estagio probatorio", "desvio de função",
)
_CREATIVE_HINTS = (
    "poema", "história", "historia", "criativ", "ideias", "brainstorm",
    "slogan", "piada", "imagine", "invente", "metáfora", "metafora",
)


def task_options(prompt: str) -> "dict | None":
    """Options do Ollama ajustadas ao tipo de pergunta, ou None (default)."""
    p = prompt.lower()
    if any(h in p for h in _CREATIVE_HINTS):
        return {"temperature": 0.9}
    if any(h in p for h in _FACTUAL_HINTS):
        return {"temperature": 0.3, "top_p": 0.9}
    return None


REASONING_SCAFFOLD = (
    "\n\n[Método de resposta — siga internamente estas etapas, na ordem: "
    "1) identifique o que exatamente foi perguntado; "
    "2) liste quais fatos, documentos ou normas do contexto acima são relevantes; "
    "3) verifique se algum dado necessário está faltando — se estiver, diga isso; "
    "4) só então conclua. "
    "Apresente a resposta organizada: primeiro os pontos-chave do raciocínio "
    "em frases curtas, depois a conclusão clara. Não invente fatos que não "
    "estejam no contexto.]"
)

# Versão específica para pedidos de parecer técnico (formato PMC)
PARECER_SCAFFOLD = (
    "\n\n[Este pedido exige formato de parecer técnico. Estruture a resposta em: "
    "EMENTA (1-2 frases), RELATÓRIO (fatos do caso), FUNDAMENTAÇÃO "
    "(normas e precedentes do contexto, citando a fonte) e CONCLUSÃO (objetiva). "
    "Se faltar informação essencial, aponte na conclusão o que falta. "
    "Não invente leis, números ou datas.]"
)


def needs_reasoning(prompt: str) -> bool:
    """True quando a pergunta é analítica o bastante para merecer o andaime."""
    p = prompt.strip().lower()
    if len(p) < 25:
        return False
    first_words = p[:20]
    if any(first_words.startswith(h) for h in _SIMPLE_HINTS):
        return False
    hits = sum(1 for h in _ANALYTICAL_HINTS if h in p)
    if hits >= 1 and (len(p) > 60 or hits >= 2):
        return True
    # Perguntas longas com múltiplas orações tendem a ser analíticas
    return len(p) > 200 and ("?" in p or hits >= 1)


def wants_parecer(prompt: str) -> bool:
    p = prompt.lower()
    return bool(re.search(r"\bparecer(es)?\b", p)) and any(
        v in p for v in ("elabor", "redij", "redig", "escrev", "faça", "faca", "gere", "monte", "minuta")
    )


def build_scaffold(prompt: str) -> str:
    """Retorna o andaime adequado à pergunta, ou '' se não precisar."""
    if wants_parecer(prompt):
        return PARECER_SCAFFOLD
    if needs_reasoning(prompt):
        return REASONING_SCAFFOLD
    return ""
