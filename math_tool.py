"""
Calculadora verificada: pergunta de cálculo → expressão → execução exata.

Modelo pequeno chuta aritmética ("15% de 2.340" vira qualquer número
plausível). Aqui o LLM só TRADUZ a pergunta em expressão ("0.15*2340");
quem calcula é o Python. O resultado entra no prompt como fato verificado
e o modelo apenas o apresenta.

Segurança: a expressão só executa se contiver exclusivamente dígitos,
operadores aritméticos, parênteses e ponto — nada de nomes, chamadas ou
builtins. Falha silenciosa em qualquer dúvida.
"""

import os
import re

import ai_engine

MATH_MODEL = os.environ.get("MATH_MODEL", os.environ.get("HELPER_MODEL", "qwen3.5:0.8b"))

_CALC_HINTS = (
    "%", "quanto é", "quanto e ", "quanto dá", "quanto da ", "calcul",
    "some ", "soma de", "somar", "subtrai", "multiplic", "divid",
    "média", "media de", "porcent", "percentual", "reajuste", "desconto",
    "acréscimo", "acrescimo", "juros", "proporcional",
)

_EXPR_OK = re.compile(r"^[0-9+\-*/().\s]+$")

_PROMPT = (
    "Converta a pergunta em UMA única expressão aritmética Python, usando "
    "apenas números e os operadores + - * / ( ) **. Sem texto, sem "
    "variáveis, sem símbolo %. Porcentagem vira multiplicação (15% de 200 "
    "-> 0.15*200). Se a pergunta não for um cálculo, responda apenas NAO.\n\n"
    "Pergunta: {q}\nExpressão:"
)


def looks_like_calculation(prompt: str) -> bool:
    p = prompt.lower()
    return any(c.isdigit() for c in p) and any(h in p for h in _CALC_HINTS)


def solve(prompt: str) -> "str | None":
    """Retorna 'expressão = resultado' verificado, ou None."""
    if not looks_like_calculation(prompt):
        return None

    raw = ai_engine.generate_text(
        MATH_MODEL, _PROMPT.format(q=prompt.strip()[:400]),
        timeout=15, options={"num_predict": 40, "temperature": 0},
        keep_alive="30m",
    )
    expr = raw.strip().splitlines()[0].strip().strip("`") if raw.strip() else ""
    expr = expr.split("=")[0].strip().replace(",", ".")
    if not expr or "NAO" in expr.upper() or len(expr) > 80:
        return None
    if not _EXPR_OK.match(expr) or not any(c.isdigit() for c in expr):
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # whitelist garante só aritmética
    except Exception:
        return None
    if isinstance(result, float):
        result = round(result, 6)
        if result == int(result):
            result = int(result)
    return f"{expr} = {result}"
