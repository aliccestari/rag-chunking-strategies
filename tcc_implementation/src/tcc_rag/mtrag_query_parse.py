"""Converte o campo `text` das queries BEIR em `input` (lista de mensagens) + pergunta atual."""

from __future__ import annotations

import re

_USER_LINE = re.compile(r"^\|user\|\s*:\s*(.*)$", re.IGNORECASE)


def texto_para_mensagens(texto_bruto: str) -> list[dict]:
    """
    Cada linha |user|: ... vira um turno (speaker=user).
    Ignora linhas vazias ou sem o marcador.
    """
    turnos: list[str] = []
    for linha in texto_bruto.strip().split("\n"):
        linha = linha.strip()
        if not linha:
            continue
        m = _USER_LINE.match(linha)
        if m:
            turnos.append(m.group(1).strip().strip('"').strip())
        elif turnos:
            turnos[-1] = f"{turnos[-1]} {linha}".strip()
    return [{"speaker": "user", "text": t} for t in turnos if t]


def historico_e_pergunta_atual(mensagens: list[dict]) -> tuple[str, str]:
    if not mensagens:
        return "N/A (sem turnos parseados).", ""
    if len(mensagens) == 1:
        return "N/A (primeiro turno).", str(mensagens[-1]["text"])
    hist = "\n".join(f"User: {m['text']}" for m in mensagens[:-1])
    return hist, str(mensagens[-1]["text"])
