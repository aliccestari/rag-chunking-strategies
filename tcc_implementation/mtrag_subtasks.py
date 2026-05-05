"""
Subtasks MTRAGEval A (retrieval), B (geração com passagens ouro), C (RAG ponta a ponta).

Embeddings/recuperação: BGE local via Chroma (mesmo índice que criar_db.py).
Geração: local_llm por padrão.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

from langchain_chroma.vectorstores import Chroma

from corpus_config import COLLECTION_NAME
from mtrag_query_parse import historico_e_pergunta_atual, texto_para_mensagens
from rag_prompts import RAG_ANSWER_PROMPT
from retrieval_core import abrir_indice, contexts_para_jsonl, recuperar_passagens_unicas


def num_turnos_utilizador(texto_query: str) -> int:
    """Número de turnos |user| no ficheiro BEIR (útil para análise H3 / comprimento de diálogo)."""
    return len(texto_para_mensagens(texto_query))


def indice_para_dominio(persist_directory: str | Path) -> Chroma:
    return abrir_indice(persist_directory)


def iter_linhas_queries(caminho: Path) -> Iterator[tuple[str, str]]:
    """Yield (task_id, texto_query) a partir do JSONL BEIR (_id, text)."""
    with caminho.open(encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            reg = json.loads(linha)
            tid = reg.get("_id") or reg.get("task_id")
            texto = reg.get("text", "")
            if tid is None:
                continue
            yield str(tid), texto


def carregar_mapa_passagens(caminho_corpus: Path) -> dict[str, dict]:
    """id da passagem -> {text, title} (para Task B e textos no JSONL)."""
    m: dict[str, dict] = {}
    with caminho_corpus.open(encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            reg = json.loads(linha)
            pid = str(reg.get("_id") or reg.get("id") or "")
            if not pid:
                continue
            m[pid] = {
                "text": reg.get("text", ""),
                "title": reg.get("title", "") or "",
            }
    return m


def carregar_qrels_por_query(caminho_qrels: Path) -> dict[str, list[str]]:
    """query_id -> lista de corpus-id relevantes (ordem do arquivo)."""
    por_q: dict[str, list[str]] = {}
    with caminho_qrels.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) < 3:
                continue
            qid, doc_id = row[0], row[1]
            if qid not in por_q:
                por_q[qid] = []
            if doc_id not in por_q[qid]:
                por_q[qid].append(doc_id)
    return por_q


def montar_prompt_rag(historico: str, pergunta: str, contextos: list[dict]) -> str:
    blocos = []
    for c in contextos:
        tit = c.get("title")
        prefix = f"[{c['document_id']}]"
        if tit:
            prefix += f" {tit}"
        blocos.append(f"{prefix}\n{c['text']}")
    base = "\n\n----\n\n".join(blocos)
    return RAG_ANSWER_PROMPT.format(
        historico=historico,
        base_conhecimento=base,
        pergunta=pergunta,
    )


def historico_e_pergunta_de_input(mensagens: list[dict]) -> tuple[str, str]:
    """Extrai histórico e pergunta atual preservando turnos user/agent oficiais."""
    if not mensagens:
        return "N/A (sem turnos parseados).", ""
    pergunta = str(mensagens[-1].get("text", ""))
    if len(mensagens) == 1:
        return "N/A (primeiro turno).", pergunta
    linhas = []
    for msg in mensagens[:-1]:
        speaker = str(msg.get("speaker", "user")).lower()
        rotulo = "Assistant" if speaker in ("agent", "assistant") else "User"
        linhas.append(f"{rotulo}: {msg.get('text', '')}")
    return "\n".join(linhas), pergunta


def limitar_contextos(
    contextos: list[dict],
    max_chars_por_contexto: int | None = None,
) -> list[dict]:
    """Limita textos muito longos antes da geração, preservando IDs/scores."""
    if not max_chars_por_contexto or max_chars_por_contexto <= 0:
        return contextos
    limite = max_chars_por_contexto
    cortados: list[dict] = []
    for c in contextos:
        item = dict(c)
        texto = str(item.get("text", ""))
        if len(texto) > limite:
            item["text"] = texto[:limite].rstrip() + "\n[truncated]"
        cortados.append(item)
    return cortados


def task_a_um_turno(
    db: Chroma,
    dominio: str,
    task_id: str,
    texto_query: str,
    top_k: int = 10,
    limiar: float | None = 0.0,
    candidatos: int = 40,
    mapa_passagens: dict[str, dict] | None = None,
    expandir_passagem_completa: bool = False,
) -> dict:
    """
    Uma linha JSONL para Subtask A: task_id, Collection, contexts[{document_id, text, score, title?}].
    A consulta de recuperação usa o texto completo (histórico multi-turn no formato BEIR).

    expandir_passagem_completa: para multi-scale, preenche text/title com a passagem integral
    do corpus (mantém scores do chunk pequeno).
    """
    recuperados = recuperar_passagens_unicas(
        db,
        texto_query,
        top_k=top_k,
        candidatos=candidatos,
        limiar=limiar,
        mapa_passagem_completa=mapa_passagens,
        expandir_texto_passagem=bool(expandir_passagem_completa and mapa_passagens),
    )
    ctx = contexts_para_jsonl(recuperados)
    return {
        "task_id": task_id,
        "Collection": COLLECTION_NAME[dominio],
        "contexts": ctx,
    }


def contexts_ouro(
    task_id: str,
    qrels_por_query: dict[str, list[str]],
    mapa_passagens: dict[str, dict],
    max_contexts: int = 10,
    max_context_chars: int | None = None,
) -> list[dict]:
    ids = qrels_por_query.get(task_id, [])[:max_contexts]
    ctx: list[dict] = []
    for pid in ids:
        info = mapa_passagens.get(pid)
        if not info:
            continue
        item: dict = {"document_id": pid, "text": info["text"], "score": 1.0}
        if info.get("title"):
            item["title"] = info["title"]
        ctx.append(item)
    return limitar_contextos(ctx, max_context_chars)


def task_b_um_turno(
    task_id: str,
    texto_query: str,
    qrels_por_query: dict[str, list[str]],
    mapa_passagens: dict[str, dict],
    max_new_tokens: int = 512,
    max_contexts: int = 10,
    max_context_chars: int | None = None,
) -> dict:
    mensagens = texto_para_mensagens(texto_query)
    historico, pergunta = historico_e_pergunta_atual(mensagens)
    ctx = contexts_ouro(
        task_id,
        qrels_por_query,
        mapa_passagens,
        max_contexts=max_contexts,
        max_context_chars=max_context_chars,
    )
    prompt = montar_prompt_rag(historico, pergunta, ctx)
    from local_llm import gerar_texto

    resposta = gerar_texto(prompt, max_new_tokens=max_new_tokens)
    return {
        "task_id": task_id,
        "input": mensagens,
        "contexts": ctx,
        "predictions": [{"text": resposta}],
    }


def task_b_reference_um_turno(
    registro: dict,
    max_new_tokens: int = 512,
    max_context_chars: int | None = None,
) -> dict:
    """Subtask B oficial: geração usando os contextos de reference.jsonl."""
    mensagens = list(registro.get("input") or [])
    ctx = limitar_contextos(list(registro.get("contexts") or []), max_context_chars)
    historico, pergunta = historico_e_pergunta_de_input(mensagens)
    prompt = montar_prompt_rag(historico, pergunta, ctx)
    from local_llm import gerar_texto

    resposta = gerar_texto(prompt, max_new_tokens=max_new_tokens)
    saida = {
        "task_id": registro["task_id"],
        "input": mensagens,
        "contexts": ctx,
        "predictions": [{"text": resposta}],
    }
    if "targets" in registro:
        saida["targets"] = registro["targets"]
    if "Collection" in registro:
        saida["Collection"] = registro["Collection"]
    return saida


def task_c_um_turno(
    db: Chroma,
    task_id: str,
    texto_query: str,
    dominio: str,
    top_k: int = 5,
    limiar: float | None = 0.0,
    max_new_tokens: int = 512,
    candidatos: int = 40,
    mapa_passagens: dict[str, dict] | None = None,
    expandir_passagem_completa: bool = False,
    max_context_chars: int | None = None,
) -> dict:
    mensagens = texto_para_mensagens(texto_query)
    historico, pergunta = historico_e_pergunta_atual(mensagens)
    linha_a = task_a_um_turno(
        db,
        dominio,
        task_id,
        texto_query,
        top_k=top_k,
        limiar=limiar,
        candidatos=candidatos,
        mapa_passagens=mapa_passagens,
        expandir_passagem_completa=expandir_passagem_completa,
    )
    ctx = limitar_contextos(linha_a["contexts"], max_context_chars)
    prompt = montar_prompt_rag(historico, pergunta, ctx)
    from local_llm import gerar_texto

    resposta = gerar_texto(prompt, max_new_tokens=max_new_tokens)
    return {
        "task_id": task_id,
        "Collection": COLLECTION_NAME[dominio],
        "input": mensagens,
        "contexts": ctx,
        "predictions": [{"text": resposta}],
    }


def task_c_sem_rag_um_turno(
    task_id: str,
    texto_query: str,
    dominio: str,
    max_new_tokens: int = 512,
) -> dict:
    """Baseline: gerador sem documentos recuperados (contextos vazios)."""
    mensagens = texto_para_mensagens(texto_query)
    historico, pergunta = historico_e_pergunta_atual(mensagens)
    ctx: list[dict] = []
    prompt = montar_prompt_rag(historico, pergunta, ctx)
    from local_llm import gerar_texto

    resposta = gerar_texto(prompt, max_new_tokens=max_new_tokens)
    return {
        "task_id": task_id,
        "Collection": COLLECTION_NAME[dominio],
        "input": mensagens,
        "contexts": ctx,
        "predictions": [{"text": resposta}],
    }
