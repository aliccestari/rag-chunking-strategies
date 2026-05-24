"""
Recuperação via Chroma + BGE (mesmo índice que `criar_db.py`), com deduplicação por id de passagem.

Modo multi-scale: depois de ranquear por chunk pequeno, pode substituir `text` (e `title`)
pelo conteúdo completo da passagem a partir do mapa do corpus.
"""

from __future__ import annotations

from pathlib import Path

from langchain_chroma.vectorstores import Chroma

from embeddings_config import criar_funcao_embedding


def abrir_indice(persist_directory: str | Path) -> Chroma:
    pasta = Path(persist_directory)
    if not pasta.is_dir():
        raise FileNotFoundError(
            f"Índice Chroma não encontrado: {pasta.resolve()}\n"
            "Rode criar_db.py com o mesmo domínio/estratégia (veja corpus_config / --strategy)."
        )
    emb = criar_funcao_embedding()
    return Chroma(persist_directory=str(pasta), embedding_function=emb)


def recuperar_passagens_unicas(
    db: Chroma,
    consulta: str,
    top_k: int = 10,
    candidatos: int = 30,
    limiar: float | None = None,
    mapa_passagem_completa: dict[str, dict] | None = None,
    expandir_texto_passagem: bool = False,
) -> list[tuple[str, float, str, str]]:
    """
    Retorna lista de (passage_id, score, texto, titulo) ordenada por score decrescente.

    Se expandir_texto_passagem e mapa_passagem_completa: após escolher o melhor score
    por passage_id, substitui texto/título pelo do JSONL (estratégia multi-scale).
    """
    k_busca = max(candidatos, top_k * 4)
    pares = db.similarity_search_with_relevance_scores(consulta, k=k_busca)
    melhor_por_id: dict[str, tuple[float, str, str]] = {}
    for doc, score in pares:
        if limiar is not None and score < limiar:
            continue
        pid = doc.metadata.get("id")
        if pid is None:
            continue
        pid = str(pid)
        titulo = str(doc.metadata.get("title", "") or "")
        corpo = doc.page_content
        if pid not in melhor_por_id or score > melhor_por_id[pid][0]:
            melhor_por_id[pid] = (float(score), corpo, titulo)

    ordenado = sorted(melhor_por_id.items(), key=lambda x: -x[1][0])
    saida: list[tuple[str, float, str, str]] = []
    for pid, (sc, texto, tit) in ordenado[:top_k]:
        if expandir_texto_passagem and mapa_passagem_completa:
            # IDs no índice Chroma têm formato composto "{corpus_id}-{int}-{int}"
            # (ex: "414940-0-474" ou "827849752_115-357-0-1033").
            # Usamos rsplit para extrair o corpus_id correto mesmo quando ele
            # contém hífens (ex: clapnq "827849752_115-357").
            info = mapa_passagem_completa.get(pid)
            if info is None:
                parts = pid.rsplit("-", 2)
                if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                    base_pid = parts[0]
                    info = mapa_passagem_completa.get(base_pid)
            if info:
                texto = str(info.get("text") or texto)
                tit = str(info.get("title") or tit)
        saida.append((pid, sc, texto, tit))
    return saida


def contexts_para_jsonl(
    recuperados: list[tuple[str, float, str, str]],
) -> list[dict]:
    """Formato de `contexts` esperado pelo format checker e pela avaliação."""
    ctx = []
    for pid, score, texto, titulo in recuperados:
        item: dict = {
            "document_id": pid,
            "text": texto,
            "score": score,
        }
        if titulo:
            item["title"] = titulo
        ctx.append(item)
    return ctx
