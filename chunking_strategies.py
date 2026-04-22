"""
Estratégias de chunking para o TCC (small vs large vs multi-scale).

- small / large: um índice Chroma cada; a recuperação usa o texto do chunk.
- multi-scale: mesmo índice que small; na geração, substitui-se o texto do chunk
  pelo texto completo da passagem (parent) vindo do JSONL passage_level.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ChunkingStrategy = Literal["legacy", "small", "large", "multiscale"]

# Parâmetros calibráveis (caracteres; alinhados a splitters típicos do LangChain)
LEGACY_CHUNK_SIZE = 2000
LEGACY_CHUNK_OVERLAP = 500

SMALL_CHUNK_SIZE = 900
SMALL_CHUNK_OVERLAP = 120

LARGE_CHUNK_SIZE = 12000
LARGE_CHUNK_OVERLAP = 400


def splitter_para_estrategia(strategy: ChunkingStrategy) -> RecursiveCharacterTextSplitter:
    if strategy == "legacy":
        size, ov = LEGACY_CHUNK_SIZE, LEGACY_CHUNK_OVERLAP
    elif strategy in ("small", "multiscale"):
        # multiscale: mesmo índice que small
        size, ov = SMALL_CHUNK_SIZE, SMALL_CHUNK_OVERLAP
    elif strategy == "large":
        size, ov = LARGE_CHUNK_SIZE, LARGE_CHUNK_OVERLAP
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=ov,
        length_function=len,
        add_start_index=True,
    )


def dividir_documentos(documentos: list[Document], strategy: ChunkingStrategy) -> list[Document]:
    sep = splitter_para_estrategia(strategy)
    chunks = sep.split_documents(documentos)
    return chunks
