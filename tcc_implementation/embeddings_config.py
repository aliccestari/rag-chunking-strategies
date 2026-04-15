"""
Embeddings locais (BGE-small), sem API do Google para vetorização.

O Gemini no main.py só gera a resposta em texto (chat), não cria embeddings.
"""

from __future__ import annotations

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

MODELO_EMBEDDING = "BAAI/bge-small-en-v1.5"


def criar_funcao_embedding():
    """Objeto que o LangChain/Chroma usa para transformar texto em vetor (BGE no Mac)."""
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Embeddings: {MODELO_EMBEDDING} | device={device}")
    return HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDING,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
