"""
Script para criar a base vetorial (Chroma) a partir de documentos.

Fluxo: carregar textos -> dividir em chunks (estratégia TCC) -> embeddings BGE -> salvar.

Estratégias (--strategy):
  legacy   — chunk 2000/500 (compatível com índices antigos `db_local_bge_<dom>`)
  small    — chunks pequenos (precisão na recuperação; base do multi-scale)
  large    — chunks longos (mais contexto por vetor)
  multiscale — alias: constrói o mesmo índice que `small` (recuperação fina + texto
             de passagem completo na geração é feito em run_mtrag com --chunking multiscale)

GOOGLE_API_KEY no .env só é usada pelo main.py para o Gemini (não para embedding).
"""

import argparse
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document

from chunking_strategies import ChunkingStrategy, dividir_documentos
from corpus_config import (
    CORPUS_PASSAGE_FILES,
    DOMINIO_ATUAL,
    caminho_corpus_passage,
    pasta_indice_chroma,
)
from embeddings_config import criar_funcao_embedding

load_dotenv()

FONTE = "jsonl"
PASTA_BASE = "base"
LIMITE_LINHAS: int | None = None
TAMANHO_LOTE_EMBEDDING = 256


def carregar_jsonl(caminho: Path, limite: int | None) -> list[Document]:
    documentos: list[Document] = []
    with caminho.open(encoding="utf-8") as arquivo:
        for indice, linha in enumerate(arquivo):
            if limite is not None and indice >= limite:
                break
            linha = linha.strip()
            if not linha:
                continue
            registro = json.loads(linha)
            passage_id = registro.get("_id") or registro.get("id")
            texto = registro.get("text", "")
            titulo = registro.get("title", "") or ""
            if titulo:
                conteudo = f"{titulo}\n{texto}"
            else:
                conteudo = texto

            documentos.append(
                Document(
                    page_content=conteudo,
                    metadata={"id": passage_id, "title": titulo},
                )
            )

    print(f"Carregadas {len(documentos)} passagens de {caminho.name}")
    return documentos


def vetorizar_chunks(chunks: list[Document], nome_pasta: str, recriar: bool) -> None:
    pasta = Path(nome_pasta)
    if recriar and pasta.exists():
        shutil.rmtree(pasta)
        print(f"Pasta antiga removida: {nome_pasta}")

    embeddings = criar_funcao_embedding()
    db = Chroma(embedding_function=embeddings, persist_directory=nome_pasta)

    total = len(chunks)
    for inicio in range(0, total, TAMANHO_LOTE_EMBEDDING):
        lote = chunks[inicio : inicio + TAMANHO_LOTE_EMBEDDING]
        db.add_documents(lote)
        feito = min(inicio + TAMANHO_LOTE_EMBEDDING, total)
        print(f"Embeddings gravados: {feito}/{total}")

    print(f"Banco salvo em ./{nome_pasta}")


def criar_db(
    dominio: str,
    strategy: ChunkingStrategy,
    limite_linhas: int | None,
    recriar: bool,
) -> None:
    build_strategy: ChunkingStrategy
    if strategy == "multiscale":
        build_strategy = "small"
        print(
            "Estratégia multiscale: a construir o mesmo índice que 'small'. "
            "Na avaliação, use run_mtrag --chunking multiscale (--index-dir opcional)."
        )
    else:
        build_strategy = strategy

    nome_pasta = pasta_indice_chroma(dominio, strategy if strategy != "multiscale" else "small")
    caminho = caminho_corpus_passage(dominio)
    print(f"Domínio: {dominio} | estratégia build: {build_strategy} | pasta: ./{nome_pasta}/")

    if FONTE == "pdf":
        carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
        documentos = carregador.load()
    else:
        documentos = carregar_jsonl(caminho, limite=limite_linhas)

    chunks = dividir_documentos(documentos, build_strategy)
    print(f"Total de chunks após divisão: {len(chunks)}")
    vetorizar_chunks(chunks, nome_pasta, recriar=recriar)


def main() -> None:
    doms = sorted(CORPUS_PASSAGE_FILES.keys())
    ap = argparse.ArgumentParser(description="Cria índice Chroma BGE para MTRAG/TCC.")
    ap.add_argument(
        "--domain",
        default=DOMINIO_ATUAL,
        choices=doms,
        help="Domínio do corpus passage_level (default: DOMINIO_ATUAL em corpus_config).",
    )
    ap.add_argument(
        "--strategy",
        default="legacy",
        choices=["legacy", "small", "large", "multiscale"],
        help="Política de chunking para o índice (multiscale = mesmo índice que small).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Só as primeiras N passagens (teste rápido).",
    )
    ap.add_argument(
        "--recrear",
        action="store_true",
        help="Apaga a pasta do índice antes de gravar.",
    )
    args = ap.parse_args()
    criar_db(args.domain, args.strategy, args.limit, args.recrear)


if __name__ == "__main__":
    main()
