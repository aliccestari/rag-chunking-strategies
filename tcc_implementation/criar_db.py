"""
Script para criar a base vetorial (Chroma) a partir de documentos.

Fluxo: carregar textos -> dividir em chunks -> embeddings BGE (local) -> salvar em disco.
Execute uma vez por configuração; depois o main.py lê a mesma pasta.

GOOGLE_API_KEY no .env só é usada pelo main.py para o Gemini gerar respostas (não para embedding).
"""

import json
import shutil
from pathlib import Path
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from corpus_config import DOMINIO_ATUAL, caminho_corpus_passage_level, pasta_indice_chroma
from embeddings_config import criar_funcao_embedding

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração simples (TCC: corpus oficial por domínio — veja corpus_config.py)
# ---------------------------------------------------------------------------

# "jsonl" = passagens MTRAG (domínio em DOMINIO_ATUAL). "pdf" = modo tutorial (pasta base/).
FONTE = "jsonl"

PASTA_BASE = "base"

# None = indexa o JSONL inteiro do domínio atual. Int (ex. 500) = só primeiras N linhas (teste rápido).
LIMITE_LINHAS: int | None = None

# Quantos chunks processar por vez na RAM (ajuste se faltar memória).
TAMANHO_LOTE_EMBEDDING = 256

# Se True, apaga a pasta do índice antes de indexar (troca de corpus ou modo de embedding).
RECRIAR_INDICE_DO_ZERO = False


def criar_db():
    print(f"Domínio: {DOMINIO_ATUAL} | pasta do índice: ./{pasta_indice_chroma()}/")
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)


def carregar_documentos():
    if FONTE == "pdf":
        carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
        return carregador.load()

    if FONTE == "jsonl":
        caminho = caminho_corpus_passage_level()
        return carregar_jsonl(caminho, limite=LIMITE_LINHAS)

    raise ValueError(f"FONTE deve ser 'pdf' ou 'jsonl', recebido: {FONTE}")


def carregar_jsonl(caminho: Path, limite: int | None) -> list[Document]:
    """
    Cada linha do arquivo é um JSON com: _id (ou id), text, title, etc.

    LangChain usa o objeto Document:
    - page_content: texto que vai virar embedding;
    - metadata: dicionário extra — aqui guardamos "id" da passagem (bate com qrels).
    """
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
            titulo = registro.get("title", "")
            # Título ajuda o modelo a situar o assunto; faz parte do que será embedado.
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


def dividir_chunks(documentos: list[Document]) -> list[Document]:
    """
    Parte textos longos em pedaços menores. O LangChain copia o metadata para cada pedaço
    (o mesmo "id" de passagem permanece — útil quando um documento vira vários chunks).
    """
    separador = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = separador.split_documents(documentos)
    print(f"Total de chunks após divisão: {len(chunks)}")
    return chunks


def vetorizar_chunks(chunks: list[Document]) -> None:
    nome_pasta = pasta_indice_chroma()
    pasta = Path(nome_pasta)
    if RECRIAR_INDICE_DO_ZERO and pasta.exists():
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


if __name__ == "__main__":
    criar_db()
