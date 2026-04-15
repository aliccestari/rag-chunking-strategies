"""
Domínio MTRAG atual, caminhos no repo (passage_level, retrieval_tasks) e nome de coleção oficial.

Altere DOMINIO_ATUAL e rode criar_db.py de novo; o índice fica em db_local_bge_<domínio>.
"""

from __future__ import annotations

from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parent.parent
CORPUS_PASSAGE_DIR = RAIZ_REPO / "semeval" / "corpora" / "passage_level"
RETRIEVAL_DIR = RAIZ_REPO / "semeval" / "mtrag-human" / "retrieval_tasks"

DOMINIO_ATUAL: str = "govt"

CORPUS_PASSAGE_FILES: dict[str, str] = {
    "govt": "govt.jsonl",
    "fiqa": "fiqa.jsonl",
    "cloud": "cloud.jsonl",
    "clapnq": "clapnq.jsonl",
}

COLLECTION_NAME: dict[str, str] = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "govt": "mt-rag-govt-elser-512-100-20240611",
    "fiqa": "mt-rag-fiqa-beir-elser-512-100-20240501",
    "cloud": "mt-rag-ibmcloud-elser-512-100-20240502",
}


def pasta_indice_chroma() -> str:
    return f"db_local_bge_{DOMINIO_ATUAL}"


def _ficheiro_passage(dominio: str) -> str:
    if dominio not in CORPUS_PASSAGE_FILES:
        opcoes = ", ".join(sorted(CORPUS_PASSAGE_FILES))
        raise ValueError(f"Domínio {dominio!r} inválido. Use: {opcoes}")
    return CORPUS_PASSAGE_FILES[dominio]


def caminho_corpus_passage(dominio: str) -> Path:
    p = CORPUS_PASSAGE_DIR / _ficheiro_passage(dominio)
    if not p.is_file():
        raise FileNotFoundError(
            f"Corpus não encontrado: {p}\nDescompacte o .zip em semeval/corpora/passage_level/."
        )
    return p


def caminho_corpus_passage_level() -> Path:
    """JSONL passage_level para DOMINIO_ATUAL (usado por criar_db)."""
    return caminho_corpus_passage(DOMINIO_ATUAL)


def caminho_queries_jsonl(dominio: str, variante: str) -> Path:
    """variante: lastturn | rewrite | questions"""
    p = RETRIEVAL_DIR / dominio / f"{dominio}_{variante}.jsonl"
    if not p.is_file():
        raise FileNotFoundError(f"Queries não encontradas: {p}")
    return p


def caminho_qrels_dev(dominio: str) -> Path:
    p = RETRIEVAL_DIR / dominio / "qrels" / "dev.tsv"
    if not p.is_file():
        raise FileNotFoundError(f"Qrels não encontrados: {p}")
    return p
