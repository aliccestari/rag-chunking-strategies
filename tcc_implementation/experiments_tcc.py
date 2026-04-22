#!/usr/bin/env python3
"""
Imprime comandos sugeridos para o desenho experimental do TCC (chunking + MTRAGEval).

Os quatro domínios MTRAG (passage_level) são: govt, fiqa, cloud, clapnq — cada um precisa
dos seus índices Chroma e das predições (repetir o fluxo por domínio).

Uso (na pasta tcc_implementation):
  python experiments_tcc.py --domain govt
  python experiments_tcc.py --domain all          # lista tudo para os 4 domínios
  python experiments_tcc.py --domain all --only index
"""

from __future__ import annotations

import argparse

from corpus_config import CORPUS_PASSAGE_FILES

DOMINIOS_MTRAG = sorted(CORPUS_PASSAGE_FILES.keys())


def linhas_comandos(domain: str, queries: str, only: str | None) -> list[str]:
    d = domain
    q = queries
    cmd: list[str] = []

    def add(s: str) -> None:
        cmd.append(s)

    if only in (None, "index"):
        add(f"# Índices (executar uma vez por domínio; demora)")
        add(f"python criar_db.py --domain {d} --strategy legacy --recrear")
        add(f"python criar_db.py --domain {d} --strategy small --recrear")
        add(f"python criar_db.py --domain {d} --strategy large --recrear")
        add("")

    if only in (None, "run"):
        add(f"# Task A — comparar estratégias (métricas: eval-a)")
        for ch in ("legacy", "small", "large", "multiscale"):
            add(
                f"python run_mtrag.py task-a --domain {d} --queries {q} --chunking {ch} "
                f"-o preds_{d}_a_{ch}.jsonl --timing-log timings_{d}_a_{ch}.csv"
            )
        add(f"# eval-a (repetir para cada preds_*_a_*.jsonl que quiseres tabular)")
        for ch in ("legacy", "small", "large", "multiscale"):
            add(
                f"python run_mtrag.py eval-a --domain {d} --predictions preds_{d}_a_{ch}.jsonl"
            )
        add("")

        add(f"# Task C — RAG + baseline sem retrieval")
        for ch in ("legacy", "small", "large", "multiscale"):
            add(
                f"python run_mtrag.py task-c --domain {d} --queries {q} --chunking {ch} "
                f"-o preds_{d}_c_{ch}.jsonl --timing-log timings_{d}_c_{ch}.csv"
            )
        add(
            f"python run_mtrag.py task-c --domain {d} --queries {q} --baseline noretrieval "
            f"-o preds_{d}_c_noretrieval.jsonl --timing-log timings_{d}_c_nr.csv"
        )
        add("")

        add(f"# Task B (gerador + ouro) — independente de chunking")
        add(f"python run_mtrag.py task-b --domain {d} --queries {q} -o preds_{d}_b.jsonl")
        add("")

        add("# Avaliação de geração (venv SemEval separado; merge com reference.jsonl)")
        add(
            f"# python run_mtrag.py merge-gen-eval -p preds_{d}_c_small.jsonl "
            f"-r ../semeval/mtrag-human/generation_tasks/reference.jsonl -o merged_c.jsonl"
        )
        add(
            f"# python run_mtrag.py gen-eval --predictions merged_c.jsonl -o eval_c.jsonl "
            f"--provider hf --judge-model ibm-granite/granite-3.3-8b-instruct"
        )

    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Comandos sugeridos para experimentos TCC / MTRAG.")
    ap.add_argument(
        "--domain",
        default="govt",
        choices=["all", "govt", "fiqa", "cloud", "clapnq"],
        help="all = imprime comandos para os 4 domínios (govt, fiqa, cloud, clapnq).",
    )
    ap.add_argument("--queries", default="lastturn", choices=["lastturn", "rewrite", "questions"])
    ap.add_argument("--only", choices=["index", "run"], default=None)
    args = ap.parse_args()

    dominios = DOMINIOS_MTRAG if args.domain == "all" else [args.domain]
    first = True
    for dom in dominios:
        if not first:
            print()
            print("# " + "=" * 72)
        first = False
        print(f"# Domínio: {dom} (corpus passage_level: {CORPUS_PASSAGE_FILES[dom]})")
        print()
        for line in linhas_comandos(dom, args.queries, args.only):
            print(line)


if __name__ == "__main__":
    main()
