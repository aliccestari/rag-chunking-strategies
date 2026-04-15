#!/usr/bin/env python3
"""
CLI para Subtasks A, B e C (corpus passage_level + índice Chroma local).

Exemplos (na pasta tcc_implementation, com DOMINIO_ATUAL alinhado ao índice):

  # A — predições retrieval (JSONL)
  python run_mtrag.py task-a --domain govt --queries lastturn --index-dir db_local_bge_govt -o preds_govt_a.jsonl

  # Métricas A (pytrec)
  python run_mtrag.py eval-a --domain govt --predictions preds_govt_a.jsonl

  # B — geração só com passagens ouro (sem índice)
  python run_mtrag.py task-b --domain govt --queries lastturn -o preds_govt_b.jsonl

  # C — RAG completo (recuperação + geração local)
  python run_mtrag.py task-c --domain govt --queries lastturn --index-dir db_local_bge_govt -o preds_govt_c.jsonl

Limite de linhas (--limit) útil para smoke test.

Avaliação oficial (formato SemEval):
  python run_mtrag.py format-check --task c --domain govt --predictions preds_govt_c.jsonl
  python run_mtrag.py gen-eval --predictions preds_govt_b.jsonl -o eval_b.jsonl --provider hf \\
 --judge-model ibm-granite/granite-3.3-8b-instruct

gen-eval precisa das dependências em semeval/scripts/evaluation/requirements.txt (ideal: venv à parte).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from corpus_config import (
    RAIZ_REPO,
    caminho_corpus_passage,
    caminho_queries_jsonl,
    caminho_qrels_dev,
    pasta_indice_chroma,
)
from mtrag_subtasks import (
    carregar_mapa_passagens,
    carregar_qrels_por_query,
    indice_para_dominio,
    iter_linhas_queries,
    task_a_um_turno,
    task_b_um_turno,
    task_c_um_turno,
)
from retrieval_metrics import eval_predictions_file


def _contar_linhas_jsonl_nao_vazias(caminho: Path) -> int:
    with caminho.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _duracao_humana(segundos: float) -> str:
    if segundos < 0:
        return "0s"
    s = int(round(segundos))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _total_jobs(contagem_ficheiro: int, limit: int | None) -> int:
    if limit is None:
        return contagem_ficheiro
    return min(contagem_ficheiro, max(0, limit))


def _imprimir_progresso(etiqueta: str, feitos: int, total: int, t0_loop: float) -> None:
    decorrido = time.perf_counter() - t0_loop
    restam = total - feitos
    if total <= 0:
        return
    if feitos >= 1 and restam > 0:
        eta = (decorrido / feitos) * restam
        extra = f", ~{_duracao_humana(eta)} restantes (estim.)"
    elif restam == 0:
        extra = ""
    else:
        extra = ""
    print(
        f"  {etiqueta}: {feitos}/{total} — {_duracao_humana(decorrido)} decorrido{extra}",
        flush=True,
    )


def _abrir_indice_arg(args: argparse.Namespace) -> Path:
    pasta = Path(args.index_dir) if args.index_dir else Path(pasta_indice_chroma())
    if not pasta.is_dir():
        sys.stderr.write(f"Índice não encontrado: {pasta.resolve()}\n")
        sys.exit(1)
    return pasta


def cmd_task_a(args: argparse.Namespace) -> None:
    t0_all = time.perf_counter()
    pasta = _abrir_indice_arg(args)
    db = indice_para_dominio(pasta)
    caminho_q = caminho_queries_jsonl(args.domain, args.queries)
    total = _total_jobs(_contar_linhas_jsonl_nao_vazias(caminho_q), args.limit)
    out = Path(args.output)
    n = 0
    t0_loop = time.perf_counter()
    with out.open("w", encoding="utf-8") as sink:
        for task_id, texto in iter_linhas_queries(caminho_q):
            if args.limit is not None and n >= args.limit:
                break
            reg = task_a_um_turno(
                db,
                args.domain,
                task_id,
                texto,
                top_k=args.top_k,
                limiar=args.limiar,
                candidatos=args.candidatos,
            )
            sink.write(json.dumps(reg, ensure_ascii=False) + "\n")
            n += 1
            _imprimir_progresso("Task A", n, total, t0_loop)
    dt_loop = time.perf_counter() - t0_loop
    dt_all = time.perf_counter() - t0_all
    print(
        f"Task A: {n} linhas -> {out} (loop {_duracao_humana(dt_loop)}, total c/ setup {_duracao_humana(dt_all)})",
        flush=True,
    )


def cmd_task_b(args: argparse.Namespace) -> None:
    t0_all = time.perf_counter()
    caminho_q = caminho_queries_jsonl(args.domain, args.queries)
    total = _total_jobs(_contar_linhas_jsonl_nao_vazias(caminho_q), args.limit)
    qrels = carregar_qrels_por_query(caminho_qrels_dev(args.domain))
    corpus = caminho_corpus_passage(args.domain)
    print("Task B: a ler corpus passage_level para RAM (pode demorar um pouco)…", flush=True)
    t0_mapa = time.perf_counter()
    mapa = carregar_mapa_passagens(corpus)
    dt_mapa = time.perf_counter() - t0_mapa
    print(
        f"Task B: {len(mapa)} passagens em memória ({_duracao_humana(dt_mapa)}). A gerar respostas…",
        flush=True,
    )
    out = Path(args.output)
    n = 0
    t0_loop = time.perf_counter()
    with out.open("w", encoding="utf-8") as sink:
        for task_id, texto in iter_linhas_queries(caminho_q):
            if args.limit is not None and n >= args.limit:
                break
            reg = task_b_um_turno(
                task_id,
                texto,
                qrels,
                mapa,
                max_new_tokens=args.max_new_tokens,
                max_contexts=args.max_contexts,
            )
            sink.write(json.dumps(reg, ensure_ascii=False) + "\n")
            n += 1
            _imprimir_progresso("Task B", n, total, t0_loop)
    dt_loop = time.perf_counter() - t0_loop
    dt_all = time.perf_counter() - t0_all
    print(
        f"Task B: {n} linhas -> {out} (loop {_duracao_humana(dt_loop)}, total c/ leitura corpus {_duracao_humana(dt_all)})",
        flush=True,
    )


def cmd_task_c(args: argparse.Namespace) -> None:
    t0_all = time.perf_counter()
    pasta = _abrir_indice_arg(args)
    db = indice_para_dominio(pasta)
    caminho_q = caminho_queries_jsonl(args.domain, args.queries)
    total = _total_jobs(_contar_linhas_jsonl_nao_vazias(caminho_q), args.limit)
    out = Path(args.output)
    n = 0
    t0_loop = time.perf_counter()
    with out.open("w", encoding="utf-8") as sink:
        for task_id, texto in iter_linhas_queries(caminho_q):
            if args.limit is not None and n >= args.limit:
                break
            reg = task_c_um_turno(
                db,
                task_id,
                texto,
                args.domain,
                top_k=args.top_k,
                limiar=args.limiar,
                max_new_tokens=args.max_new_tokens,
                candidatos=args.candidatos,
            )
            sink.write(json.dumps(reg, ensure_ascii=False) + "\n")
            n += 1
            _imprimir_progresso("Task C", n, total, t0_loop)
    dt_loop = time.perf_counter() - t0_loop
    dt_all = time.perf_counter() - t0_all
    print(
        f"Task C: {n} linhas -> {out} (loop {_duracao_humana(dt_loop)}, total c/ setup índice {_duracao_humana(dt_all)})",
        flush=True,
    )


def cmd_eval_a(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    res = eval_predictions_file(Path(args.predictions), args.domain)
    print(json.dumps({k: v for k, v in res.items() if k != "per_query"}, indent=2))
    print(f"(eval-a: {_duracao_humana(time.perf_counter() - t0)})", flush=True)


_FORMAT_CHECK_MODE = {"a": "retrieval_taska", "b": "generation_taskb", "c": "rag_taskc"}


def cmd_format_check(args: argparse.Namespace) -> None:
    """Chama semeval/scripts/evaluation/format_checker.py com stub de task_id (queries BEIR usam _id)."""
    from mtrag_subtasks import iter_linhas_queries

    script = RAIZ_REPO / "semeval/scripts/evaluation/format_checker.py"
    if not script.is_file():
        sys.stderr.write(f"Script não encontrado: {script}\n")
        sys.exit(1)

    qpath = caminho_queries_jsonl(args.domain, args.queries)
    mode = _FORMAT_CHECK_MODE[args.task]
    pred = Path(args.predictions).resolve()

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        stub_path = Path(tmp.name)
        n = 0
        for tid, _ in iter_linhas_queries(qpath):
            if args.limit is not None and n >= args.limit:
                break
            tmp.write(json.dumps({"task_id": tid}) + "\n")
            n += 1

    try:
        r = subprocess.run(
            [
                sys.executable,
                str(script),
                "--input_file",
                str(stub_path),
                "--prediction_file",
                str(pred),
                "--mode",
                mode,
            ],
            cwd=str(RAIZ_REPO),
        )
        if r.returncode != 0:
            sys.exit(r.returncode)
    finally:
        stub_path.unlink(missing_ok=True)


def _carregar_referencia_por_task_id(caminho: Path) -> dict[str, dict]:
    ref: dict[str, dict] = {}
    with caminho.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            ref[str(o["task_id"])] = o
    return ref


def merge_predictions_com_reference(
    pred_jsonl: Path,
    ref_jsonl: Path,
    out_jsonl: Path,
) -> None:
    """Acrescenta `targets` (e `conversation_id` se faltar) a partir de reference.jsonl oficial."""
    ref = _carregar_referencia_por_task_id(ref_jsonl)
    with pred_jsonl.open(encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            pred = json.loads(line)
            tid = str(pred["task_id"])
            if tid not in ref:
                raise KeyError(f"task_id {tid!r} não está em {ref_jsonl}")
            src = ref[tid]
            if "targets" not in src:
                raise KeyError(f"Linha de referência sem 'targets' para {tid!r}")
            pred["targets"] = src["targets"]
            if "conversation_id" not in pred and "conversation_id" in src:
                pred["conversation_id"] = src["conversation_id"]
            fout.write(json.dumps(pred, ensure_ascii=False) + "\n")


def cmd_merge_gen_eval(args: argparse.Namespace) -> None:
    """Junta preds (B/C) com generation_tasks/reference.jsonl para o run_generation_eval."""
    merge_predictions_com_reference(
        Path(args.predictions),
        Path(args.reference),
        Path(args.output),
    )
    print(f"Gravado: {args.output} (com targets para métricas ROUGE/BERTScore)", flush=True)


def cmd_gen_eval(args: argparse.Namespace) -> None:
    """Roda semeval/scripts/evaluation/run_generation_eval.py (RAGAS + juízes; pesado)."""
    script = RAIZ_REPO / "semeval/scripts/evaluation/run_generation_eval.py"
    cfg = RAIZ_REPO / "semeval/scripts/evaluation/config.yaml"
    if not script.is_file():
        sys.stderr.write(f"Script não encontrado: {script}\n")
        sys.exit(1)
    if not cfg.is_file():
        sys.stderr.write(f"Config não encontrada: {cfg}\n")
        sys.exit(1)

    pred = Path(args.predictions).resolve()
    merged_tmp: Path | None = None
    if args.reference:
        refp = Path(args.reference)
        if not refp.is_file():
            sys.stderr.write(f"Referência não encontrada: {refp}\n")
            sys.exit(1)
        fd, tmp_name = tempfile.mkstemp(suffix=".jsonl", prefix="mtrag_merged_")
        os.close(fd)
        tmp = Path(tmp_name)
        merge_predictions_com_reference(pred, refp, tmp)
        pred = tmp
        merged_tmp = tmp
    else:
        with pred.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    if "targets" not in json.loads(line):
                        sys.stderr.write(
                            "As tuas predições não têm 'targets' (resposta ouro).\n"
                            "O run_generation_eval precisa disso para ROUGE/BERTScore.\n"
                            "Obtenha `mtrag-human/generation_tasks/reference.jsonl` no dataset e rode:\n"
                            "  python run_mtrag.py merge-gen-eval -p preds.jsonl -r reference.jsonl -o preds_merged.jsonl\n"
                            "  python run_mtrag.py gen-eval -p preds_merged.jsonl -o eval.jsonl --provider hf --judge-model ...\n"
                            "Ou use gen-eval --reference reference.jsonl (merge temporário).\n"
                        )
                        sys.exit(1)
                    break

    out = Path(args.output).resolve()
    cmd: list[str] = [
        sys.executable,
        str(script),
        "-i",
        str(pred),
        "-o",
        str(out),
        "-e",
        str(cfg),
        "--provider",
        args.provider,
    ]
    if args.provider == "openai":
        if not args.openai_key or not args.azure_host:
            sys.stderr.write("gen-eval: --provider openai requer --openai-key e --azure-host\n")
            sys.exit(1)
        cmd += ["--openai_key", args.openai_key, "--azure_host", args.azure_host]
    else:
        if not args.judge_model:
            sys.stderr.write("gen-eval: use --judge-model (ex.: ibm-granite/granite-3.3-8b-instruct)\n")
            sys.exit(1)
        cmd += ["--judge_model", args.judge_model]

    print("Avaliação de geração (pode demorar horas e exige RAM/GPU para o juiz HF)...", flush=True)
    print(" ".join(cmd), flush=True)
    try:
        r = subprocess.run(cmd, cwd=str(RAIZ_REPO))
        if r.returncode != 0:
            sys.stderr.write(
                "\nSe faltar dependência (ragas, flash_attn, etc.), crie um venv só para avaliação:\n"
                f"  cd {RAIZ_REPO} && pip install -r semeval/scripts/evaluation/requirements.txt\n"
                "(em conflito com o tcc_implementation, use outro ambiente virtual).\n"
            )
            sys.exit(r.returncode)
        print(f"Métricas gravadas em: {out}", flush=True)
    finally:
        if merged_tmp is not None:
            merged_tmp.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description="MTRAG Subtasks A/B/C (local)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("task-a", help="Subtask A: só retrieval")
    pa.add_argument("--domain", required=True, choices=["govt", "fiqa", "cloud", "clapnq"])
    pa.add_argument("--queries", default="lastturn", choices=["lastturn", "rewrite", "questions"])
    pa.add_argument("--index-dir", default=None, help="Pasta Chroma (default: corpus_config.pasta_indice_chroma())")
    pa.add_argument("-o", "--output", required=True)
    pa.add_argument("--top-k", type=int, default=10)
    pa.add_argument("--candidatos", type=int, default=40)
    pa.add_argument("--limiar", type=float, default=0.0, help="Descarta scores abaixo disto (0 = sem corte)")
    pa.add_argument("--limit", type=int, default=None)
    pa.set_defaults(func=cmd_task_a)

    pb = sub.add_parser("task-b", help="Subtask B: geração com passagens ouro dos qrels")
    pb.add_argument("--domain", required=True, choices=["govt", "fiqa", "cloud", "clapnq"])
    pb.add_argument("--queries", default="lastturn", choices=["lastturn", "rewrite", "questions"])
    pb.add_argument("-o", "--output", required=True)
    pb.add_argument("--max-new-tokens", type=int, default=512)
    pb.add_argument("--max-contexts", type=int, default=10)
    pb.add_argument("--limit", type=int, default=None)
    pb.set_defaults(func=cmd_task_b)

    pc = sub.add_parser("task-c", help="Subtask C: retrieval + geração")
    pc.add_argument("--domain", required=True, choices=["govt", "fiqa", "cloud", "clapnq"])
    pc.add_argument("--queries", default="lastturn", choices=["lastturn", "rewrite", "questions"])
    pc.add_argument("--index-dir", default=None)
    pc.add_argument("-o", "--output", required=True)
    pc.add_argument("--top-k", type=int, default=5)
    pc.add_argument("--candidatos", type=int, default=40)
    pc.add_argument("--limiar", type=float, default=0.0)
    pc.add_argument("--max-new-tokens", type=int, default=512)
    pc.add_argument("--limit", type=int, default=None)
    pc.set_defaults(func=cmd_task_c)

    pe = sub.add_parser("eval-a", help="Recall@k / nDCG@k para arquivo Task A (k=1,3,5,10)")
    pe.add_argument("--domain", required=True, choices=["govt", "fiqa", "cloud", "clapnq"])
    pe.add_argument("--predictions", required=True)
    pe.set_defaults(func=cmd_eval_a)

    pf = sub.add_parser(
        "format-check",
        help="Valida JSONL com format_checker.py do SemEval (stub de task_id a partir das queries)",
    )
    pf.add_argument("--task", required=True, choices=["a", "b", "c"])
    pf.add_argument("--domain", required=True, choices=["govt", "fiqa", "cloud", "clapnq"])
    pf.add_argument("--queries", default="lastturn", choices=["lastturn", "rewrite", "questions"])
    pf.add_argument("--predictions", required=True, help="preds_govt_a.jsonl, _b ou _c")
    pf.add_argument("--limit", type=int, default=None)
    pf.set_defaults(func=cmd_format_check)

    pm = sub.add_parser(
        "merge-gen-eval",
        help="Junta preds B/C com reference.jsonl (campo targets) para run_generation_eval",
    )
    pm.add_argument("-p", "--predictions", required=True)
    pm.add_argument("-r", "--reference", required=True, help="generation_tasks/reference.jsonl")
    pm.add_argument("-o", "--output", required=True)
    pm.set_defaults(func=cmd_merge_gen_eval)

    pg = sub.add_parser(
        "gen-eval",
        help="Roda avaliação de geração oficial (Tasks B/C): RAGAS + juízes",
    )
    pg.add_argument("--predictions", required=True, help="preds com 'targets' OU use --reference")
    pg.add_argument(
        "--reference",
        default=None,
        help="reference.jsonl: faz merge temporário antes da avaliação",
    )
    pg.add_argument("-o", "--output", required=True, help="JSONL de saída com métricas")
    pg.add_argument(
        "--provider",
        required=True,
        choices=["openai", "hf", "vllm"],
        help="hf = juiz HuggingFace local; vllm = servidor em localhost:8001",
    )
    pg.add_argument(
        "--judge-model",
        dest="judge_model",
        default=None,
        help="Obrigatório se provider é hf ou vllm (ex.: ibm-granite/granite-3.3-8b-instruct)",
    )
    pg.add_argument("--openai-key", dest="openai_key", default=None)
    pg.add_argument("--azure-host", dest="azure_host", default=None)
    pg.set_defaults(func=cmd_gen_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
