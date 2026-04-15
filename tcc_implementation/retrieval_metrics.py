"""
Métricas Subtask A (Recall@k, nDCG@k) com caminhos corretos para este repositório.

O script oficial `semeval/scripts/evaluation/run_retrieval_eval.py` aponta para
pastas `human/retrieval_tasks_convid/` que não existem aqui; usamos
`semeval/mtrag-human/retrieval_tasks/<dominio>/qrels/dev.tsv`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytrec_eval

from corpus_config import COLLECTION_NAME, caminho_qrels_dev


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            qid, doc_id, score = row[0], row[1], int(row[2])
            qrels.setdefault(qid, {})[doc_id] = score
    return qrels


def predictions_to_results(path: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = item["task_id"]
            doc_scores: dict[str, float] = {}
            for ctx in item.get("contexts", []):
                doc_scores[str(ctx["document_id"])] = float(ctx["score"])
            results[qid] = doc_scores
    return results


def evaluate_collection(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> tuple[dict[str, float], dict[str, Any]]:
    ndcg_string = "ndcg_cut." + ",".join(str(k) for k in k_values)
    recall_string = "recall." + ",".join(str(k) for k in k_values)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string, recall_string})
    scores = evaluator.evaluate(results)

    ndcg_avg = {f"NDCG@{k}": 0.0 for k in k_values}
    recall_avg = {f"Recall@{k}": 0.0 for k in k_values}
    for qid in scores:
        for k in k_values:
            ndcg_avg[f"NDCG@{k}"] += scores[qid][f"ndcg_cut_{k}"]
            recall_avg[f"Recall@{k}"] += scores[qid][f"recall_{k}"]
    n = len(scores)
    if n:
        for k in k_values:
            ndcg_avg[f"NDCG@{k}"] = round(ndcg_avg[f"NDCG@{k}"] / n, 5)
            recall_avg[f"Recall@{k}"] = round(recall_avg[f"Recall@{k}"] / n, 5)
    merged = {**ndcg_avg, **recall_avg}
    return merged, scores


def eval_predictions_file(predictions_jsonl: Path, dominio: str) -> dict[str, Any]:
    qrels = load_qrels(caminho_qrels_dev(dominio))
    preds_raw = predictions_to_results(predictions_jsonl)
    preds = {qid: preds_raw[qid] for qid in preds_raw if qid in qrels}
    # MTRAGEval Task A reporta @1,3,5,10 (https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)
    k_values = [1, 3, 5, 10]
    averages, per_query = evaluate_collection(qrels, preds, k_values)
    return {
        "dominio": dominio,
        "collection": COLLECTION_NAME[dominio],
        "num_queries_eval": len(preds),
        "averages": averages,
        "per_query": per_query,
    }
