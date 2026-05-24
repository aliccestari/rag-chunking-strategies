#!/usr/bin/env python3
"""
Gemini-based LLM-as-a-Judge for MTRAG generation outputs.

Designed for free-tier quota:
- resumable JSONL outputs;
- configurable model;
- optional per-run cap to respect RPD limits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# macOS may lack default CA bundle; bypass SSL verification for the Gemini API.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


DOMAINS = ("fiqa", "clapnq", "cloud", "govt")
TASK_C_STRATEGIES = ("legacy", "small", "large", "multiscale", "noretrieval")
DEFAULT_BASE = Path("tcc_implementation/results/llama/llama3_1_8b_q4_ctx8192")
DEFAULT_MODEL = "gemini-3.1-flash-lite"


def load_dotenv_if_present(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_api_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    raise SystemExit("Set GEMINI_API_KEY, GOOGLE_API_KEY, or GOOGLE_GENAI_API_KEY in .env/environment.")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def done_task_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(row.get("task_id")) for row in iter_jsonl(path) if row.get("task_id")}


def conversation_text(turns: Any, max_chars: int = 4000) -> str:
    if not isinstance(turns, list):
        return ""
    text = "\n".join(
        f"{turn.get('speaker', 'unknown')}: {turn.get('text', '')}"
        for turn in turns
        if isinstance(turn, dict)
    )
    return text[-max_chars:]


def first_prediction(row: dict[str, Any]) -> str:
    preds = row.get("predictions") or []
    if isinstance(preds, list) and preds and isinstance(preds[0], dict):
        return str(preds[0].get("text", ""))
    return ""


def first_target(row: dict[str, Any]) -> str:
    targets = row.get("targets") or []
    if isinstance(targets, list) and targets and isinstance(targets[0], dict):
        return str(targets[0].get("text", ""))
    return ""


def contexts_text(row: dict[str, Any], max_contexts: int, max_context_chars: int) -> str:
    contexts = row.get("contexts") or []
    if not isinstance(contexts, list):
        return ""
    parts: list[str] = []
    for i, ctx in enumerate(contexts[:max_contexts], start=1):
        if not isinstance(ctx, dict):
            continue
        text = str(ctx.get("text", ""))
        if len(text) > max_context_chars:
            text = text[:max_context_chars] + "..."
        title = ctx.get("title") or ""
        doc_id = ctx.get("document_id") or ""
        header = f"[Context {i}]"
        if title or doc_id:
            header += f" title={title!r} document_id={doc_id!r}"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def build_prompt(
    row: dict[str, Any],
    *,
    domain: str,
    strategy: str,
    task: str,
    max_contexts: int,
    max_context_chars: int,
) -> str:
    return f"""You are an impartial evaluator for a Retrieval-Augmented Generation (RAG) benchmark.

Evaluate the assistant answer for the current multi-turn dialogue.
Use only the provided conversation, retrieved/reference contexts, and reference answer.

Return ONLY a valid JSON object with these exact keys:
{{
  "faithfulness": integer from 1 to 5,
  "answer_relevance": integer from 1 to 5,
  "context_usefulness": integer from 1 to 5,
  "completeness": integer from 1 to 5,
  "hallucination": "yes" | "no" | "partial",
  "justification": "one or two concise sentences"
}}

Scoring guidance:
- faithfulness: 5 means all factual claims are supported by the contexts/conversation; 1 means mostly unsupported.
- answer_relevance: 5 means the answer directly addresses the current user intent; 1 means irrelevant.
- context_usefulness: 5 means the contexts are sufficient/useful for the answer; 1 means not useful.
- completeness: 5 means the answer covers the important points in the reference answer; 1 means mostly incomplete.
- hallucination: "yes" if the answer invents unsupported facts, "partial" if some unsupported facts appear, otherwise "no".

Metadata:
task: {task}
domain: {domain}
strategy: {strategy}
task_id: {row.get("task_id", "")}

Conversation:
{conversation_text(row.get("input"))}

Reference answer:
{first_target(row)}

Assistant answer:
{first_prediction(row)}

Retrieved/reference contexts:
{contexts_text(row, max_contexts=max_contexts, max_context_chars=max_context_chars)}
"""


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = re.sub(
                r"(?m)([\{,\s])"
                r"(faithfulness|answer_relevance|context_usefulness|completeness|hallucination|justification)\s*:",
                r'\1"\2":',
                candidate,
            )
            repaired = re.sub(r":\s*'(.*?)'", r': "\1"', repaired)
            repaired = re.sub(r":\s*(yes|no|partial)([\s,\}])", r': "\1"\2', repaired)
            return json.loads(repaired)


def norm_score(value: Any) -> int | None:
    try:
        return max(1, min(5, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


def normalize_judgment(raw: dict[str, Any]) -> dict[str, Any]:
    hallucination = str(raw.get("hallucination", "")).strip().lower()
    if hallucination not in {"yes", "no", "partial"}:
        hallucination = "partial"
    return {
        "faithfulness": norm_score(raw.get("faithfulness")),
        "answer_relevance": norm_score(raw.get("answer_relevance")),
        "context_usefulness": norm_score(raw.get("context_usefulness")),
        "completeness": norm_score(raw.get("completeness")),
        "hallucination": hallucination,
        "justification": str(raw.get("justification", "")).strip(),
    }


def call_gemini(
    *,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
    retries: int,
) -> tuple[dict[str, Any], str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "faithfulness": {"type": "INTEGER"},
                    "answer_relevance": {"type": "INTEGER"},
                    "context_usefulness": {"type": "INTEGER"},
                    "completeness": {"type": "INTEGER"},
                    "hallucination": {"type": "STRING", "enum": ["yes", "no", "partial"]},
                    "justification": {"type": "STRING"},
                },
                "required": [
                    "faithfulness",
                    "answer_relevance",
                    "context_usefulness",
                    "completeness",
                    "hallucination",
                    "justification",
                ],
            },
        },
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return normalize_judgment(extract_json_object(text)), text
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {body[:500]}")
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = exc
        sleep_s = min(90, (2**attempt) * 5) + random.random()
        print(f"Gemini call failed ({last_error}); retrying in {sleep_s:.1f}s", file=sys.stderr)
        time.sleep(sleep_s)
    assert last_error is not None
    raise last_error


def input_files_for_task(base: Path, task: str, domains: list[str], strategies: list[str]) -> list[tuple[str, str, Path]]:
    files: list[tuple[str, str, Path]] = []
    if task == "c":
        for domain in domains:
            for strategy in strategies:
                files.append((domain, strategy, base / "evaluations" / "task_c" / domain / f"eval_c_{domain}_{strategy}.algonly.jsonl"))
    elif task == "b":
        for domain in domains:
            files.append((domain, "reference", base / "evaluations" / "task_b" / f"eval_b_{domain}.algonly.jsonl"))
    return files


def output_path_for(base: Path, task: str, domain: str, strategy: str, model: str) -> Path:
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    if task == "c":
        return base / "evaluations" / "gemini_judge" / model_slug / "task_c" / domain / f"gemini_judge_c_{domain}_{strategy}.jsonl"
    return base / "evaluations" / "gemini_judge" / model_slug / "task_b" / f"gemini_judge_b_{domain}.jsonl"


def mean_score(rows: list[dict[str, Any]], key: str) -> float:
    vals = [r.get(key) for r in rows if isinstance(r.get(key), int | float)]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def rate_value(rows: list[dict[str, Any]], key: str, value: str) -> float:
    return round(sum(1 for r in rows if r.get(key) == value) / len(rows), 4) if rows else 0.0


def summarize_outputs(base: Path, model: str) -> None:
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    root = base / "evaluations" / "gemini_judge" / model_slug
    if not root.exists():
        return
    summary_rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("gemini_judge_*.jsonl")):
        rows = list(iter_jsonl(path))
        if not rows:
            continue
        task = str(rows[0].get("task", ""))
        domain = str(rows[0].get("domain", ""))
        strategy = str(rows[0].get("strategy", ""))
        judged = [r.get("gemini_judge", {}) for r in rows]
        summary_rows.append(
            {
                "task": task,
                "domain": domain,
                "strategy": strategy,
                "n": len(judged),
                "faithfulness_mean": mean_score(judged, "faithfulness"),
                "answer_relevance_mean": mean_score(judged, "answer_relevance"),
                "context_usefulness_mean": mean_score(judged, "context_usefulness"),
                "completeness_mean": mean_score(judged, "completeness"),
                "hallucination_yes_rate": rate_value(judged, "hallucination", "yes"),
                "hallucination_partial_rate": rate_value(judged, "hallucination", "partial"),
            }
        )
    if not summary_rows:
        return
    out = root / "summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary written to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini LLM-as-a-Judge over MTRAG outputs.")
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", nargs="+", default=["c"], choices=["b", "c"])
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    parser.add_argument("--strategies", nargs="+", default=list(TASK_C_STRATEGIES), choices=list(TASK_C_STRATEGIES))
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--max-contexts", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit-per-file", type=int, default=None)
    parser.add_argument("--max-new-judgments", type=int, default=None, help="Stop after writing this many new judgments.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv_if_present()
    api_key = get_api_key()
    base = Path(args.base)
    total_new = 0

    for task in args.tasks:
        for domain, strategy, input_path in input_files_for_task(base, task, args.domains, args.strategies):
            if not input_path.is_file():
                print(f"Skipping missing file: {input_path}", file=sys.stderr)
                continue
            output_path = output_path_for(base, task, domain, strategy, args.model)
            done_ids = done_task_ids(output_path)
            rows = list(iter_jsonl(input_path))
            if args.limit_per_file is not None:
                rows = rows[: args.limit_per_file]
            remaining = [row for row in rows if str(row.get("task_id")) not in done_ids]
            print(
                f"=== Gemini judge model={args.model} task={task} domain={domain} strategy={strategy} "
                f"({len(done_ids)} done, {len(remaining)} remaining, {len(rows)} total considered) ===",
                flush=True,
            )
            for idx, row in enumerate(remaining, start=1):
                if args.max_new_judgments is not None and total_new >= args.max_new_judgments:
                    summarize_outputs(base, args.model)
                    print(f"Reached --max-new-judgments={args.max_new_judgments}")
                    return
                prompt = build_prompt(
                    row,
                    domain=domain,
                    strategy=strategy,
                    task=task,
                    max_contexts=args.max_contexts,
                    max_context_chars=args.max_context_chars,
                )
                t0 = time.perf_counter()
                judgment, raw_response = call_gemini(
                    api_key=api_key,
                    model=args.model,
                    prompt=prompt,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                elapsed = time.perf_counter() - t0
                append_jsonl(
                    output_path,
                    {
                        "task_id": row.get("task_id"),
                        "conversation_id": row.get("conversation_id"),
                        "Collection": row.get("Collection"),
                        "task": task,
                        "domain": domain,
                        "strategy": strategy,
                        "model": args.model,
                        "gemini_judge": judgment,
                        "raw_response": raw_response,
                        "elapsed_sec": round(elapsed, 3),
                    },
                )
                total_new += 1
                print(
                    f"  {idx}/{len(remaining)} {row.get('task_id')} "
                    f"faithfulness={judgment.get('faithfulness')} "
                    f"relevance={judgment.get('answer_relevance')} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
                time.sleep(args.sleep_seconds)

    summarize_outputs(base, args.model)
    print(f"Done. New judgments written: {total_new}")


if __name__ == "__main__":
    main()
