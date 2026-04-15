#!/usr/bin/env python3
"""
Conta tokens por linha (entrada) em corpora JSONL passage-level.

Métricas por arquivo: número de entradas, total de tokens, média, mediana,
desvio padrão, mínimo e máximo.

Por padrão usa tiktoken (cl100k_base), comum em relatórios de custo/contexto LLM.
Use --hf-tokenizer para alinhar ao modelo de embedding (ex.: BGE-small).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from corpus_config import CORPUS_PASSAGE_DIR, CORPUS_PASSAGE_FILES

DEFAULT_JSONL = [
    CORPUS_PASSAGE_DIR / CORPUS_PASSAGE_FILES[d]
    for d in ("clapnq", "cloud", "fiqa", "govt")
]


def entry_text(obj: dict, include_title: bool) -> str:
    text = obj.get("text") or ""
    if not include_title:
        return text
    title = (obj.get("title") or "").strip()
    if title:
        return f"{title}\n{text}"
    return text


def count_one_tiktoken(text: str, enc) -> int:
    return len(enc.encode(text))


def count_one_hf(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def analyze_file(path: Path, counter, tqdm_bar: bool) -> dict:
    counts: list[int] = []
    n_skipped = 0
    with path.open(encoding="utf-8") as raw_f:
        f = raw_f
        if tqdm_bar:
            try:
                from tqdm import tqdm

                f = tqdm(raw_f, desc=path.name, unit="linha")
            except ImportError:
                pass
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            text = entry_text(obj, include_title=counter["include_title"])
            n = counter["fn"](text, counter["backend"])
            counts.append(n)

    if not counts:
        return {
            "path": str(path),
            "n_entries": 0,
            "n_skipped_lines": n_skipped,
            "total_tokens": 0,
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "min": 0,
            "max": 0,
        }

    return {
        "path": str(path),
        "n_entries": len(counts),
        "n_skipped_lines": n_skipped,
        "total_tokens": sum(counts),
        "mean": statistics.mean(counts),
        "median": statistics.median(counts),
        "stdev": statistics.stdev(counts) if len(counts) > 1 else 0.0,
        "min": min(counts),
        "max": max(counts),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Métricas de tokens por entrada (JSONL passage-level).")
    p.add_argument(
        "jsonl",
        nargs="*",
        type=Path,
        default=DEFAULT_JSONL,
        help="Arquivos .jsonl (padrão: os quatro passage_level do SemEval).",
    )
    p.add_argument(
        "--tiktoken-model",
        default="cl100k_base",
        help="Nome do encoding tiktoken (default: cl100k_base).",
    )
    p.add_argument(
        "--hf-tokenizer",
        metavar="MODEL",
        default=None,
        help="Se definido, usa AutoTokenizer deste modelo (ex.: BAAI/bge-small-en-v1.5) em vez de tiktoken.",
    )
    p.add_argument(
        "--include-title",
        action="store_true",
        help="Concatena title + texto da entrada antes de contar.",
    )
    p.add_argument("--no-progress", action="store_true", help="Desativa barra tqdm.")
    args = p.parse_args()

    include_title = args.include_title
    if args.hf_tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
        counter = {
            "fn": count_one_hf,
            "backend": tok,
            "include_title": include_title,
        }
        backend_label = f"HuggingFace {args.hf_tokenizer}"
    else:
        import tiktoken

        enc = tiktoken.get_encoding(args.tiktoken_model)
        counter = {
            "fn": count_one_tiktoken,
            "backend": enc,
            "include_title": include_title,
        }
        backend_label = f"tiktoken {args.tiktoken_model}"

    rows = []
    for path in args.jsonl:
        path = path.resolve()
        if not path.is_file():
            print(f"Aviso: arquivo não encontrado, ignorando: {path}", file=sys.stderr)
            continue
        rows.append(analyze_file(path, counter, tqdm_bar=not args.no_progress))

    print(f"Tokenizer: {backend_label}")
    if include_title:
        print("Texto contado: title + '\\n' + text")
    else:
        print("Texto contado: campo 'text' apenas")
    print()

    hdr = (
        f"{'dataset':<20} {'n':>8} {'total_tok':>12} {'média':>10} {'mediana':>10} "
        f"{'std':>8} {'mín':>6} {'máx':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    grand_n = grand_tok = 0
    for r in rows:
        name = Path(r["path"]).name
        print(
            f"{name:<20} {r['n_entries']:>8} {r['total_tokens']:>12} "
            f"{r['mean']:>10.2f} {r['median']:>10.1f} {r['stdev']:>8.2f} "
            f"{r['min']:>6} {r['max']:>6}"
        )
        grand_n += r["n_entries"]
        grand_tok += r["total_tokens"]
        if r["n_skipped_lines"]:
            print(f"  ({r['n_skipped_lines']} linhas ignoradas por JSON inválido)")

    if len(rows) > 1:
        overall_mean = grand_tok / grand_n if grand_n else 0.0
        print("-" * len(hdr))
        print(f"{'TOTAL (soma)':<20} {grand_n:>8} {grand_tok:>12} {'—':>10} {'—':>10} {'—':>8} {'—':>6} {'—':>6}")
        print(f"Média global (total_tokens / total_entradas): {overall_mean:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
