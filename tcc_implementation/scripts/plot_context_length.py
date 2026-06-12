#!/usr/bin/env python3
"""
Gera a figura de tamanho do contexto enviado ao gerador (Subtask C),
por estratégia e domínio, com a linha do limite da janela de contexto.

Recalcula o tamanho a partir do campo `contexts` das predições da Task C.

Uso:
    python tcc_implementation/scripts/plot_context_length.py

Saída: tcc_implementation/figures/fig_context_length.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PREDS = ROOT / "results/llama/llama3_1_8b_q4_ctx8192/predictions/task_c"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# Ordem: domínios de documentos curtos primeiro, depois longos, para o contraste
DOMAINS = ["clapnq", "fiqa", "cloud", "govt"]
DOMAIN_LABELS = {"clapnq": "ClapNQ", "fiqa": "FiQA", "cloud": "Cloud", "govt": "Govt"}

# noretrieval não tem contexto; comparamos as quatro estratégias de chunking
STRATEGIES = ["small", "large", "multiscale", "legacy"]
STRATEGY_LABELS = {"small": "Small", "large": "Large",
                   "multiscale": "Multiscale", "legacy": "Legacy"}
PALETTE = {"small": "#1f77b4", "large": "#ff7f0e",
           "multiscale": "#2ca02c", "legacy": "#9467bd"}

# Janela do gerador: 8192 tokens. Aproximação usual ~4 chars por token.
CTX_LIMIT_TOKENS = 8192
CHARS_PER_TOKEN = 4
CTX_LIMIT_CHARS = CTX_LIMIT_TOKENS * CHARS_PER_TOKEN  # ~32768


def iter_objects(path: Path):
    """Itera objetos JSON de um arquivo, robusto a quebras de linha dentro de strings."""
    text = path.read_text()
    dec = json.JSONDecoder()
    idx, n = 0, len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = dec.raw_decode(text, idx)
        yield obj
        idx = end


def context_sizes(path: Path) -> list[int]:
    return [sum(len(c.get("text", "")) for c in obj.get("contexts", []))
            for obj in iter_objects(path)]


def collect() -> tuple[dict, float]:
    """Retorna {(domain, strategy): mean_chars} e o máximo global de chars."""
    means: dict = {}
    overall_max = 0.0
    for dom in DOMAINS:
        for strat in STRATEGIES:
            f = PREDS / dom / f"preds_{dom}_c_{strat}.jsonl"
            if not f.is_file():
                continue
            sizes = context_sizes(f)
            if not sizes:
                continue
            means[(dom, strat)] = sum(sizes) / len(sizes)
            overall_max = max(overall_max, max(sizes))
    return means, overall_max


def main() -> None:
    means, overall_max = collect()

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(DOMAINS))
    width = 0.2
    offsets = np.linspace(-(len(STRATEGIES) - 1) / 2,
                          (len(STRATEGIES) - 1) / 2, len(STRATEGIES)) * width

    for i, strat in enumerate(STRATEGIES):
        vals = [means.get((dom, strat), 0.0) for dom in DOMAINS]
        ax.bar(x + offsets[i], vals, width=width,
               label=STRATEGY_LABELS[strat], color=PALETTE[strat],
               edgecolor="white")

    # Linha do limite da janela de contexto
    ax.axhline(CTX_LIMIT_CHARS, color="#d62728", linestyle="--", linewidth=1.2)
    ax.text(ax.get_xlim()[1], CTX_LIMIT_CHARS, "  ctx limit (~8192 tokens)",
            color="#d62728", fontsize=12, va="bottom", ha="right")

    # Anotação do caso extremo
    ax.annotate(f"max: {overall_max/1000:.0f}K chars",
                xy=(0.99, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=12, color="#d62728",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Average context size (characters)", fontsize=13)
    ax.set_title(
        "Average Context Size Sent to the Generator\n"
        "by Strategy and Domain (Subtask C)",
        fontsize=15, fontweight="bold"
    )
    ax.legend(title="Strategy", fontsize=11, title_fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(means.values()) * 1.15)

    fig.tight_layout()
    out = FIGURES / "fig_context_length.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out}")

    # Também imprime medianas em tokens (para o texto da discussão)
    for dom in ["cloud", "govt"]:
        f = PREDS / dom / f"preds_{dom}_c_multiscale.jsonl"
        if f.is_file():
            toks = sorted(s / CHARS_PER_TOKEN for s in context_sizes(f))
            med = toks[len(toks) // 2]
            print(f"  {dom} multiscale: mediana ~{med/1000:.0f}K tokens")


if __name__ == "__main__":
    main()
