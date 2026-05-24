#!/usr/bin/env python3
"""
Gera todas as tabelas (CSV) e gráficos (PNG) dos resultados do TCC.

Uso:
    python tcc_implementation/plot_results.py

Saída em: tcc_implementation/figures/
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ── Caminhos ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
BASE_LLAMA = ROOT / "results/llama/llama3_1_8b_q4_ctx8192/evaluations"
BASE_A     = ROOT / "results/task_a_retrieval/metrics"
FIGURES    = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

DOMAINS    = ["fiqa", "clapnq", "cloud", "govt"]
STRATEGIES_A = ["legacy", "small", "large", "multiscale"]
STRATEGIES_C = ["noretrieval", "small", "multiscale", "legacy", "large"]

STRATEGY_LABELS = {
    "noretrieval": "No-Retrieval",
    "small":       "Small",
    "multiscale":  "Multiscale",
    "legacy":      "Legacy",
    "large":       "Large",
}
DOMAIN_LABELS = {"fiqa": "FiQA", "clapnq": "ClapNQ", "cloud": "Cloud", "govt": "Govt"}

PALETTE_STRAT = {
    "noretrieval": "#d62728",
    "small":       "#aec7e8",
    "multiscale":  "#ffbb78",
    "legacy":      "#98df8a",
    "large":       "#1f77b4",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def mean(vals: list) -> float:
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 4) if v else 0.0

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def save(fig: plt.Figure, name: str) -> None:
    p = FIGURES / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {p}")

# ── Coletar dados ─────────────────────────────────────────────────────────────

def load_task_a() -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        for strategy in STRATEGIES_A:
            f = BASE_A / domain / f"eval_a_{domain}_{strategy}.json"
            if not f.is_file():
                continue
            d = json.loads(f.read_text())
            avg = d.get("averages", {})
            rows.append({
                "domain": domain, "strategy": strategy,
                "NDCG@1":  avg.get("NDCG@1",  0),
                "NDCG@3":  avg.get("NDCG@3",  0),
                "NDCG@5":  avg.get("NDCG@5",  0),
                "NDCG@10": avg.get("NDCG@10", 0),
                "Recall@1":  avg.get("Recall@1",  0),
                "Recall@3":  avg.get("Recall@3",  0),
                "Recall@5":  avg.get("Recall@5",  0),
                "Recall@10": avg.get("Recall@10", 0),
            })
    return pd.DataFrame(rows)


def load_task_b() -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        f = BASE_LLAMA / "task_b" / f"eval_b_{domain}.algonly.jsonl"
        data = read_jsonl(f)
        if not data:
            continue
        rouge, bsr, bsp, rb = [], [], [], []
        for row in data:
            m = row.get("metrics", {})
            if m.get("RougeL_stemFalse"): rouge.append(m["RougeL_stemFalse"][0])
            if m.get("BertscoreR"):       bsr.append(m["BertscoreR"][0])
            if m.get("BertscoreP"):       bsp.append(m["BertscoreP"][0])
            if m.get("RB_agg"):           rb.append(m["RB_agg"][0])
        rows.append({"domain": domain, "n": len(data),
                     "ROUGE-L": mean(rouge), "BertScore-R": mean(bsr),
                     "BertScore-P": mean(bsp), "RB_agg": mean(rb)})
    return pd.DataFrame(rows)


def load_task_c() -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        for strategy in STRATEGIES_C:
            f = BASE_LLAMA / "task_c" / domain / f"eval_c_{domain}_{strategy}.algonly.jsonl"
            data = read_jsonl(f)
            if not data:
                continue
            rouge, bsr, bsp, rb = [], [], [], []
            for row in data:
                m = row.get("metrics", {})
                if m.get("RougeL_stemFalse"): rouge.append(m["RougeL_stemFalse"][0])
                if m.get("BertscoreR"):       bsr.append(m["BertscoreR"][0])
                if m.get("BertscoreP"):       bsp.append(m["BertscoreP"][0])
                if m.get("RB_agg"):           rb.append(m["RB_agg"][0])
            rows.append({"domain": domain, "strategy": strategy, "n": len(data),
                         "ROUGE-L": mean(rouge), "BertScore-R": mean(bsr),
                         "BertScore-P": mean(bsp), "RB_agg": mean(rb)})
    return pd.DataFrame(rows)


def load_judges() -> tuple[pd.DataFrame, pd.DataFrame]:
    qwen   = pd.read_csv(BASE_LLAMA / "ollama_judge/qwen2.5_7b/summary.csv")
    gemini = pd.read_csv(BASE_LLAMA / "gemini_judge/gemini-3.1-flash-lite/summary.csv")
    return qwen, gemini


# ── Tabelas CSV ───────────────────────────────────────────────────────────────

def export_tables(df_a, df_b, df_c, qwen, gemini):
    # Task A — média por estratégia
    a_strat = df_a.groupby("strategy")[["NDCG@10", "Recall@10"]].mean().round(4)
    a_strat.to_csv(FIGURES / "tabela_task_a_por_estrategia.csv")

    # Task A — por domínio x estratégia
    df_a.to_csv(FIGURES / "tabela_task_a_detalhada.csv", index=False)

    # Task B — por domínio
    df_b.to_csv(FIGURES / "tabela_task_b.csv", index=False)

    # Task C — média por estratégia
    c_strat = df_c.groupby("strategy")[["ROUGE-L", "BertScore-R", "BertScore-P", "RB_agg"]].mean().round(4)
    c_strat.to_csv(FIGURES / "tabela_task_c_por_estrategia.csv")

    # Task C — detalhada
    df_c.to_csv(FIGURES / "tabela_task_c_detalhada.csv", index=False)

    # Juízes
    qwen.to_csv(FIGURES / "tabela_qwen_judge.csv", index=False)
    gemini.to_csv(FIGURES / "tabela_gemini_judge.csv", index=False)

    print("  Tabelas CSV exportadas.")


# ── Gráfico 1: Task A — nDCG@10 e Recall@10 por estratégia ──────────────────

def plot_task_a(df_a: pd.DataFrame):
    avg = df_a.groupby("strategy")[["NDCG@10", "Recall@10"]].mean().reindex(STRATEGIES_A)
    avg.index = [STRATEGY_LABELS[s] for s in avg.index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    colors = [PALETTE_STRAT[s] for s in STRATEGIES_A]

    for ax, metric in zip(axes, ["NDCG@10", "Recall@10"]):
        bars = ax.bar(avg.index, avg[metric], color=colors, edgecolor="white", width=0.6)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, avg[metric].max() * 1.25)
        ax.tick_params(axis="x", rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Task A — Retrieval por Estratégia de Chunking\n(média de 4 domínios)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "01_task_a_retrieval.png")


# ── Gráfico 2: Task A — nDCG@10 por domínio x estratégia ────────────────────

def plot_task_a_by_domain(df_a: pd.DataFrame):
    pivot = df_a.pivot(index="domain", columns="strategy", values="NDCG@10")
    pivot = pivot.reindex(columns=STRATEGIES_A)
    pivot.index = [DOMAIN_LABELS[d] for d in pivot.index]
    pivot.columns = [STRATEGY_LABELS[s] for s in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot))
    width = 0.18
    offsets = np.linspace(-(len(STRATEGIES_A)-1)/2, (len(STRATEGIES_A)-1)/2, len(STRATEGIES_A)) * width

    for i, (col, strat) in enumerate(zip(pivot.columns, STRATEGIES_A)):
        bars = ax.bar(x + offsets[i], pivot[col], width=width,
                      label=col, color=PALETTE_STRAT[strat], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=11)
    ax.set_ylabel("nDCG@10", fontsize=11)
    ax.set_title("Task A — nDCG@10 por Domínio e Estratégia", fontsize=13, fontweight="bold")
    ax.legend(title="Estratégia", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, pivot.values.max() * 1.25)
    fig.tight_layout()
    save(fig, "02_task_a_by_domain.png")


# ── Gráfico 3: Task C — RB_agg por estratégia (média de domínios) ────────────

def plot_task_c_mean(df_c: pd.DataFrame):
    avg = df_c.groupby("strategy")[["ROUGE-L", "BertScore-R", "RB_agg"]].mean()
    avg = avg.reindex(STRATEGIES_C)
    avg.index = [STRATEGY_LABELS[s] for s in avg.index]
    colors = [PALETTE_STRAT[s] for s in STRATEGIES_C]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for ax, metric in zip(axes, ["RB_agg", "ROUGE-L", "BertScore-R"]):
        bars = ax.bar(avg.index, avg[metric], color=colors, edgecolor="white", width=0.6)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=10)
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
        ymax = avg[metric].max()
        ax.set_ylim(min(0, avg[metric].min() - 0.02), ymax * 1.25)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + ymax * 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    fig.suptitle("Task C — Geração RAG por Estratégia de Chunking\n(média de 4 domínios, Llama 3.1 8B Q4)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "03_task_c_mean_metrics.png")


# ── Gráfico 4: Task C — RB_agg por domínio x estratégia (barras agrupadas) ──

def plot_task_c_by_domain(df_c: pd.DataFrame):
    pivot = df_c.pivot(index="domain", columns="strategy", values="RB_agg")
    pivot = pivot.reindex(columns=STRATEGIES_C)
    pivot.index = [DOMAIN_LABELS[d] for d in pivot.index]
    pivot.columns = [STRATEGY_LABELS[s] for s in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(pivot))
    width = 0.15
    n = len(STRATEGIES_C)
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    for i, (col, strat) in enumerate(zip(pivot.columns, STRATEGIES_C)):
        vals = pivot[col].fillna(0)
        bars = ax.bar(x + offsets[i], vals, width=width,
                      label=col, color=PALETTE_STRAT[strat], edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=11)
    ax.set_ylabel("RB_agg", fontsize=11)
    ax.set_title("Task C — RB_agg por Domínio e Estratégia de Chunking", fontsize=13, fontweight="bold")
    ax.legend(title="Estratégia", fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, pivot.values.max() * 1.25)
    fig.tight_layout()
    save(fig, "04_task_c_by_domain.png")


# ── Gráfico 5: Heatmap RB_agg domínio x estratégia ──────────────────────────

def plot_heatmap(df_c: pd.DataFrame):
    pivot = df_c.pivot(index="domain", columns="strategy", values="RB_agg")
    pivot = pivot.reindex(columns=STRATEGIES_C)
    pivot.index = [DOMAIN_LABELS[d] for d in pivot.index]
    pivot.columns = [STRATEGY_LABELS[s] for s in pivot.columns]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn",
                linewidths=0.5, ax=ax, vmin=0, vmax=0.5,
                annot_kws={"size": 11})
    ax.set_title("Task C — RB_agg por Domínio × Estratégia", fontsize=13, fontweight="bold")
    ax.set_xlabel("Estratégia", fontsize=11)
    ax.set_ylabel("Domínio", fontsize=11)
    fig.tight_layout()
    save(fig, "05_heatmap_rb_agg.png")


# ── Gráfico 6: Comparação Task B vs baseline MTRAG ───────────────────────────

def plot_baseline_comparison(df_b: pd.DataFrame):
    baselines = {
        "Reference\n(teto)":    0.89,
        "GPT-4o":               0.60,
        "Qwen 2.5 7B":          0.55,
        "Llama 3.1 8B\n(paper)":0.45,
        "Seu sistema\n(RB_agg)": df_b["RB_agg"].mean(),
        "Llama 3.1 8B\n1.5B Qwen\n(inicial)": None,  # placeholder, skip
    }
    # só os que têm valor
    labels = ["Reference\n(teto)", "GPT-4o", "Qwen 2.5 7B",
              "Llama 3.1 8B\n(paper)", "Seu sistema\n(RB_agg)"]
    values = [0.89, 0.60, 0.55, 0.45, round(df_b["RB_agg"].mean(), 4)]
    colors_bar = ["#2ca02c", "#ff7f0e", "#9467bd", "#1f77b4", "#d62728"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors_bar, edgecolor="white", width=0.5)
    ax.set_ylabel("HM3 / RB_agg", fontsize=11)
    ax.set_title("Task B — Comparação com Baselines Oficiais MTRAG\n(mesmo conjunto público, 842 instâncias)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.45, color="#1f77b4", linestyle="--", linewidth=1, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # nota de rodapé
    ax.text(0.5, -0.18,
            "* Baselines (HM3) incluem RB_llm e RL_F além de RB_alg. Seu sistema calculou apenas RB_alg (RB_agg).",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    save(fig, "06_baseline_comparison.png")


# ── Gráfico 7: LLM-as-a-Judge — Qwen vs Gemini (faithfulness por estratégia) ─

def plot_judge_comparison(qwen: pd.DataFrame, gemini: pd.DataFrame):
    strat_order = ["noretrieval", "small", "multiscale", "legacy", "large"]
    strat_labels = [STRATEGY_LABELS[s] for s in strat_order]

    # Média dos domínios por estratégia
    q_avg = qwen.groupby("strategy")[["faithfulness_mean", "answer_relevance_mean"]].mean().reindex(strat_order)
    g_avg = gemini.groupby("strategy")[["faithfulness_mean", "answer_relevance_mean"]].mean().reindex(strat_order)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    x = np.arange(len(strat_order))
    width = 0.35

    for ax, metric, label in zip(
        axes,
        ["faithfulness_mean", "answer_relevance_mean"],
        ["Faithfulness (média)", "Answer Relevance (média)"]
    ):
        q_vals = q_avg[metric].values
        g_vals = g_avg[metric].values

        b1 = ax.bar(x - width/2, q_vals, width, label="Qwen 2.5 7B (local, escala 1–5)",
                    color="#1f77b4", edgecolor="white")
        b2 = ax.bar(x + width/2, g_vals, width, label="Gemini 3.1 Flash Lite (amostra, escala 1–5)",
                    color="#ff7f0e", edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(strat_labels, rotation=15, fontsize=10)
        ax.set_ylabel("Score (1–5)", fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 5.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("LLM-as-a-Judge — Qwen 2.5 7B (completo) vs Gemini 3.1 Flash Lite (amostra estratificada)\nTask C, média de 4 domínios",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "07_judge_comparison.png")


# ── Gráfico 8: Task C — RAG vs No-Retrieval (ganho do RAG) ──────────────────

def plot_rag_gain(df_c: pd.DataFrame):
    avg = df_c.groupby("strategy")["RB_agg"].mean().reindex(STRATEGIES_C)
    baseline_val = avg["noretrieval"]
    gains = (avg - baseline_val).drop("noretrieval")
    gains.index = [STRATEGY_LABELS[s] for s in gains.index]
    colors_gain = [PALETTE_STRAT[s] for s in ["small", "multiscale", "legacy", "large"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(gains.index, gains.values, color=colors_gain, edgecolor="white", width=0.5)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Δ RB_agg vs No-Retrieval", fontsize=11)
    ax.set_title("Ganho do RAG sobre o Baseline Sem Retrieval\n(Task C — média de 4 domínios)",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"+{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "08_rag_gain.png")


# ── Gráfico 9: Qwen judge — hallucination rate por estratégia ────────────────

def plot_hallucination(qwen: pd.DataFrame):
    strat_order = ["noretrieval", "small", "multiscale", "legacy", "large"]
    strat_labels = [STRATEGY_LABELS[s] for s in strat_order]

    avg = qwen.groupby("strategy")[["hallucination_yes_rate", "hallucination_partial_rate"]].mean().reindex(strat_order)
    avg.index = strat_labels

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(strat_labels))
    width = 0.35
    b1 = ax.bar(x - width/2, avg["hallucination_yes_rate"],   width, label="Alucinação (yes)",     color="#d62728", edgecolor="white")
    b2 = ax.bar(x + width/2, avg["hallucination_partial_rate"], width, label="Alucinação (partial)", color="#ff7f0e", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=10)
    ax.set_ylabel("Taxa (0–1)", fontsize=11)
    ax.set_title("Taxa de Alucinação por Estratégia — Qwen 2.5 7B Judge\nTask C, média de 4 domínios",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 0.35)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save(fig, "09_hallucination_rates.png")


# ── Gráfico 10: H3 — RB_agg por faixa de turno e estratégia ─────────────────

def turn_band(n: int) -> str:
    if n <= 2:
        return "Início\n(turnos 1–2)"
    elif n <= 5:
        return "Meio\n(turnos 3–5)"
    else:
        return "Final\n(turnos 6+)"

TURN_ORDER = ["Início\n(turnos 1–2)", "Meio\n(turnos 3–5)", "Final\n(turnos 6+)"]


def load_task_c_by_turn() -> pd.DataFrame:
    base_c = BASE_LLAMA / "task_c"
    rows = []
    for domain in DOMAINS:
        for strategy in STRATEGIES_C:
            f = base_c / domain / f"eval_c_{domain}_{strategy}.algonly.jsonl"
            data = read_jsonl(f)
            for row in data:
                m_tid = re.search(r"<::>(\d+)", row.get("task_id", ""))
                if not m_tid:
                    continue
                turn = int(m_tid.group(1))
                band = turn_band(turn)
                metrics = row.get("metrics", {})
                rb = metrics.get("RB_agg", [None])[0]
                rouge = metrics.get("RougeL_stemFalse", [None])[0]
                rows.append({"domain": domain, "strategy": strategy,
                             "turn": turn, "band": band,
                             "RB_agg": rb, "ROUGE-L": rouge})
    return pd.DataFrame(rows)


def plot_h3(df_turn: pd.DataFrame):
    # Aggregate: mean RB_agg by strategy x turn band (across all domains)
    agg = (
        df_turn.dropna(subset=["RB_agg"])
        .groupby(["strategy", "band"])["RB_agg"]
        .agg(["mean", "count"])
        .reset_index()
    )
    agg.columns = ["strategy", "band", "RB_agg_mean", "n"]
    agg = agg[agg["strategy"].isin(STRATEGIES_C)]

    # Pivot for grouped bar chart
    pivot = agg.pivot(index="band", columns="strategy", values="RB_agg_mean")
    pivot = pivot.reindex(index=TURN_ORDER, columns=STRATEGIES_C)
    pivot.columns = [STRATEGY_LABELS[s] for s in STRATEGIES_C]

    # Count pivot for annotations
    n_pivot = agg.pivot(index="band", columns="strategy", values="n")
    n_pivot = n_pivot.reindex(index=TURN_ORDER, columns=STRATEGIES_C)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(TURN_ORDER))
    n_strat = len(STRATEGIES_C)
    width = 0.14
    offsets = np.linspace(-(n_strat - 1) / 2, (n_strat - 1) / 2, n_strat) * width

    for i, (col, strat) in enumerate(zip(pivot.columns, STRATEGIES_C)):
        vals = pivot[col].values
        bars = ax.bar(x + offsets[i], vals, width=width,
                      label=col, color=PALETTE_STRAT[strat], edgecolor="white")
        for j, bar in enumerate(bars):
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(TURN_ORDER, fontsize=11)
    ax.set_ylabel("RB_agg (média)", fontsize=11)
    ax.set_title(
        "H3 — RB_agg por Faixa de Turno e Estratégia de Chunking\n"
        "(Task C, média de 4 domínios — Llama 3.1 8B Q4)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(title="Estratégia", fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, pivot.values.max() * 1.30)

    # Footnote with sample sizes
    ns = agg[agg["strategy"] == "legacy"].set_index("band")["n"].reindex(TURN_ORDER)
    footnote = "n por faixa (legacy): " + ", ".join(
        f"{b.split(chr(10))[0]}={int(v)}" for b, v in zip(TURN_ORDER, ns.values)
    )
    ax.text(0.5, -0.08, footnote, transform=ax.transAxes,
            ha="center", fontsize=8, color="gray")

    fig.tight_layout()
    save(fig, "10_h3_turn_depth.png")

    # Also export table
    pivot.round(4).to_csv(FIGURES / "tabela_h3_por_turno.csv")
    print("  Tabela H3 exportada.")


def plot_h3_line(df_turn: pd.DataFrame):
    """Gráfico de linha: RB_agg por número de turno (1–8) para cada estratégia."""
    agg = (
        df_turn[df_turn["turn"] <= 8]
        .dropna(subset=["RB_agg"])
        .groupby(["strategy", "turn"])["RB_agg"]
        .agg(["mean", "count"])
        .reset_index()
    )
    agg.columns = ["strategy", "turn", "RB_agg_mean", "n"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for strat in STRATEGIES_C:
        sub = agg[agg["strategy"] == strat].sort_values("turn")
        ax.plot(sub["turn"], sub["RB_agg_mean"],
                marker="o", label=STRATEGY_LABELS[strat],
                color=PALETTE_STRAT[strat], linewidth=2, markersize=6)

    ax.set_xlabel("Número do Turno na Conversa", fontsize=11)
    ax.set_ylabel("RB_agg (média)", fontsize=11)
    ax.set_title(
        "H3 — Evolução do RB_agg por Turno e Estratégia\n"
        "(Task C, média de 4 domínios, turnos 1–8)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticks(range(1, 9))
    ax.legend(title="Estratégia", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 0.55)
    ax.text(0.5, -0.08,
            "Turnos 9–10 omitidos por baixo volume amostral (n < 10).",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    save(fig, "11_h3_turn_line.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Carregando dados...")
    df_a   = load_task_a()
    df_b   = load_task_b()
    df_c   = load_task_c()
    qwen, gemini = load_judges()

    print("\nExportando tabelas CSV...")
    export_tables(df_a, df_b, df_c, qwen, gemini)

    print("\nGerando gráficos...")
    plot_task_a(df_a)
    plot_task_a_by_domain(df_a)
    plot_task_c_mean(df_c)
    plot_task_c_by_domain(df_c)
    plot_heatmap(df_c)
    plot_baseline_comparison(df_b)
    plot_judge_comparison(qwen, gemini)
    plot_rag_gain(df_c)
    plot_hallucination(qwen)

    print("\nCarregando dados por turno (H3)...")
    df_turn = load_task_c_by_turn()
    plot_h3(df_turn)
    plot_h3_line(df_turn)

    print(f"\nPronto! Todos os arquivos estão em: {FIGURES.resolve()}")


if __name__ == "__main__":
    main()
