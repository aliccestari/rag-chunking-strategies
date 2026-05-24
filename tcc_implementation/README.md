# TCC — Estratégias de Chunking em RAG Multi-Turn

Pipeline RAG local construído em cima do benchmark
[MTRAG](../README.md) (IBM) para comparar **estratégias de chunking** em
recuperação e geração multi-turn. Foco do trabalho: medir o efeito de
chunks pequenos, grandes e da estratégia *multi-scale* (recuperação em
chunks pequenos, geração com a passagem inteira) sobre as três subtasks
do MTRAG.

## Componentes

- **Recuperação**: índice [Chroma](https://www.trychroma.com/) local +
  embeddings BGE-small (`BAAI/bge-small-en-v1.5`) via
  `sentence-transformers`, com MPS no Mac.
- **Geração**: dois backends de LLM
  - `ollama` — Llama 3.1 8B Q4 (default; precisa do daemon `ollama`).
  - `hf` — Hugging Face Transformers (testado com Qwen 2.5 1.5B Instruct).
- **Subtasks MTRAG**:
  - **A** — só retrieval (qrels e métricas oficiais).
  - **B** — geração com passagens ouro do MTRAG.
  - **C** — RAG end-to-end (retrieval + geração).
- **LLM-as-Judge**:
  - `run_gemini_judge.py` — Gemini (API gratuita; resumível).
  - `run_ollama_judge.py` — Qwen 2.5 7B local via Ollama (mitiga viés
    de auto-avaliação contra o gerador Llama 3.1 8B).
- **Análise**: `plot_results.py` produz 11 figuras e 8 tabelas finais
  em `figures/` a partir dos JSONLs em `results/`.

## Estratégias de chunking

Configuradas em `chunking_strategies.py`:

| Estratégia | Chunk size | Overlap | Notas |
|---|---:|---:|---|
| `legacy`    | 2000 |  500 | Compatível com índices antigos |
| `small`     |  900 |  120 | Precisão na recuperação |
| `large`     | 12000 | 400 | Mais contexto por vetor |
| `multiscale` | 900 | 120 | Mesmo índice que `small`; na geração devolve a passagem **completa** do JSONL |

## Setup

```bash
# Ambiente principal do TCC
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/base-lock.txt   # reprodutível

# (opcional) Avaliação oficial SemEval — venv separado
python -m venv .venv-eval
source .venv-eval/bin/activate
pip install -r requirements/eval.txt
```

Variáveis de ambiente em `.env` (raiz do repo):

```ini
GOOGLE_API_KEY=...           # main.py (chat) e run_gemini_judge.py
LOCAL_LLM_BACKEND=ollama     # ou "hf"
LOCAL_LLM_MODEL=llama3.1:8b  # ou "Qwen/Qwen2.5-1.5B-Instruct"
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_NUM_CTX=8192          # janela de contexto (tokens) usada no TCC
```

Corpora MTRAG precisam estar descompactados em
`semeval/corpora/passage_level/<domínio>.jsonl` (ver
[README upstream](../README.md)).

## Fluxo típico

Todos os comandos abaixo rodam **a partir de `tcc_implementation/`**.

### 1. Construir os índices (uma vez por domínio × estratégia)

```bash
python criar_db.py --domain govt --strategy small
python criar_db.py --domain govt --strategy large
python criar_db.py --domain govt --strategy legacy
# multiscale reaproveita o índice "small"; não precisa rodar de novo
```

Os índices ficam em `indices/db_local_bge_<dom>[_<estratégia>]/` e estão
no `.gitignore` (cada um ocupa 0,5–1,4 GB).

### 2. Rodar as subtasks

```bash
# Subtask A — retrieval
python run_mtrag.py task-a --domain govt --queries lastturn \
    --chunking small -o preds_a_small.jsonl

# Subtask B — geração com passagens ouro
python run_mtrag.py task-b --domain govt -o preds_b.jsonl

# Subtask C — RAG end-to-end
python run_mtrag.py task-c --domain govt --queries lastturn \
    --chunking multiscale -o preds_c_ms.jsonl

# Baseline sem retrieval (só LLM)
python run_mtrag.py task-c --domain govt --queries lastturn \
    --baseline noretrieval -o preds_c_nr.jsonl

# Logar tempos por turno (CSV) para a análise H3
python run_mtrag.py task-c --domain govt --chunking large \
    --timing-log results/timings/govt/timings_govt_c_large.csv \
    -o preds.jsonl
```

### 3. Avaliação

```bash
# Recall@k e nDCG@k para Subtask A
python run_mtrag.py eval-a --domain govt --predictions preds_a_small.jsonl \
    -o metrics_a_small.json

# Format check (formato oficial SemEval)
python run_mtrag.py format-check --task c --domain govt \
    --predictions preds_c_ms.jsonl

# Generation eval (DeBERTa/ROUGE + juiz LLM) — usa .venv-eval
python run_mtrag.py gen-eval --predictions preds_b.jsonl -o eval_b.jsonl \
    --provider hf --judge-model ibm-granite/granite-3.3-8b-instruct

# Retomar só os juízes (não repete DeBERTa/ROUGE; ~20 min/domínio):
python run_mtrag.py gen-eval --predictions eval_b.jsonl -o eval_b.jsonl \
    --skip-algorithmic --provider hf \
    --judge-model ibm-granite/granite-3.3-8b-instruct
```

### 4. Juízes LLM (Task C)

```bash
# Juiz Gemini (gratuito; usa GEMINI_API_KEY ou GOOGLE_API_KEY do .env)
python run_gemini_judge.py --domain govt --strategy multiscale

# Juiz Qwen 2.5 7B via Ollama
python run_ollama_judge.py --domain govt --strategy multiscale
```

Ambos suportam **retomada**: re-executar pula `task_id`s já avaliados.

### 5. Plots e tabelas finais

```bash
python plot_results.py
# Gera figures/01_*.png ... figures/11_*.png + figures/tabela_*.csv
```

## Chat interativo (uso ad-hoc)

```bash
# Pergunta única no terminal, usando o índice atual de DOMINIO_ATUAL
python main.py
# Por padrão usa o LLM local; export USE_GEMINI=1 para usar Gemini.
```

## Layout do diretório

```
tcc_implementation/
├── README.md                   (este arquivo)
├── requirements.txt            (atalho → requirements/base-lock.txt)
├── requirements/
│   ├── base.txt                deps diretas
│   ├── base-lock.txt           pip freeze reprodutível
│   ├── eval.txt                referência ao gen-eval do SemEval
│   └── dev.txt                 ruff
│
├── corpus_config.py            caminhos por domínio + pasta_indice_chroma
├── embeddings_config.py        BGE-small
├── chunking_strategies.py      legacy / small / large / multiscale
├── retrieval_core.py           recuperar_passagens_unicas + multiscale
├── retrieval_metrics.py        Recall@k, nDCG@k
├── mtrag_query_parse.py        parse de queries BEIR multi-turn
├── mtrag_subtasks.py           task_a/b/c_um_turno
├── rag_prompts.py              prompts de geração
├── local_llm.py                backend ollama/hf
│
├── criar_db.py                 CLI: construir índices Chroma
├── run_mtrag.py                CLI: subtasks A/B/C + eval + format-check + gen-eval
├── run_gemini_judge.py         CLI: juiz Gemini
├── run_ollama_judge.py         CLI: juiz Qwen 2.5 7B via Ollama
├── plot_results.py             CLI: figuras e tabelas finais
├── count_dataset_tokens.py     estatística do dataset
├── experiments_tcc.py          batch de experimentos
├── main.py                     chat interativo (smoke test)
│
├── indices/                    (ignorado) índices Chroma — recriar com criar_db.py
├── results/                    artefatos de experimentos
│   ├── llama/                  predições + avaliações por configuração
│   ├── qwen/
│   ├── task_a_retrieval/       metrics + predictions
│   └── timings/<dom>/          CSVs de latência por turno
└── figures/                    figuras (PNG) e tabelas (CSV) finais do TCC
```

## Resultados versionados

- `figures/` → figuras (11 PNGs) e tabelas (9 CSVs) **finais** que entram
  no TCC.
- `results/**/summary.csv` (por modelo de juiz) → métricas agregadas
  finais.
- `results/**/preds_*.jsonl`, `eval_*.jsonl`, `timings_*.csv`,
  `*_judge_*.jsonl` → **ignorados** (regeráveis e volumosos).
- `indices/` → **ignorada** (~11 GB).

## Reproduzindo do zero

1. Clonar o repo e baixar corpora (`semeval/corpora/passage_level/*.jsonl`).
2. `pip install -r requirements/base-lock.txt`.
3. `python criar_db.py --domain <dom> --strategy <s>` para cada
   combinação desejada.
4. Rodar `run_mtrag.py task-a/b/c` para gerar predições.
5. Rodar `run_gemini_judge.py` e/ou `run_ollama_judge.py` para Task C.
6. Rodar `plot_results.py` para regerar `figures/`.
