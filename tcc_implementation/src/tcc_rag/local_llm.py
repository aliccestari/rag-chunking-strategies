"""
Geração de texto local para os experimentos do TCC.

Usa Llama 3.1 8B Q4 via Ollama. O TCC isola chunking como variável,
mantendo o gerador fixo — por isso o backend é único.

Configure por variáveis de ambiente:
- LOCAL_LLM_MODEL=llama3.1:8b
- OLLAMA_HOST=http://127.0.0.1:11434
- OLLAMA_NUM_CTX=8192
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_OLLAMA_MODEL_ID = "llama3.1:8b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_NUM_CTX = 8192


def gerar_texto(prompt_usuario: str, max_new_tokens: int = 512) -> str:
    model_id = os.environ.get("LOCAL_LLM_MODEL", DEFAULT_OLLAMA_MODEL_ID)
    host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
    num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", str(DEFAULT_OLLAMA_NUM_CTX)))
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_usuario}],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": max_new_tokens,
            "num_ctx": num_ctx,
        },
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Falha ao chamar Ollama em {host}. "
            "Confirme que `ollama serve` está rodando."
        ) from exc

    msg = data.get("message") or {}
    return str(msg.get("content", "")).strip()
