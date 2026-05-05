"""
Geração de texto local para os experimentos do TCC.

Backends:
- ollama: usa a API local do Ollama (default para Llama quantizado).
- hf: usa Hugging Face Transformers (mantém compatibilidade com Qwen).

Configure por variáveis de ambiente:
- LOCAL_LLM_BACKEND=ollama|hf
- LOCAL_LLM_MODEL=llama3.1:8b ou Qwen/Qwen2.5-1.5B-Instruct
- OLLAMA_HOST=http://127.0.0.1:11434
- OLLAMA_NUM_CTX=4096
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from functools import lru_cache

DEFAULT_HF_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OLLAMA_MODEL_ID = "llama3.1:8b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


def _backend() -> str:
    return os.environ.get("LOCAL_LLM_BACKEND", "ollama").strip().lower()


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def _carregar():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.environ.get("LOCAL_LLM_MODEL", DEFAULT_HF_MODEL_ID)
    device = _device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    model = model.to(device)
    model.eval()
    return tok, model, device


def _gerar_texto_hf(prompt_usuario: str, max_new_tokens: int = 512) -> str:
    import torch

    tokenizer, model, device = _carregar()
    messages = [{"role": "user", "content": prompt_usuario}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        texto = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        texto = prompt_usuario

    inputs = tokenizer(texto, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gerado = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gerado, skip_special_tokens=True).strip()


def _gerar_texto_ollama(prompt_usuario: str, max_new_tokens: int = 512) -> str:
    model_id = os.environ.get("LOCAL_LLM_MODEL", DEFAULT_OLLAMA_MODEL_ID)
    host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
    num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
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


def gerar_texto(prompt_usuario: str, max_new_tokens: int = 512) -> str:
    if _backend() == "hf":
        return _gerar_texto_hf(prompt_usuario, max_new_tokens=max_new_tokens)
    if _backend() == "ollama":
        return _gerar_texto_ollama(prompt_usuario, max_new_tokens=max_new_tokens)
    raise ValueError("LOCAL_LLM_BACKEND deve ser 'ollama' ou 'hf'.")
