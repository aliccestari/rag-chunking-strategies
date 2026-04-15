"""
Geração de texto com modelo causal local (Hugging Face Transformers).

Configure o modelo via variável de ambiente LOCAL_LLM_MODEL (default: Qwen2.5-1.5B-Instruct).
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def _carregar():
    model_id = os.environ.get("LOCAL_LLM_MODEL", DEFAULT_MODEL_ID)
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


def gerar_texto(prompt_usuario: str, max_new_tokens: int = 512) -> str:
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
