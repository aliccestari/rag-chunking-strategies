"""Templates de prompt alinhados ao MTRAG (inglês nas tasks; resposta no idioma da pergunta)."""

RAG_ANSWER_PROMPT = """You are a helpful assistant for question answering over noisy web/FAQ snippets.

Instructions:
- Answer in the same language as the user's question.
- Base your answer on the context below: paraphrase and combine relevant sentences. Ignore navigation menus, "Skip to main content", search boxes, and other boilerplate.
- Do not invent specific facts, numbers, or claims that are not supported by the context.
- If the context has no information related to the question at all, reply exactly with: I cannot answer from the given context.

Conversation so far (may be a single turn):
{historico}

Context:
{base_conhecimento}

Current user question:
{pergunta}

Answer:"""
