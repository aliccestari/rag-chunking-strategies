import os

from langchain_chroma.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from corpus_config import pasta_indice_chroma
from embeddings_config import criar_funcao_embedding
from local_llm import gerar_texto
from rag_prompts import RAG_ANSWER_PROMPT

load_dotenv()


CAMINHO_DB = pasta_indice_chroma()

LIMIAR_RELEVANCIA = 0.25

# Mais trechos ajudam quando cada chunk é fatia de uma FAQ longa.
TOP_K = 6

# Quantos caracteres do início de cada chunk mostrar no debug (0 = não mostrar).
PREVIEW_CHARS = 220


def perguntar():
    pergunta = input("Escreva sua pergunta: ").strip()
    if not pergunta:
        print("Pergunta vazia.")
        return

    funcao_embedding = criar_funcao_embedding()
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

    resultados = db.similarity_search_with_relevance_scores(pergunta, k=TOP_K)
    if len(resultados) == 0 or resultados[0][1] < LIMIAR_RELEVANCIA:
        print("Não foi encontrado contexto acima do limiar de relevância.")
        return

    print("\n--- Passagens recuperadas (document_id = metadata id; útil para Subtask A / qrels) ---")
    blocos_ctx: list[str] = []
    for rank, (doc, score) in enumerate(resultados, start=1):
        doc_id = doc.metadata.get("id", "(sem id no metadata)")
        print(f"  [{rank}] id={doc_id!r}  score={score:.4f}")
        if PREVIEW_CHARS > 0:
            corpo = doc.page_content.replace("\n", " ").strip()
            trecho = corpo[:PREVIEW_CHARS] + ("…" if len(corpo) > PREVIEW_CHARS else "")
            print(f"      preview: {trecho}")
        pid = doc.metadata.get("id", "") or ""
        tit = doc.metadata.get("title", "") or ""
        prefix = f"[{pid}]" + (f" {tit}" if tit else "")
        blocos_ctx.append(f"{prefix}\n{doc.page_content}")
    print("--- fim da lista ---\n")

    base_conhecimento = "\n\n----\n\n".join(blocos_ctx)
    historico = "N/A (pergunta única no terminal)."
    prompt_txt = RAG_ANSWER_PROMPT.format(
        historico=historico,
        base_conhecimento=base_conhecimento,
        pergunta=pergunta,
    )

    if os.environ.get("USE_GEMINI", "").strip() in ("1", "true", "yes"):
        modelo = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        texto_resposta = modelo.invoke([HumanMessage(content=prompt_txt)]).content
    else:
        texto_resposta = gerar_texto(prompt_txt)
    print("Resposta da IA:", texto_resposta)


if __name__ == "__main__":
    perguntar()
