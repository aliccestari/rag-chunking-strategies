from langchain_chroma.vectorstores import Chroma # para vetorizacao
from langchain_google_genai import GoogleGenerativeAIEmbeddings # embedding, precisa de chave de API
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


CAMINHO_DB = "db"

# texto de prompt
prompt_template = """ 
Responda a pergunta do usuario: {pergunta}
com base nessas informacoes: {base_conhecimento}

Se voce nao encontrar a resposta para a pergunta do usuario nessas informacoes, responda: nao sei a responder a sua pergunta"""


def perguntar():
    pergunta = input("Escreva sua pergunta: ")

    #carregar o banco de dados

    funcao_embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

    # comparar a pergunta do usuario (embbeding) com o meu bando de dados
    resultados = db.similarity_search_with_relevance_scores(pergunta, k=4) # nota entre 0 e 1, parametro K diz quantas respostas ele retorna, chunks para passar ao contexto
    if len(resultados) == 0 or resultados[0][1] < 0.6:
        print("Não conseguiu encontrar alguma informação relevante na base")
        return #interrompe a procura pois nao achou nada relevante

    textos_resultado = []
    for resultado in resultados:
        texto = resultado[0].page_content
        textos_resultado.append(texto)

    base_conhecimento = "\n\n----\n\n".join(textos_resultado)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"pergunta": pergunta, "base_conhecimento": base_conhecimento})

    modelo = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    texto_resposta = modelo.invoke(prompt).content
    print("Resposta da IA: ", texto_resposta)

perguntar()