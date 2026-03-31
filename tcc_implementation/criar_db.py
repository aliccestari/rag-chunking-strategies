from langchain_community.document_loaders import PyPDFDirectoryLoader # consegue transformar pdf em texto, parseando
from langchain_text_splitters import RecursiveCharacterTextSplitter # vai fazer a divisao em chunks
from langchain_chroma.vectorstores import Chroma # para vetorizacao
from langchain_google_genai import GoogleGenerativeAIEmbeddings # embedding, precisa de chave de API
from dotenv import load_dotenv

load_dotenv()

# ESSE PROCESSO DE CRIACAO DA BASE DE DADOS ACONTECE UMA UNICA VEZ

PASTA_BASE = "base"

def criar_db():
    # 1. carregar_documentos
    documentos = carregar_documentos() # lista com todos docs
    # print(documentos)
    # 2. dividir os documentos em pedaços/chunks
    chunks = dividir_chunks(documentos)
    # 3. vetorizar os chunks com o processo de embedding (esse processo le o texto e transformar ele em numeros, precisa de uma IA e deve ser a mesma usada na transformacao da pergunta do usuario em numeros)
    vetorizar_chunks(chunks)

def carregar_documentos():
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
    documentos = carregador.load()
    return documentos

# chunk_size -> quantidade de caracteres
# chunk_overlap -> quantos caracteres podem estar sobrepostos entre 2 chunks diferentes
# length_function -> quantos caracteres tem um texto
# add_start_index -> a partir de qual indice ta comecando o chunk que estou analisando
def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(documentos)
    print(len(chunks))
    return chunks

def vetorizar_chunks(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    db = Chroma.from_documents(chunks, embeddings, persist_directory="db")
    print("Banco de dados criado")

criar_db()
