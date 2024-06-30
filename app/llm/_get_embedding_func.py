from langchain_community.embeddings.ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "all-mpnet-base-v2"

def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    return embeddings
