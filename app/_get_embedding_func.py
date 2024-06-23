from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    llm = OllamaEmbeddings(model='mistral')
    return llm
