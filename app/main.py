import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from llm import get_embedding_function, query_llm
from query import query_rag
import logging

logger = logging.getLogger(__name__)
CHROMA_PATH = "../chroma"

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    logger.info('Query parsed !')
    embedding_function = get_embedding_function()
    logger.info('Embedding function prepared!')
    query_rag(query_text, embedding_function, CHROMA_PATH=CHROMA_PATH)
    logger.info('querying finished !')
    

if __name__ == "__main__":
    main()