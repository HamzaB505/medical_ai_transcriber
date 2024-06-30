import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from llm import get_embedding_function, query_llm
import logging

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}
---

Answer the question based on the above context: {question}
"""

logger = logging.getLogger(__name__)


def query_rag(query_text: str,
              embedding_function,
              CHROMA_PATH):
    # Prepare the DB.

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    logger.info("Searching for TOP 5 best results.....")
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    logger.info("Prompt prepared !")
    # print(prompt)

    # Invoking model
    logger.info("Invoking model")
    output = query_llm({
	"inputs": prompt,
    })
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"""Response: {output[0]["generated_text"]} \n\n Sources: {sources}"""
    print("##################### Answer ################")
    print(formatted_response)

    return formatted_response

