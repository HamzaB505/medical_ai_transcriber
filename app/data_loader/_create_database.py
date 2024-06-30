import os
import shutil
import argparse
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from llm._get_embedding_func import get_embedding_function
from tqdm import tqdm

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    return chunks

def create_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # if the page ID is the same as the last one, increment the index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def add_to_chroma(chunks: list[Document],
                  embedding_function,
                  chroma_path):

    print('starting embedding')

    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_function
    )
    print("finished embedding")
    chunks_with_ids = create_chunk_ids(chunks)

    #Add or Update the documents
    existing_items = db.get(include=[]) #IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    
    # only add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        for i, chunk in tqdm(enumerate(new_chunks)):
                print(f"Adding document {i+1}/{len(new_chunks)}")
                db.add_documents([chunk], ids=[new_chunk_ids[i]])
        print("Persisting changes...")
        db.persist()
        print("Changes persisted.")
    else:
        print("✅ No new documents to add")

def clear_database(CHROMA_PATH):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def save_to_chroma(chunks: list[Document],
                   CHROMA_PATH):
    # clears out if the database exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    # create a new embeddings DB from the documents
    db = Chroma.from_documents(
        documents=chunks,
        embedding_function=get_embedding_function(),
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f'Saved {len(chunks)} chunks to {CHROMA_PATH}')

