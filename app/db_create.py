
import argparse
from llm._get_embedding_func import get_embedding_function
from data_loader import (load_documents,
                         clear_database,
                         split_documents,
                         add_to_chroma) 


CHROMA_PATH = "../chroma"
DATA_PATH = "../data"

def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database(CHROMA_PATH=CHROMA_PATH)

    # Create (or update) the data store.
    documents = load_documents(DATA_PATH=DATA_PATH)
    chunks = split_documents(documents)
    embedding_function = get_embedding_function()
    add_to_chroma(chunks,
                  embedding_function,
                  chroma_path=CHROMA_PATH)

if __name__ == "__main__":
    main()