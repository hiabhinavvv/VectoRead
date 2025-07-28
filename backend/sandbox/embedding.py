import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
from typing import List
import os

PDF_PATH = "./data/attention.pdf"
COLLECTION_NAME = "pdf_documents"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_and_process_data(pdf_path: str) -> List[str]:
    text_chunks = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
                text_chunks.extend(chunks)
    return text_chunks

def create_chroma_collection(chunks: List[str]):
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name = MODEL_NAME
    )

    collection = chroma_client.get_or_create_collection(
        name = COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )

    ids = [f"id_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids = ids
    )

    print(f"successfully stored {len(chunks)} chunks in chromadb")
    return collection

if __name__=="__main__":
    print(f"extracting feature from {PDF_PATH}...")
    text_chunks = load_and_process_data(PDF_PATH)
    print(f"extracted {len(text_chunks)} text chunks")

    print("creating chromadb...")
    collection = create_chroma_collection(text_chunks)

    results = collection.query(
        query_texts=["sample search query"],
        n_results = 3
    )
    print("/nsample query results: ")
    for doc in results['documents'][0]:
        print(f"- {doc[:100]}...")