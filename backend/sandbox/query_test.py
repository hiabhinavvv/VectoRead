from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import os
from embedding import load_and_process_data, create_chroma_collection

load_dotenv()
COLLECTION_NAME = "pdf_documents"
MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

def setup_chroma_connection():
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )
    return chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )

def setup_groq_chain():
    prompt_template = """You are an advanced AI research assistant trained to assist in academic and scientific inquiry. You have access to retrieved information from a vector database based on a user’s query. Your task is two-fold:

Summarize or synthesize the retrieved content (provided below) clearly and concisely, preserving its technical accuracy and context.

Augment the response with relevant scholarly insights, background theory, related concepts, or supporting evidence from your training knowledge—but only if they are directly relevant to the topic.

Maintain a formal, academic tone throughout. Avoid opinions, speculation, or conversational style. Support any additional information with theoretical or empirical grounding where possible. Do not include citations unless explicitly asked to.

User Query:
{question}

Retrieved Chunks from Vector DB:

Copy
Edit
{context}
Now, write a consolidated research-oriented response that addresses the query using the retrieved content and your academic knowledge."""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    return (
        prompt
        | ChatGroq(
            temperature=0.4,
            model_name = GROQ_MODEL,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        | StrOutputParser()
    )

def retrieve_context(collection, query: str, k: int = 3):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return "/n/n".join([
        f"document excerpt {i+1}:\n{chunk}"
        for i, chunk in enumerate(results['documents'][0])
    ])

def main():
    collection = setup_chroma_connection()
    rag_chain = setup_groq_chain()
    if collection.count() == 0:
        print("Collection is empty. Run 'python embed.py' first.")
        exit(1)

    print("rag system ready. type 'exit' to quit.\n")

    while True:
        query = input("your question bout the document: ")
        if query.lower() in ('exit', 'quit'):
            break

        context = retrieve_context(collection, query)
        response = rag_chain.invoke({
            "context": context,
            "question": query
        })

        print("\n=== relevant context ===")
        print(context[:500] + "..." if len(context) > 500 else context)

        print("\n=== Answer ===")
        print(response + "\n")

if __name__ == "__main__":
    main()