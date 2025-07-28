import chromadb
from sentence_transformers import SentenceTransformer
import html
from dotenv import load_dotenv
import os
from groq import Groq
import base64

load_dotenv()

def analyze_image_with_groq(image_path):
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return "Error: GROQ_API_KEY not set."

        client = Groq(api_key=api_key)

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        image_url = f"data:image/png;base64,{base64_image}"

        prompt_text = "Describe this image in detail. If it is a diagram or chart, explain its components, the relationships between them, and the process it illustrates."

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        
        if chat_completion.choices:
            return chat_completion.choices[0].message.content
        else:
            return "VLM analysis failed. No description generated."

    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}"
    except Exception as e:
        return f"Error during Groq vision API call: {e}"


def generate_llm_response(query, context):
    """
    Sends the query and context to the Groq LLM and streams the response.
    """
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            yield "Error: GROQ_API_KEY not set."
            return

        client = Groq(api_key=api_key)
        
        system_prompt = """You are an expert-level AI assistant trained to help students understand and explain complex research papers clearly and accurately. 
        Based on the retrieved context from the vector database, answer the following question in an exam-style format:

        Question: "query"
        Follow these instructions:

        Strictly base your answer on the retrieved context, but rewrite in your own words to sound like a knowledgeable student.

        Use precise terminology from the source paper (especially if it's a technical concept).

        If mathematical expressions or algorithms are involved, include them concisely using inline math or pseudocode.

        Avoid vague generalizations — focus on clarity, technical correctness, and relevance to the specific paper.

        If the question asks "what", "why", or "how", answer with structure and depth, not fluff.

        Keep the answer exam-appropriate and ~150 words, unless the query explicitly asks for more.

        Your goal: provide a concise, 10/10 academic answer that reflects deep understanding of the original research.""" 
        user_prompt = f"CONTEXT:\n---\n{context}\n---\n\nQUESTION:\n{query}"

        stream = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"Error calling Groq API: {e}"

if __name__ == "__main__":
    print("--- Starting Query Phase ---")
    
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = client.get_collection(name="pdf_multimodal_embeddings")
        print(f"Successfully connected to collection with {collection.count()} items.")
    except ValueError:
        print("Error: Database not found. Please run ingest.py first.")
        exit()

    embedding_model = SentenceTransformer('clip-ViT-B-32')

    while True:
        query_text = input("\nPlease enter your query about the PDF (or type 'quit' to exit): ")
        
        # Check if the user wants to exit
        if query_text.lower() in ['quit', 'exit']:
            print("Exiting program. Goodbye!")
            break

        print(f"\nUser Query: '{query_text}'\n")

        print("Retrieving context from database...")
        query_embedding = embedding_model.encode([query_text]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=10)
        
        print("Formatting context and analyzing images with Hugging Face API...")
        context_parts = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]
            
            content_to_add = ""
            if metadata['type'] == 'image':
                print(f"  > Analyzing image: {document}...")
                image_description = analyze_image_with_groq(document)
                content_to_add = f"Source: Image Description\nContent: {image_description}"
            elif metadata['type'] == 'table':
                unescaped_doc = html.unescape(document)
                readable_table = unescaped_doc.replace('<br>', '\n')
                content_to_add = f"Source: Table\nContent:\n{readable_table}"
            else: # Text
                content_to_add = f"Source: Text Chunk\nContent: {document}"
            
            context_parts.append(content_to_add)

        formatted_context = "\n---\n".join(context_parts)
        print("Context retrieved and formatted.\n")

        print("--- Calling Groq LLM & Streaming Response ---")
        for chunk in generate_llm_response(query_text, formatted_context):
            print(chunk, end="", flush=True)
        print("\n" + "="*50) 