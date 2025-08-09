import fitz
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import numpy as np
from PIL import Image
import io
import os
import html
import base64
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


embedding_model = None
collection = None
groq_client = None



def extract_content_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    text, images, tables = "", [], []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"]))
            images.append((image, page_num))
        for table in page.find_tables():
            tables.append((table.to_markdown(clean=True), page_num))
    return text, images, tables

def generate_embeddings(text_chunks, images, tables, model):
    text_embeddings = model.encode(text_chunks)
    image_objects = [img for img, _ in images]
    image_embeddings = model.encode(image_objects) if image_objects else np.array([])
    table_markdowns = [tbl for tbl, _ in tables]
    table_embeddings = model.encode(table_markdowns) if table_markdowns else np.array([])
    return text_embeddings, image_embeddings, table_embeddings

def store_in_chromadb(text_chunks, text_embeddings, images, image_embeddings, tables, table_embeddings):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="pdf_multimodal_embeddings")
    image_dir = "extracted_images"
    os.makedirs(image_dir, exist_ok=True)

    ids, embeddings_list, documents, metadatas = [], [], [], []
    
    for i, chunk in enumerate(text_chunks):
        ids.append(f"text_chunk_{i}")
        embeddings_list.append(text_embeddings[i].tolist())
        documents.append(chunk)
        metadatas.append({'type': 'text'})

    for i, (image, page_num) in enumerate(images):
        image_id = f"image_{i}"
        image_path = os.path.join(image_dir, f"{image_id}.png")
        image.save(image_path, 'PNG')
        ids.append(image_id)
        if image_embeddings.size > 0:
            embeddings_list.append(image_embeddings[i].tolist())
        documents.append(image_path)
        metadatas.append({'type': 'image', 'page': page_num})
        
    for i, (table_markdown, page_num) in enumerate(tables):
        ids.append(f"table_{i}")
        embeddings_list.append(table_embeddings[i].tolist())
        documents.append(table_markdown)
        metadatas.append({'type': 'table', 'page': page_num})

    if ids:
        collection.upsert(ids=ids, embeddings=embeddings_list, documents=documents, metadatas=metadatas)
    return collection.count()

def load_query_models():
    global embedding_model, collection, groq_client
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('clip-ViT-B-32')
    if collection is None:
        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path="./chroma_db")
        try:
            collection = client.get_collection(name="pdf_multimodal_embeddings")
            print(f"Connected to collection with {collection.count()} items.")
        except Exception:
            print("Collection not found. It will be created upon first ingestion.")
            collection = None 
            
    if groq_client is None:
        print("Initializing Groq client...")
        groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("All query models and clients are loaded.")

def analyze_image_with_groq(image_path: str):
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        image_url = f"data:image/png;base64,{base64_image}"
        prompt = "Describe this image in detail. If it's a diagram, explain its components, relationships, and the process it illustrates."
        
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}]
        )
        return completion.choices[0].message.content if completion.choices else "VLM analysis failed."
    except Exception as e:
        return f"Error during Groq vision call: {e}"

def process_query_and_generate(query: str):
    """Main RAG function: retrieves context and yields streaming LLM response."""
    if not all([collection, embedding_model, groq_client]):
        yield "Error: Models or DB not loaded. Please check server startup logs or ingest a document first."
        return

    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=10)
    
    context_parts = []
    if 'ids' in results and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            metadata, document = results['metadatas'][0][i], results['documents'][0][i]
            if metadata['type'] == 'image':
                print(f"  > Analyzing image: {document}...")
                desc = analyze_image_with_groq(document)
                context_parts.append(f"Source: Image Description\nContent: {desc}")
            elif metadata['type'] == 'table':
                table = html.unescape(document).replace('<br>', '\n')
                context_parts.append(f"Source: Table\nContent:\n{table}")
            else:
                context_parts.append(f"Source: Text Chunk\nContent: {document}")
    
    formatted_context = "\n---\n".join(context_parts)
    
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
    user_prompt = f"CONTEXT:\n---\n{formatted_context}\n---\n\nQUESTION:\n{query}"
    
    try:
        stream = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Error calling Groq API: {e}"