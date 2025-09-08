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

# --- Setup ---
load_dotenv()

# --- Part 1: Data Ingestion Pipeline ---

def extract_content_from_pdf(file_content: bytes, min_image_size: int = 100):
    """
    Extracts content from a PDF with robust, filtered image extraction.
    """
    doc = fitz.open(stream=file_content, filetype="pdf")
    full_text, images, tables = "", [], []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text() + "\n"
        
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                if not base_image or "image" not in base_image or not base_image["image"]:
                    continue
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))

                if image.width < min_image_size or image.height < min_image_size:
                    continue
                images.append((image, page_num))
            except Exception as e:
                print(f"WARNING: Skipping a problematic image on page {page_num}. Error: {e}")
                continue
        
        for table in page.find_tables():
            tables.append((table.to_markdown(clean=True), page_num))
            
    return full_text, images, tables

def generate_embeddings(text_chunks, images, tables, text_model, image_model):
    """
    Generates embeddings using separate models for text/tables and images.
    """
    # Use the text-specific model for text and tables
    text_embeddings = text_model.encode(text_chunks) if text_chunks else np.array([])
    table_markdowns = [tbl for tbl, _ in tables]
    table_embeddings = text_model.encode(table_markdowns) if table_markdowns else np.array([])

    # Use the CLIP model (image_model) only for images
    image_objects = [img.convert("RGB") for img, _ in images]
    image_embeddings = image_model.encode(image_objects) if image_objects else np.array([])
        
    return text_embeddings, image_embeddings, table_embeddings

def store_in_chromadb(session_id: str, text_chunks, text_embeddings, images, image_embeddings, tables, table_embeddings):
    """
    Stores documents and their embeddings in a ChromaDB collection.
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=session_id)
    image_dir = f"/tmp/extracted_images/{session_id}"
    os.makedirs(image_dir, exist_ok=True)

    ids, embeddings_list, documents, metadatas = [], [], [], []
    
    # Process and add text chunks
    for i, chunk in enumerate(text_chunks):
        ids.append(f"text_chunk_{i}")
        embeddings_list.append(text_embeddings[i].tolist())
        documents.append(chunk)
        metadatas.append({'type': 'text'})

    # Process and add images
    for i, (image, page_num) in enumerate(images):
        try:
            image_id = f"image_{i}"
            image_path = os.path.join(image_dir, f"{image_id}.png")
            image.save(image_path, 'PNG')
            ids.append(image_id)
            if image_embeddings.size > 0:
                embeddings_list.append(image_embeddings[i].tolist())
            documents.append(image_path)
            metadatas.append({'type': 'image', 'page': page_num})
        except Exception as e:
            print(f"WARNING: Skipping a problematic image on page {page_num} during save. Error: {e}")

    # Process and add tables
    for i, (table_markdown, page_num) in enumerate(tables):
        ids.append(f"table_{i}")
        embeddings_list.append(table_embeddings[i].tolist())
        documents.append(table_markdown)
        metadatas.append({'type': 'table', 'page': page_num})

    if ids:
        collection.add(ids=ids, embeddings=embeddings_list, documents=documents, metadatas=metadatas)
    
    return collection.count()

def process_and_store_pdf(session_id: str, file_content: bytes, text_embedding_model, image_embedding_model):
    """
    Orchestrates the full ingestion pipeline: extract, chunk, deduplicate, embed, and store.
    """
    print("--- Starting PDF Ingestion ---")
    # 1. Extract content
    full_text, images, tables = extract_content_from_pdf(file_content)
    print(f"Extracted {len(images)} images and {len(tables)} tables.")

    # 2. Chunk and deduplicate text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(full_text)
    print(f"Original number of text chunks: {len(text_chunks)}")
    unique_chunks = list(set(text_chunks))
    print(f"Number of unique text chunks after deduplication: {len(unique_chunks)}")

    # 3. Generate embeddings
    text_embeds, image_embeds, table_embeds = generate_embeddings(
        unique_chunks, images, tables, text_embedding_model, image_embedding_model
    )

    # 4. Store in ChromaDB
    count = store_in_chromadb(
        session_id, unique_chunks, text_embeds, images, image_embeds, tables, table_embeds
    )
    print(f"--- Successfully stored {count} items for session '{session_id}' ---")
    return count


# --- Part 2: Querying Pipeline ---

def analyze_image_with_groq(image_path: str, groq_client: Groq):
    """
    Generates a description of an image using Groq's vision model.
    """
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        image_url = f"data:image/png;base64,{base64_image}"
        prompt = "Describe this image in detail. If it's a diagram, explain its components, relationships, and the process it illustrates."
        
        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192", # Update to a model that supports vision if available
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}]
        )
        return completion.choices[0].message.content if completion.choices else "VLM analysis failed."
    except Exception as e:
        return f"Error during Groq vision call: {e}"

def process_query_and_generate(query: str, session_id: str, text_embedding_model, groq_client):
    """
    Processes a user query, retrieves context from ChromaDB, and generates a response.
    """
    print("\n--- Processing Query ---")
    # 1. Connect to the existing ChromaDB collection
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name=session_id)
    except Exception as e:
        yield f"Error: Could not find a database for session '{session_id}'. Please upload a document first. Details: {e}"
        return

    # 2. Embed the user's query using the TEXT model
    print("Embedding user query...")
    query_embedding = text_embedding_model.encode([query]).tolist()
    
    # 3. Retrieve relevant documents from ChromaDB
    print("Retrieving context...")
    results = collection.query(query_embeddings=query_embedding, n_results=10)
    
    context_parts = []
    if 'ids' in results and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            doc_id, metadata, document = results['ids'][0][i], results['metadatas'][0][i], results['documents'][0][i]
            
            if metadata['type'] == 'image':
                print(f"  > Analyzing retrieved image: {document}...")
                desc = analyze_image_with_groq(document, groq_client)
                context_parts.append(f"Source: {doc_id}\nContent: {desc}")
            else:
                context_parts.append(f"Source: {doc_id}\nContent: {document}")
    
    if not context_parts:
        yield "Could not find relevant context to answer the question."
        return
        
    formatted_context = "\n---\n".join(context_parts)
    
    # 4. Generate the final response with the LLM
    system_prompt = "You are an expert AI assistant. Answer the user's question based ONLY on the provided context, which includes text, tables, and image descriptions. Cite your sources using the format [Source: source_id]."
    user_prompt = f"CONTEXT:\n---\n{formatted_context}\n---\n\nQUESTION:\n{query}"
    
    print("Generating final answer...")
    try:
        stream = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=2048,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Error calling Groq API: {e}"