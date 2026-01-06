import base64
import fitz
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import numpy as np
from PIL import Image
import io
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def extract_content_from_pdf(file_content: bytes, min_image_size: int = 100):
    doc = fitz.open(stream=file_content, filetype="pdf")
    page_texts, images, tables = [], [], []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_texts.append((page.get_text(), page_num))
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
    return page_texts, images, tables

def generate_embeddings(text_chunks, images, tables, text_model, image_model):
    text_embeddings = text_model.encode(text_chunks) if text_chunks else np.array([])
    table_markdowns = [tbl for tbl, _ in tables]
    table_embeddings = text_model.encode(table_markdowns) if table_markdowns else np.array([])
    image_objects = [img.convert("RGB") for img, _ in images]
    image_embeddings = image_model.encode(image_objects) if image_objects else np.array([])
    return text_embeddings, image_embeddings, table_embeddings

def store_in_chromadb(session_id: str, text_chunks, text_embeddings, images, image_embeddings, tables, table_embeddings, text_metadatas):
    client = chromadb.PersistentClient(path="./chroma_db")
    image_dir = f"/tmp/extracted_images/{session_id}"
    os.makedirs(image_dir, exist_ok=True)
    
    total_items = 0
    if len(text_chunks) > 0:
            collection_text = client.get_or_create_collection(name=f"{session_id}_text")
            ids_text = [f"text_chunk_{i}" for i in range(len(text_chunks))]
            embeddings_list = [emb.tolist() for emb in text_embeddings]
            
            collection_text.add(
                ids=ids_text, 
                embeddings=embeddings_list, 
                documents=text_chunks, 
                metadatas=text_metadatas
            )
            total_items += collection_text.count()
            print(f"Stored {len(text_chunks)} text chunks.")

    if len(tables) > 0:
        collection_tables = client.get_or_create_collection(name=f"{session_id}_tables")
        ids_table = []
        docs_table = []
        metadatas_table = []
        embeddings_table_list = [emb.tolist() for emb in table_embeddings]

        for i, (table_markdown, page_num) in enumerate(tables):
            ids_table.append(f"table_{i}")
            docs_table.append(table_markdown)
            metadatas_table.append({'type': 'table', 'page': page_num})
            
        collection_tables.add(
            ids=ids_table, 
            embeddings=embeddings_table_list, 
            documents=docs_table, 
            metadatas=metadatas_table
        )
        total_items += collection_tables.count()
        print(f"Stored {len(tables)} tables.")

    if len(images) > 0:
            collection_images = client.get_or_create_collection(name=f"{session_id}_images")
            ids_img, embeddings_img, docs_img, metadatas_img = [], [], [], []
            
            for i, (image, page_num) in enumerate(images):
                try:
                    image_id = f"image_{i}"
                    image_path = os.path.join(image_dir, f"{image_id}.png")
                    image.save(image_path, 'PNG')
                    
                    ids_img.append(image_id)
                    embeddings_img.append(image_embeddings[i].tolist())
                    docs_img.append(image_path) # Storing path as document
                    metadatas_img.append({'type': 'image', 'page': page_num})
                except Exception as e:
                    print(f"WARNING: Skipping image save on page {page_num}. Error: {e}")
            
            if ids_img:
                collection_images.add(
                    ids=ids_img, 
                    embeddings=embeddings_img, 
                    documents=docs_img, 
                    metadatas=metadatas_img
                )
                total_items += collection_images.count()
                print(f"Stored {len(ids_img)} images.")
            
            return total_items

def process_and_store_pdf(session_id: str, file_content: bytes, text_embedding_model, image_embedding_model):
    print("--- Starting PDF Ingestion (Dual Collection) ---")
    page_texts, images, tables = extract_content_from_pdf(file_content)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_text_chunks = []
    all_text_metadatas = []

    for text_content, page_num in page_texts:
        chunks = text_splitter.split_text(text_content)
        for chunk in chunks:
            all_text_chunks.append(chunk)
            all_text_metadatas.append({'type': 'text', 'page': page_num})
    
    text_embeds, image_embeds, table_embeds = generate_embeddings(
        all_text_chunks, images, tables, text_embedding_model, image_embedding_model
    )
    
    count = store_in_chromadb(
        session_id, all_text_chunks, text_embeds, images, image_embeds, tables, table_embeds, all_text_metadatas
    )
    print(f"--- Successfully stored {count} items across collections for session '{session_id}' ---")
    return count


def analyze_image_with_groq(image_path: str, groq_client: Groq):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        image_url = f"data:image/png;base64,{encoded_image}"
        prompt = "Describe this image in detail. If it's a diagram, explain its components, relationships, and the process it illustrates."
        
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt}, 
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        )
        return completion.choices[0].message.content if completion.choices else "VLM analysis failed."
    except Exception as e:
        return f"Error during Groq vision call: {e}"


def process_query_and_generate(query: str, session_id: str, text_embedding_model, image_embedding_model, groq_client):
    print("\n--- Processing Query (Dual Collection) ---")
    client = chromadb.PersistentClient(path="./chroma_db")
    context_parts = []
    query_embedding_text = text_embedding_model.encode([query]).tolist()
    query_embedding_image = image_embedding_model.encode([query]).tolist()
    
    try:
        collection_text = client.get_collection(name=f"{session_id}_text")
        results_text = collection_text.query(
            query_embeddings=query_embedding_text, 
            n_results=5, 
            include=["metadatas", "documents"]
        )
        if results_text['ids'] and results_text['ids'][0]:
            for i in range(len(results_text['ids'][0])):
                doc = results_text['documents'][0][i]
                meta = results_text['metadatas'][0][i]
                citation = f"Source: Page {meta.get('page', '?') + 1} (text)"
                context_parts.append(f"{citation}\nContent: {doc}")
    except Exception as e:
        print(f"Text retrieval error: {e}")

    # --- B. Retrieve Tables (NEW) ---
    try:
        collection_tables = client.get_collection(name=f"{session_id}_tables")
        # We use the SAME query_embedding_text because tables were embedded with the text model
        results_tables = collection_tables.query(
            query_embeddings=query_embedding_text, 
            n_results=3, # Usually need fewer tables than text chunks
            include=["metadatas", "documents"]
        )
        if results_tables['ids'] and results_tables['ids'][0]:
            for i in range(len(results_tables['ids'][0])):
                doc = results_tables['documents'][0][i]
                meta = results_tables['metadatas'][0][i]
                citation = f"Source: Page {meta.get('page', '?') + 1} (table)"
                context_parts.append(f"{citation}\nContent: {doc}")
    except Exception as e:
        print(f"Table retrieval error (or collection empty): {e}")

    # --- C. Retrieve Images ---
    try:
        collection_images = client.get_collection(name=f"{session_id}_images")
        results_images = collection_images.query(
            query_embeddings=query_embedding_image, 
            n_results=3, 
            include=["metadatas", "documents"]
        )
        if results_images['ids'] and results_images['ids'][0]:
            for i in range(len(results_images['ids'][0])):
                path = results_images['documents'][0][i]
                meta = results_images["metadatas"][0][i]
                page_num = meta.get('page', -1) + 1
                
                print(f"  > Analyzing retrieved image: {path} (from Page {page_num})...")
                desc = analyze_image_with_groq(path, groq_client)
                
                citation = f"Source: Page {page_num} (image)"
                context_parts.append(f"{citation}\nContent: {desc}")
    except Exception as e:
        print(f"Image retrieval error (or collection empty): {e}")
        
    if not context_parts:
        yield "Could not find relevant context to answer the question."
        return
        
    formatted_context = "\n---\n".join(context_parts)
    system_prompt = """You are a highly intelligent expert AI assistant. Your primary purpose is to analyze and synthesize information from a provided context to answer a user's question with depth, clarity, and precision.

Follow these instructions meticulously:

1.  **Comprehensive Analysis:** Your answer must be based *only* on the provided context, which may include text chunks, tables, and detailed descriptions of images or diagrams. Synthesize information from all relevant sources to form a complete picture.

2.  **Expert Tone:** Rewrite the information in your own words to sound like a subject-matter expert. Use precise terminology found in the context, but explain it clearly.

3.  **Data-Rich Responses:** If the context contains data, numbers, or specific examples, you must include them in your answer to support your claims. If there are formulas or code, represent them accurately.

4.  **Structured and Deep Answers:** Avoid vague or superficial responses. If the question asks "what," "why," or "how," provide a well-structured answer with logical flow and sufficient detail. Do not add fluff or filler.

5.  **Cite Your Sources:** After every key piece of information, you MUST cite its source using the format [Source: Page X (type)]. For example: [Source: Page 5 (text)] or [Source: Page 12 (image)]. This is a critical requirement.

Your goal is to act as a world-class analyst, providing answers that are not only correct but also insightful, well-supported, and directly derived from the source material.""" 
    user_prompt = f"CONTEXT:\n---\n{formatted_context}\n---\n\nQUESTION:\n{query}"
    
    try:
        stream = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model="llama-3.3-70b-versatile",
            temperature= 0.75,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Error calling Groq API: {e}"