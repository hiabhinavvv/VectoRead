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


def extract_content_from_pdf(file_content: bytes):
    doc = fitz.open(stream=file_content, filetype="pdf")
    
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
    image_objects = [img.convert("RGB") for img, _ in images]
    image_embeddings = model.encode(image_objects) if image_objects else np.array([])
    table_markdowns = [tbl for tbl, _ in tables]
    table_embeddings = model.encode(table_markdowns) if table_markdowns else np.array([])
    return text_embeddings, image_embeddings, table_embeddings

def store_in_chromadb(session_id: str, text_chunks, text_embeddings, images, image_embeddings, tables, table_embeddings):
    client = chromadb.PersistentClient(path="./chroma_db")
    
    collection = client.get_or_create_collection(name=session_id)
    
    image_dir = "/tmp/extracted_images"
    os.makedirs(image_dir, exist_ok=True)

    ids, embeddings_list, documents, metadatas = [], [], [], []
    
    for i, chunk in enumerate(text_chunks):
        ids.append(f"text_chunk_{i}")
        embeddings_list.append(text_embeddings[i].tolist())
        documents.append(chunk)
        metadatas.append({'type': 'text'})

    for i, (image, page_num) in enumerate(images):
        try:
            image_id = f"image_{i}"
            image_path = os.path.join(image_dir, f"{image_id}.png")
            
            # Ensure image is valid before saving
            if image.width > 0 and image.height > 0:
                image.save(image_path, 'PNG')
                ids.append(image_id)
                if image_embeddings.size > 0:
                    embeddings_list.append(image_embeddings[i].tolist())
                documents.append(image_path)
                metadatas.append({'type': 'image', 'page': page_num})
        except Exception as e:
            # If a single image fails, log the error and continue
            print(f"WARNING: Skipping a problematic image on page {page_num}. Error: {e}")
        
    for i, (table_markdown, page_num) in enumerate(tables):
        ids.append(f"table_{i}")
        embeddings_list.append(table_embeddings[i].tolist())
        documents.append(table_markdown)
        metadatas.append({'type': 'table', 'page': page_num})

    if ids:
        collection.add(ids=ids, embeddings=embeddings_list, documents=documents, metadatas=metadatas)
    
    return collection.count()

def load_query_models():
    global embedding_model, collection, groq_client
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('clip-ViT-B-32') 
            
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

def process_query_and_generate(query: str, session_id: str):
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        session_collection = client.get_collection(name=session_id)
    except Exception as e:
        yield f"Error: Could not find a database for the provided session. Please upload a document first. Details: {e}"
        return

    if not all([session_collection, embedding_model, groq_client]):
        yield "Error: Models not loaded correctly. Please check server startup logs."
        return

    query_embedding = embedding_model.encode([query]).tolist()
    
    results = session_collection.query(query_embeddings=query_embedding, n_results=10)
    
    context_parts = []
    if 'ids' in results and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            metadata, document = results['metadatas'][0][i], results['documents'][0][i]
            if metadata['type'] == 'image':
                print(f"  > Analyzing image: {document}...")
                desc = analyze_image_with_groq(document)
                context_parts.append(f"Source: {doc_id}\nContent: {desc}")
            elif metadata['type'] == 'table':
                table = html.unescape(document).replace('<br>', '\n')
                context_parts.append(f"Source: {doc_id}\nContent:\n{table}")
            else:
                context_parts.append(f"Source: {doc_id}\nContent: {document}")
    
    formatted_context = "\n---\n".join(context_parts)
    
    system_prompt = """You are a highly intelligent expert AI assistant. Your primary purpose is to analyze and synthesize information from a provided context to answer a user's question with depth, clarity, and precision.

Follow these instructions meticulously:

1.  **Comprehensive Analysis:** Your answer must be based *only* on the provided context, which may include text chunks, tables, and detailed descriptions of images or diagrams. Synthesize information from all relevant sources to form a complete picture.

2.  **Expert Tone:** Rewrite the information in your own words to sound like a subject-matter expert. Use precise terminology found in the context, but explain it clearly.

3.  **Data-Rich Responses:** If the context contains data, numbers, or specific examples, you must include them in your answer to support your claims. If there are formulas or code, represent them accurately.

4.  **Structured and Deep Answers:** Avoid vague or superficial responses. If the question asks "what," "why," or "how," provide a well-structured answer with logical flow and sufficient detail. Do not add fluff or filler.

5.  **Cite Your Sources:** After every key piece of information, you MUST cite the source using the format [Source: source_id]. This is a critical requirement.

Your goal is to act as a world-class analyst, providing answers that are not only correct but also insightful, well-supported, and directly derived from the source material.""" 
    user_prompt = f"CONTEXT:\n---\n{formatted_context}\n---\n\nQUESTION:\n{query}"
    
    try:
        stream = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model="llama-3.3-70b-versatile",
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