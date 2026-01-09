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
import csv
import json

load_dotenv()

def collection_exists(client, name: str) -> bool:
    try:
        client.get_collection(name)
        return True
    except:
        return False

def extract_content_from_pdf(file_content: bytes, min_image_size: int = 100):
    doc = fitz.open(stream=file_content, filetype="pdf")
    page_texts, images, tables = [], [], []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_tables = []
        for table in page.find_tables():
            md = table.to_markdown(clean=True)
            page_tables.append(md)
            tables.append((md, page_num))
        page_text = page.get_text()

        for table_md in page_tables:
            page_text = page_text.replace(table_md, "")

        headings = extract_headings_from_page(page)

        page_texts.append((page_text, page_num, headings))

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

    return page_texts, images, tables

def extract_content_from_csv(file_content: bytes):
    decoded = file_content.decode("utf-8").splitlines()
    reader = csv.DictReader(decoded)

    rows = []
    for row_idx, row in enumerate(reader):
        rows.append((row, row_idx))

    return rows


def serialize_csv_row(row: dict):
    return "; ".join([f"{k}={v}" for k, v in row.items()])

def extract_headings_from_page(page):
    headings = []
    blocks = page.get_text("dict")["blocks"]

    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            text = " ".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue

            font_sizes = [span["size"] for span in line["spans"]]
            avg_size = sum(font_sizes) / len(font_sizes)

            if avg_size >= 14 or text.isupper():
                headings.append(text)

    return headings

def store_section_index(
    session_id: str,
    page_texts,
    text_embedding_model
):
    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name=f"{session_id}_sections"
    )

    section_docs = []
    section_metadatas = []

    seen = set()

    for _, page_num, headings in page_texts:
        for heading in headings:
            heading = heading.strip().title()
            key = (heading.lower(), page_num)
            if key in seen:
                continue
            seen.add(key)

            section_docs.append(heading)
            section_metadatas.append({
                "type": "section",
                "page": page_num,
                "section": heading
            })

    if not section_docs:
        print("No sections found for section index.")
        return 0

    embeddings = text_embedding_model.encode(section_docs)

    collection.add(
        ids=[f"section_{i}" for i in range(len(section_docs))],
        documents=section_docs,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=section_metadatas
    )

    print(f"Stored {len(section_docs)} sections in section index.")
    return len(section_docs)

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
                    docs_img.append(image_path)
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
    store_section_index(session_id, page_texts, text_embedding_model)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_text_chunks = []
    all_text_metadatas = []
    current_section = None

    for text_content, page_num, headings in page_texts:
        if headings:
            current_section = headings[0]
        chunks = text_splitter.split_text(text_content)
        for chunk in chunks:
            all_text_chunks.append(chunk)
            all_text_metadatas.append({'type': 'text', 'page': page_num, 'section': current_section})
    text_embeds, image_embeds, table_embeds = generate_embeddings(
        all_text_chunks, images, tables, text_embedding_model, image_embedding_model
    )
    
    count = store_in_chromadb(
        session_id, all_text_chunks, text_embeds, images, image_embeds, tables, table_embeds, all_text_metadatas
    )
    print(f"--- Successfully stored {count} items across collections for session '{session_id}' ---")
    return count

def process_and_store_csv(
    session_id: str,
    file_content: bytes,
    text_embedding_model
):
    print("--- Starting CSV Ingestion ---")

    client = chromadb.PersistentClient(path="./chroma_db")
    rows = extract_content_from_csv(file_content)

    if not rows:
        print("No rows found in CSV.")
        return 0

    texts = [serialize_csv_row(row) for row, _ in rows]
    embeddings = text_embedding_model.encode(texts)

    collection_tables = client.get_or_create_collection(
        name=f"{session_id}_tables"
    )

    collection_tables.add(
        ids=[f"csv_row_{i}" for i in range(len(texts))],
        documents=texts,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[
            {"type": "table", "row": i, "source": "csv"}
            for i in range(len(texts))
        ]
    )

    print(f"Stored {len(texts)} CSV rows as table entries.")
    return len(texts)



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
    
def plan_query_with_llm(query: str, groq_client: Groq):
    prompt = f"""
You are a query planner for a document intelligence system.

Classify the user query and decide retrieval strategy.

Return EXACTLY one JSON object.
No markdown.
No explanations.
No extra text.

The response MUST match this schema exactly:

{{
  "intent": "section_lookup" | "factual_lookup" | "table_lookup" | "image_lookup" | "broad_search"
}}

Rules:
- Use section_lookup for definitions, concepts, lifecycle, phases
- Use factual_lookup for numbers, values, metrics
- Use table_lookup for CSV or tabular data
- Use image_lookup for diagrams or figures

You MUST choose the single most appropriate intent.
broad_search is allowed ONLY if the query is vague or exploratory.
DO NOT default to broad_search.


User query:
"{query}"
"""
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a strict JSON classifier. Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        raw = completion.choices[0].message.content.strip()
        print("[Planner RAW]:", raw)

        plan = json.loads(raw)
        assert plan["intent"] in {
            "section_lookup",
            "factual_lookup",
            "table_lookup",
            "image_lookup",
            "broad_search"
        }
        
        return plan
    
    except Exception as e:
        print(f"[Planner Fallback] {e}")
        return {"intent": "broad_search"}


def process_query_and_generate(
    query: str,
    session_id: str,
    text_embedding_model,
    image_embedding_model,
    groq_client
):
    print("\n--- Processing Query (Agentic RAG – Stable) ---")
    client = chromadb.PersistentClient(path="./chroma_db")

    plan = plan_query_with_llm(query, groq_client)
    intent = plan.get("intent", "broad_search")
    q = query.lower()

    print(f"[Planner] Intent: {intent}")

    context_parts = []

    if intent == "section_lookup" and collection_exists(client, f"{session_id}_sections"):
        try:
            sections = client.get_collection(f"{session_id}_sections")
            query_embedding = text_embedding_model.encode([query]).tolist()

            res = sections.query(
                query_embeddings=query_embedding,
                n_results=1,
                include=["metadatas"]
            )

            if res["metadatas"] and res["metadatas"][0]:
                page = res["metadatas"][0][0]["page"]
                print(f"[Router] Section → Page {page + 1}")

                if collection_exists(client, f"{session_id}_text"):
                    text_col = client.get_collection(f"{session_id}_text")
                    page_chunks = text_col.query(
                        query_embeddings=query_embedding,
                        where={"page": page},
                        n_results=20,
                        include=["documents", "metadatas"]
                    )

                    print(
    f"[Router] Retrieved {len(page_chunks['documents'][0])} chunks from Page {page + 1}"
)

                    for doc, meta in zip(
                        page_chunks["documents"][0],
                        page_chunks["metadatas"][0]
                    ):
                        context_parts.append(
                            f"Source: Page {meta['page'] + 1} (text)\nContent: {doc}"
                        )
        except Exception as e:
            print(f"[Section Error] {e}")

    elif intent == "table_lookup" and collection_exists(client, f"{session_id}_tables"):
        try:
            tables = client.get_collection(f"{session_id}_tables")
            query_embedding = text_embedding_model.encode([query]).tolist()

            res = tables.query(
                query_embeddings=query_embedding,
                n_results=10,
                include=["documents"]
            )

            for doc in res["documents"][0]:
                context_parts.append(f"Source: Table\nContent: {doc}")
        except Exception as e:
            print(f"[Table Error] {e}")

    elif intent == "image_lookup" and collection_exists(client, f"{session_id}_images"):
        try:
            images = client.get_collection(f"{session_id}_images")
            query_embedding = image_embedding_model.encode([query]).tolist()

            res = images.query(
                query_embeddings=query_embedding,
                n_results=2,
                include=["documents", "metadatas"]
            )

            for path, meta in zip(res["documents"][0], res["metadatas"][0]):
                desc = analyze_image_with_groq(path, groq_client)
                context_parts.append(
                    f"Source: Page {meta['page'] + 1} (image)\nContent: {desc}"
                )
        except Exception as e:
            print(f"[Image Error] {e}")

    if not context_parts:
        print("[Fallback] Capability-aware search")

        if collection_exists(client, f"{session_id}_text"):
            text_col = client.get_collection(f"{session_id}_text")
            query_embedding = text_embedding_model.encode([query]).tolist()

            res = text_col.query(
                query_embeddings=query_embedding,
                n_results=5,
                include=["documents", "metadatas"]
            )

            for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                context_parts.append(
                    f"Source: Page {meta['page'] + 1} (text)\nContent: {doc}"
                )

        elif collection_exists(client, f"{session_id}_tables"):
            tables = client.get_collection(f"{session_id}_tables")
            res = tables.get(include=["documents"])

            for doc in res["documents"]:
                context_parts.append(f"Source: Table\nContent: {doc}")

        else:
            yield "No data available for this session."
            return
        
    formatted_context = "\n---\n".join(context_parts)
    system_prompt = """System Prompt: Geospatial Intelligence Analyst

You are a highly intelligent, expert AI assistant specializing in geospatial intelligence and location-based analytics. Your primary responsibility is to analyze, synthesize, and reason strictly from the provided context to answer user queries related to locations, regions, spatial entities, and business-relevant geographic insights.

Follow these instructions meticulously:

1. Context-Strict Reasoning

Your response must be derived only from the provided context, which may include structured data (CSV rows, tables), unstructured text, and metadata describing geographic attributes (e.g., city, locality, coordinates, administrative boundaries, catchment areas).

Do not infer, assume, or supplement information from external knowledge.

If the required geographic entity (city, locality, region, grid, or boundary) is not explicitly present in the context, you must clearly state that the data is unavailable.

2. Explicit Geospatial Validation (Critical)

Before answering, validate that the retrieved data explicitly matches the user’s requested location or spatial scope.

Check for:

Exact or clearly defined geographic identifiers (e.g., locality name, city, ward, district, region)

Spatial containment or hierarchy only if explicitly stated in the context

If no valid match exists:

State that the requested location is not covered in the available data

Do not substitute with nearby, similar, or semantically related locations

3. Expert Location-Intelligence Tone

Rewrite and interpret the information as a geospatial intelligence analyst supporting real business decisions (e.g., site selection, market expansion, risk assessment, demand estimation).

Use precise spatial and analytical terminology (e.g., catchment, density, coverage, penetration, proximity, clustering).

Explain insights clearly and professionally, without speculation.

4. Data-Driven and Evidence-Based Outputs

If the context includes:

Metrics (e.g., AQI values, footfall counts, population density, affluence indices)

Spatial attributes (e.g., buffers, radii, zones, polygons)

Time ranges or measurement units

You must:

Include them accurately

Preserve units, ranges, and definitions

Avoid generalizations beyond the provided data

If formulas, rules, or calculations are present, represent them faithfully.

5. Structured, Analytical Responses

Organize your answer logically, especially for “what,” “why,” or “how” questions.

A strong response may include:

Location identification

Data availability confirmation

Key spatial metrics

Business interpretation

Limitations of the data (if any)

Avoid vague summaries or filler content.

6. Transparent Handling of Missing or Partial Coverage

If the context contains related data for other locations but not the one requested:

Clearly state that the requested location is not present

Optionally mention that other locations exist in the dataset without using them to answer

Do not imply completeness of coverage unless explicitly stated

7. Mandatory Source Attribution

After every key factual statement, cite the source using the format:

[Source: Page X (text/table/image)]


Examples:

[Source: Page 3 (table)]

[Source: Page 7 (text)]""" 
    user_prompt = f"CONTEXT:\n---\n{formatted_context}\n---\n\nQUESTION:\n{query}"
    
    try:
        stream = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model="llama-3.3-70b-versatile",
            temperature= 0,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Error calling Groq API: {e}"