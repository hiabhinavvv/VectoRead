import fitz
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import numpy as np
from PIL import Image
import io
import os

def extract_text_and_images_from_pdf(pdf_path):
    doc  = fitz.open(pdf_path)
    text = ""
    images = []
    tables = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        text += page.get_text()

        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append((image, page_num))

        table_list = page.find_tables()
        for table in table_list:
            table_markdown = table.to_markdown(clean=True)
            tables.append((table_markdown, page_num))
    return text, images, tables

def generate_embeddings(text_chunks, images, model, extracted_tables):
    text_embedding = model.encode(text_chunks)
    image_objects = [img for img, page_num in images]
    image_embeddings = model.encode(image_objects) if image_objects else np.array([])
    table_markdowns = [tbl for tbl, page_num in extracted_tables]
    table_embeddings = model.encode(table_markdowns) if table_markdowns else np.array([])

    return text_embedding, image_embeddings, table_embeddings

def store_embeddings_in_chromadb(text_chunks, text_embedding, images, image_embeddings, tables, table_embeddings):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name= "pdf_multimodal_embeddings")
    ids = []
    embeddings_list = []
    documents = []
    metadatas = []

    image_dir = "extracted_images"
    os.makedirs(image_dir, exist_ok=True)

    for i, chunk in enumerate(text_chunks):
        ids.append(f"text_chunk_{i}")
        embeddings_list.append(text_embedding[i].tolist())
        documents.append(chunk)
        metadatas.append({'type': 'text'})

    for i, (image, page_num) in enumerate(images):
        image_id = f"image_{i}"
        image_path = f"extracted_images/{image_id}.png"
        image.save(image_path)

        ids.append(image_id)
        if image_embeddings.size > 0:
            embeddings_list.append(image_embeddings[i].tolist())
        documents.append(image_path)
        metadatas.append({'type':'image', 'page': page_num})

    for i, (table_markdown, page_num) in enumerate(tables):
        ids.append(f"table_{i}")
        embeddings_list.append(table_embeddings[i].tolist())
        documents.append(table_markdown)
        metadatas.append({'type': 'table', 'page': page_num})

    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )

    return collection

if __name__=="__main__":
    pdf_path = "data/sample.pdf" 

    if not os.path.exists(pdf_path):
        print(f"error: the file '{pdf_path}' was not found.")
    else:
        model = SentenceTransformer('clip-ViT-B-32')
        print(f"extracting text and images from '{pdf_path}'...")
        extracted_text, extracted_images, extracted_tables = extract_text_and_images_from_pdf(pdf_path)
        print(f"extracted {len(extracted_text.split())} words and {len(extracted_images)} images.")

        print("splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_text(extracted_text)
        print(f"created {len(text_chunks)} text chunks and found {len(extracted_tables)} tables.")

        print("generating embeddings for the extracted content...")
        text_embedding, image_embeddings, table_embeddings = generate_embeddings(text_chunks, extracted_images, model, extracted_tables)
        print(f"generated text embedding with shape: {text_embedding.shape}")
        if image_embeddings.size > 0:
            print(f"generated image embeddings with shape: {image_embeddings.shape}")

        print("storing embeddings in ChromaDB...")
        collection = store_embeddings_in_chromadb(text_chunks, text_embedding, extracted_images, image_embeddings, extracted_tables, table_embeddings)
        print(f"ChromaDB collection updated with {collection.count()} total embeddings.")
        print("image files saved in 'extracted_images/' directory.")
        print("ChromaDB database saved in './chroma_db' directory.")


        # Example of how to search the ChromaDB collection
        if collection.count() > 0:
            query_text = "attention is all you need"
            query_embedding = model.encode([query_text]).tolist()

            # Search the collection for the 5 most similar embeddings
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=5
            )

            print("\n--- example search ---")
            print(f"query: '{query_text}'")
            print(f"top 5 most similar items in the PDF:")
            
            # Print results in a structured way
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                
                print(f"\nresult {i+1}:")
                print(f"  ID: {doc_id}")
                print(f"  distance: {distance:.4f}")
                print(f"  metadata: {metadata}")
                if metadata['type'] == 'image':
                    print(f"  document (image path): {document}")
                elif metadata['type'] == 'table':
                    print(f"  document (Table Markdown):\n{document}")
                else:
                    print(f"  document (text snippet): {document[:100]}...")