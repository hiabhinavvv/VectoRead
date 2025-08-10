from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import uvicorn
from contextlib import asynccontextmanager
import os
import shutil
import rag_logic
import uuid

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_logic.load_query_models()
    yield
    print("Server shutting down.")

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:5173",  # For local development
    "https://vecto-read.vercel.app/", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: str

class IngestResponse(BaseModel):
    message: str
    item_count: int
    session_id: str


@app.get("/")
def read_root():
    return {"status": "Multimodal RAG API is running"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        session_id = str(uuid.uuid4())
        print(f"New session started: {session_id}")

        file_content = await file.read()
        
        text, images, tables = rag_logic.extract_content_from_pdf(file_content)
        
        text_splitter = rag_logic.RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_text(text)
        
        print("Generating embeddings...")
        embedding_model = rag_logic.SentenceTransformer('clip-ViT-B-32')
        text_emb, img_emb, tbl_emb = rag_logic.generate_embeddings(text_chunks, images, tables, embedding_model)
        
        print("Storing in ChromaDB...")
        item_count = rag_logic.store_in_chromadb(
            session_id, text_chunks, text_emb, images, img_emb, tables, tbl_emb
        )
        
        rag_logic.load_query_models()

        return IngestResponse(
            message=f"Successfully ingested '{file.filename}'", 
            item_count=item_count,
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during ingestion: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/query")
async def handle_query(request: QueryRequest):
    return StreamingResponse(
        rag_logic.process_query_and_generate(request.query, request.session_id), 
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)