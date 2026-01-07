from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import uvicorn
from contextlib import asynccontextmanager
import os
import uuid
import rag_logic
from sentence_transformers import SentenceTransformer
from groq import Groq

model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("INFO:     Loading models...")
    model_cache["text_model"] = SentenceTransformer('all-MiniLM-L6-v2')
    model_cache["image_model"] = SentenceTransformer('clip-ViT-B-32')
    model_cache["groq_client"] = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("INFO:     Models loaded successfully.")
    yield
    print("INFO:     Server shutting down. Clearing model cache.")
    model_cache.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
class QueryRequest(BaseModel): query: str; session_id: str
class IngestResponse(BaseModel): message: str; item_count: int; session_id: str

@app.get("/")
def read_root():
    return {"status": "Multimodal RAG API is running"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    try:
        session_id = str(uuid.uuid4())
        file_content = await file.read()
        filename = file.filename.lower()

        if filename.endswith(".pdf"):
            item_count = rag_logic.process_and_store_pdf(
                session_id=session_id,
                file_content=file_content,
                text_embedding_model=model_cache["text_model"],
                image_embedding_model=model_cache["image_model"]
            )

        elif filename.endswith(".csv"):
            item_count = rag_logic.process_and_store_csv(
                session_id=session_id,
                file_content=file_content,
                text_embedding_model=model_cache["text_model"]
            )

        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only PDF and CSV are supported."
            )

        return IngestResponse(
            message=f"Successfully ingested '{file.filename}'",
            item_count=item_count,
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during ingestion: {str(e)}"
        )

@app.post("/query")
async def handle_query(request: QueryRequest):
    response_generator = rag_logic.process_query_and_generate(
        query=request.query, 
        session_id=request.session_id,
        text_embedding_model=model_cache["text_model"],
        image_embedding_model=model_cache["image_model"],
        groq_client=model_cache["groq_client"]
    )
    return StreamingResponse(response_generator, media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)