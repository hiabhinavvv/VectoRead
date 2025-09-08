from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import uvicorn
from contextlib import asynccontextmanager
import os
import uuid

# Import your refactored logic and the necessary model libraries
import rag_logic
from sentence_transformers import SentenceTransformer
from groq import Groq

# This dictionary will act as a simple cache to hold the loaded models
# and make them accessible to your API endpoints.
model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan. Models are loaded on startup
    and cleared on shutdown.
    """
    # --- Code to run on startup ---
    print("INFO:     Loading models...")
    model_cache["text_model"] = SentenceTransformer('all-MiniLM-L6-v2')
    model_cache["image_model"] = SentenceTransformer('clip-ViT-B-32')
    model_cache["groq_client"] = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("INFO:     Models loaded successfully.")
    
    yield # The application is now ready to run
    
    # --- Code to run on shutdown ---
    print("INFO:     Server shutting down. Clearing model cache.")
    model_cache.clear()

# Initialize the FastAPI app with the new lifespan manager
app = FastAPI(lifespan=lifespan)

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request/Response Validation ---
class QueryRequest(BaseModel):
    query: str
    session_id: str

class IngestResponse(BaseModel):
    message: str
    item_count: int
    session_id: str

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Multimodal RAG API is running"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Handles the ingestion of a PDF file by calling the orchestrator function.
    """
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        session_id = str(uuid.uuid4())
        print(f"New ingestion session started: {session_id}")

        file_content = await file.read()
        
        # A single call to the orchestrator function from rag_logic.py
        item_count = rag_logic.process_and_store_pdf(
            session_id=session_id,
            file_content=file_content,
            text_embedding_model=model_cache["text_model"],
            image_embedding_model=model_cache["image_model"]
        )
        
        return IngestResponse(
            message=f"Successfully ingested '{file.filename}'", 
            item_count=item_count,
            session_id=session_id
        )

    except Exception as e:
        # Provide a more detailed error message for debugging
        print(f"ERROR during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during ingestion: {str(e)}")


@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Handles a user query by streaming the response from the RAG pipeline.
    """
    # Create the generator by calling the query processing function
    response_generator = rag_logic.process_query_and_generate(
        query=request.query, 
        session_id=request.session_id,
        text_embedding_model=model_cache["text_model"],
        groq_client=model_cache["groq_client"]
    )
    
    # Return the generator in a streaming response
    return StreamingResponse(
        response_generator, 
        media_type="text/event-stream"
    )

# --- Main entry point for running the server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)