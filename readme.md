<div align="center">
  <img src="https://raw.githubusercontent.com/user-attachments/assets/a8b2c8e3-b3e1-4b10-8b1e-7b7e8c0e9a0a" alt="Vectoread Logo" width="150">
  <h1 style="font-size: 3.5em; font-weight: bold; border-bottom: none;">
    Vectoread
  </h1>
  <p style="font-size: 1.5em; color: #888; margin-top: -10px;">
    Your AI, Off the Grid.
  </p>
  <p>
    An advanced, multimodal RAG pipeline for deep, private document analysis.
  </p>
</div>

---

## 🚀 About The Project

Vectoread is a full-stack application that transforms dense, static PDF documents into interactive, conversational knowledge bases. It addresses the critical need for private, local-first document analysis, ensuring your data is never sent to a third-party cloud for processing.

The system ingests any PDF and intelligently parses not just the text, but also understands the content within **images**, **diagrams**, and **tables**. Users can then ask complex questions in natural language and receive accurate, context-aware answers synthesized from all extracted modalities, creating a powerful tool for research, study, and professional analysis.

## ✨ Key Features

- **🧠 Multimodal Understanding**: Goes beyond simple text extraction to analyze images and parse structured tables.
- **🔒 Privacy First**: All document processing and analysis happens through your own secure API keys or local models. Your documents are never uploaded to a third-party service.
- **🚀 Advanced RAG Pipeline**: Utilizes a two-stage process for high-quality answers:
    1.  A powerful Vision-Language Model (VLM) analyzes retrieved images.
    2.  A state-of-the-art Large Language Model (LLM) synthesizes the final answer from all context.
- **⚡ Real-time Interaction**: The FastAPI backend provides a streaming API endpoint for a responsive, real-time chat experience.
- **🐳 Fully Containerized**: The entire application (React Frontend + FastAPI Backend) is containerized with Docker, allowing for consistent, one-command setup and deployment.

## 🛠️ How It Works: The RAG Architecture

Vectoread operates in two main phases:

#### 1. Ingestion Phase (`/ingest`)
When a PDF is uploaded, the system performs a comprehensive analysis:
1.  **Extract Content**: Text, images, and tables are extracted from the PDF using PyMuPDF.
2.  **Chunk & Process**: Long text is split into smaller, overlapping chunks. Tables are converted to Markdown.
3.  **Generate Embeddings**: The multimodal `CLIP` model converts all content chunks (text, images, tables) into vector embeddings.
4.  **Store in Vector DB**: The content and its corresponding embeddings are stored in a local ChromaDB database for efficient retrieval.

#### 2. Query Phase (`/query`)
When a user asks a question, the pipeline is activated:
1.  **Retrieve Context**: The user's query is embedded, and a similarity search is performed on ChromaDB to find the most relevant text, tables, and images.
2.  **Analyze Images (VLM)**: For any retrieved images, a powerful Vision-Language Model (Groq's Llama 4 Scout) is called to generate a detailed text description.
3.  **Synthesize Answer (LLM)**: The retrieved text, tables, and the new image descriptions are combined into a rich context. This context is then sent to a powerful text-based LLM (Groq's Llama 3 70B) to generate a final, synthesized, and human-readable answer.

## 💻 Technology Stack

| Area      | Technology / Model                                       |
| :-------- | :------------------------------------------------------- |
| **Backend** | Python, FastAPI                                          |
| **Frontend** | React, JavaScript, CSS                                   |
| **Containerization** | Docker, Docker Compose                                   |
| **Vector DB** | ChromaDB                                                 |
| **Embedding** | `sentence-transformers/clip-ViT-B-32`                    |
| **Vision (VLM)** | Groq `meta-llama/llama-4-scout-17b-16e-instruct`         |
| **Generation (LLM)** | Groq `llama3-70b-8192`                                   |
| **PDF Parsing** | PyMuPDF                                                  |
| **Text Chunking** | LangChain                                                |

---

## ⚙️ Getting Started: Local Setup

Follow these steps to get the entire application running on your local machine.

### Prerequisites
-   Git
-   Python 3.9+
-   Node.js & npm
-   Docker & Docker Compose

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Vectoread.git](https://github.com/your-username/Vectoread.git)
    cd Vectoread
    ```

2.  **Set up Environment Variables:**
    Navigate to the `backend/` directory, copy the example environment file, and add your secret API keys.
    ```bash
    cd backend
    cp .env.example .env
    nano .env  # Or use your favorite text editor
    ```

3.  **Build and Run with Docker Compose:**
    From the **root directory** of the project, run the following command:
    ```bash
    docker-compose up --build -d
    ```
    This will build the frontend and backend images and start the containers in detached mode.

4.  **Access the Application:**
    -   The **React Frontend** will be available at `http://localhost:5173`.
    -   The **FastAPI Backend** will be running on `http://localhost:8000`. You can access the interactive API docs at `http://localhost:8000/docs`.

## 🛣️ Future Roadmap

-   [ ] Implement a re-ranking model to further improve retrieval accuracy.
-   [ ] Add support for more document types (e.g., `.docx`, `.pptx`).
-   [ ] Create a chat history feature.
-   [ ] Set up a full CI/CD pipeline with Jenkins for automated deployment.
-   [ ] Deploy the backend to a cloud service like Render and the frontend to Vercel.

