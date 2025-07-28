<div align="center"> <img src="https://raw.githubusercontent.com/user-attachments/assets/a8b2c8e3-b3e1-4b10-8b1e-7b7e8c0e9a0a" alt="Vectoread Logo" width="150"> <h1 style="font-size: 3.5em; font-weight: bold; border-bottom: none;"> Vectoread </h1> <p style="font-size: 1.5em; color: #888; margin-top: -10px;"> Your AI, Off the Grid. </p> <p> Deep document analysis—multimodal, private, and right on your machine. </p> </div>
🚀 What is Vectoread?
Vectoread is your personal, full-stack AI "librarian" that transforms dense, dusty PDF files into something truly interactive: a chat-ready, context-aware knowledge base. Whether you’re a privacy stickler, a researcher, or just sick of hunting through endless PDFs, Vectoread has your back—locally, securely, and intelligently.

Key to it all? Vectoread doesn’t just read text—it “sees” your diagrams, deciphers images, and parses tricky tables. You ask questions in plain language and get answers synthesized from all that extracted content, so your PDFs finally talk back (usefully).

✨ Features at a Glance
🧠 True Multimodal Intelligence: Not just skimming for words—Vectoread processes text, images and tables. If it’s part of your document, it’s part of the answer.

🔒 Zero Cloud. 100% Private: All processing happens with your API keys or entirely offline—your documents never leave your own machine.

🚀 Advanced RAG Architecture:

Stage 1: A Vision-Language Model (VLM) interprets visuals.

Stage 2: An LLM synthesizes all modalities into clear, detailed answers.

⚡ Real-time Chat: With FastAPI’s streaming, responses pop up as they’re generated—no more waiting ages for a reply.

🐳 Dockerized & Portable: Frontend and backend in tidy Docker containers. Go from zero to analysis with a single command, regardless of OS.

🛠️ How It Works: Under the Hood
Vectoread runs a two-phase Retrieval-Augmented Generation (RAG) pipeline:

1. Ingestion (The /ingest Endpoint)
Upload a PDF, and here’s what happens:

Extraction: Text, images, and tables pulled from the PDF (using PyMuPDF).

Chunking & Formatting: Long text? Sliced for maximum retrieval. Tables become easy-to-use Markdown.

Vector Embeddings: Everything—text, images, tables—gets “vectorized” via the CLIP model, so the machine “remembers” what’s in each chunk.

Indexing: Content + vectors land in your local ChromaDB, ready for lightning-fast search.

2. Querying (The /query Endpoint)
Ask a question, get synthesized insight:

Context Retrieval: Your query is “vectorized,” then ChromaDB surfaces the most relevant passages, tables, and visuals.

Visual Analysis (VLM): For any images, a state-of-the-art VLM (Llama 4 Scout) generates detailed descriptions—no more squinting at JPEGs.

Answer Synthesis (LLM): Text, tables, VLM-generated descriptions—all combined as context for a top-tier LLM (Llama 3 70B), which delivers a clear, concise answer.

💻 Tech Stack
Area	Tech / Model
Backend	Python, FastAPI
Frontend	React, JavaScript, CSS
DevOps	Docker, Docker Compose
Vector DB	ChromaDB
Embeddings	sentence-transformers/clip-ViT-B-32
Vision (VLM)	Groq Llama 4 Scout
Gen (LLM)	Groq Llama 3 70B
PDF Parsing	PyMuPDF
Chunking	LangChain
⚙️ Quickstart: Run Vectoread Locally
Curious to try it out? Here’s how you get started in a few minutes:

Prerequisites
Git

Python 3.9+

Node.js & npm

Docker & Docker Compose

1. Clone the Repo
bash
git clone https://github.com/your-username/Vectoread.git
cd Vectoread
2. Configure Secrets
Go to backend/, copy the example .env, and insert your own API keys as needed:

bash
cd backend
cp .env.example .env
nano .env   # Or any editor you like
3. Fire Up Docker Compose
From the root project directory:

bash
docker-compose up --build -d
This will spin up both frontend and backend containers—no dependency hell, no drama.

4. Explore
Frontend: http://localhost:5173

Backend API/docs: http://localhost:8000

🛣️ Roadmap & What’s Next
 Smarter retrieval with a re-ranking model (more accuracy, less noise)

 Support for more file types (.docx, .pptx, etc.)

 Chat history and conversation context

 CI/CD pipeline (hello automation!)

 Easy, one-click cloud deployment (e.g., Render, Vercel)

Vectoread is always growing. Questions, ideas, bug reports? Open an issue, start a discussion, or send a PR!

Why rely on expensive remote AI silos? Take back control of your documents—and let AI work for you, not the other way around.