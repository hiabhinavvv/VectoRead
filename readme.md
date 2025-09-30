<div align="center">
<h1 style="font-size: 3.5em; font-weight: bold; border-bottom: none;">
VectoRead
</h1>
<p style="font-size: 1.5em; color: #888; margin-top: -10px;">
Your AI, Off the Grid.
</p>
<p>
An advanced, multimodal RAG pipeline for deep, private document analysis.
</p>
</div>

🚀 About The Project
VectoRead is a full-stack application that transforms static PDF documents into interactive, conversational knowledge bases. It addresses the critical need for private, local-first document analysis, ensuring your data is never sent to a third-party cloud for processing.

The system ingests any PDF and intelligently parses not just the text, but also understands the content within images, diagrams, and tables. Users can then ask complex questions in natural language and receive accurate, context-aware answers synthesized from all extracted modalities, creating a powerful tool for research, study, and professional analysis.

✨ Key Features
🧠 Multimodal Understanding: Goes beyond simple text extraction to analyze images and parse structured tables.

🎯 High-Fidelity Retrieval: Employs a dual-model embedding strategy with specialized models for text and images to maximize retrieval relevance for each modality.

🔒 Privacy First: All document processing and analysis happens through your own secure API keys. Your documents are never uploaded to a third-party service.

🚀 Advanced RAG Pipeline: Utilizes a sophisticated query process that includes real-time VLM analysis for retrieved images before final synthesis by a powerful LLM.

⚡ Real-time Interaction: The FastAPI backend provides a streaming API endpoint for a responsive, real-time chat experience.

🐳 Fully Containerized: The entire application (React Frontend + FastAPI Backend) is containerized with Docker, allowing for consistent, one-command setup and deployment.

🛠️ How It Works: The Dual-Engine RAG Architecture
VectoRead uses a sophisticated dual-collection architecture to optimize retrieval for different types of data.

1. Ingestion Phase (/ingest)
When a PDF is uploaded, the system performs a comprehensive analysis:

Extract Content: Robustly extracts text, images, and tables using PyMuPDF, skipping corrupted data.

Chunk & Process: Long text is split into smaller chunks and deduplicated to remove redundant headers/footers. Tables are converted to Markdown.

Generate Specialized Embeddings: The system uses two separate, specialized models:

Text & Tables: all-MiniLM-L6-v2 creates high-quality, 384-dimensional embeddings optimized for semantic text understanding.

Images: clip-ViT-B-32 creates 512-dimensional embeddings that capture the visual meaning of images.

Store in Vector DB: The embeddings are stored in two separate ChromaDB collections (_text and _images) to manage the different vector dimensions and optimize searches.

2. Query Phase (/query)
When a user asks a question, the hybrid pipeline is activated:

Dual Retrieval: The user's query is embedded twice, using both the text and image models.

The text embedding is used to search the _text collection for relevant text and tables.

The image embedding is used for a semantic text-to-image search in the _images collection.

Analyze Images (VLM): For any retrieved images, a powerful Vision-Language Model (Groq's LLaVA) is called to generate a detailed, real-time text description.

Synthesize Answer (LLM): The retrieved text, tables, and the new VLM-generated image descriptions are combined into a rich context. This context is then sent to a powerful text-based LLM (Groq's Llama 3) to generate a final, synthesized, and human-readable answer.

💻 Technology Stack
Area	Technology / Model
Backend	Python, FastAPI
Frontend	React, JavaScript, CSS
Containerization	Docker, Docker Compose
Vector DB	ChromaDB (Dual-Collection)
Text Embedding	sentence-transformers/all-MiniLM-L6-v2
Image Embedding	sentence-transformers/clip-ViT-B-32
Vision (VLM)	Groq llava-llama-3-8b-32768
Generation (LLM)	Groq llama3-70b-8192
PDF Parsing	PyMuPDF
Text Chunking	LangChain

Export to Sheets
🚀 Access the Live Application
You can interact with the live, hosted version of VectoRead here:

➡️ vecto-read.vercel.app

Simply visit the link, upload a PDF, and start asking questions!


🛣️ Future Roadmap
[ ] Implement advanced re-ranking strategies (e.g., Reciprocal Rank Fusion) to better merge results from the dual collections.

[ ] Add support for more document types (e.g., .docx, .pptx).

[ ] Add support of OCR for handwritten docs.

[ ] Implement AI Agents such as critique bot for better responses + calculator support for better mathematical computation of equations that look scary.

