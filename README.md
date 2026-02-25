# RAG Single-File (Groq + LangChain + Qdrant)

A modern, high-performance Retrieval-Augmented Generation (RAG) application leveraging **Groq**'s ultra-fast inference, **LangChain** for orchestration, and **Qdrant** for local vector storage. This project features a full-stack implementation with a FastAPI backend and a Vite/React frontend.

## 🚀 Key Features

- **Blazing Fast Inference**: Powered by Groq's LPU with `llama-3.1-8b-instant`.
- **Local Vector Database**: Qdrant running in local mode (no Docker/Server required for basic usage).
- **Efficient Retrieval**: Semantic search using `all-MiniLM-L6-v2` HuggingFace embeddings.
- **Full-Stack Chat**: Modern React interface to interact with your documents.
- **Easy Ingestion**: Simple Python script to process and index PDF documents.

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, FastAPI, LangChain, Qdrant Client.
- **LLM**: Groq (Llama 3.1 8B).
- **Embeddings**: HuggingFace (Sentence Transformers).
- **Frontend**: React, Vite, Tailwind CSS (optional but likely used in the structure).
- **Storage**: Qdrant (Local disk storage).

## 📁 Project Structure

```text
rag-single-file/
├── api/                # FastAPI application & routes
├── app/                # Core RAG logic, pipeline, and config
├── frontend/           # Vite + React frontend
├── data/               # Place your PDF files here
├── qdrant_data/        # Local Qdrant database storage
├── ingest.py           # Document ingestion script
├── query.py            # CLI query tool
└── requirements.txt    # Python dependencies
```

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.10 or higher
- Node.js & npm (for frontend)
- A Groq API Key (Get one at [console.groq.com](https://console.groq.com/))

### 2. Backend Setup
1. **Clone the repository** (or navigate to the folder).
2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   LLM_PROVIDER=groq
   ```

### 3. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```

## 🏃 Usage

### 🚀 Step 1: Ingest Data
Place your PDF files in the `data/` folder. Update the `FILE_PATH` in `ingest.py` if necessary, then run:
```bash
python ingest.py
```
This will chunk the document, generate embeddings, and store them in the local Qdrant instance.

### 🧪 Step 2: Test via CLI (Optional)
You can test the RAG pipeline directly from your terminal:
```bash
python query.py
```

### 🌐 Step 3: Run the Full Application

**Start the Backend:**
```bash
# From the root directory
python -m uvicorn api.main:app --reload
```

**Start the Frontend:**
```bash
cd frontend
npm run dev
```
Open your browser at the URL provided by Vite (usually `http://localhost:5173`).

## 🧠 How it Works

1. **Ingestion**: Documents are loaded via `PyPDFLoader`, split into manageable chunks using `RecursiveCharacterTextSplitter`, and embedded using HuggingFace models.
2. **Storage**: The vectors are stored locally in the `qdrant_data` folder.
3. **Retrieval**: When a user asks a question, the application searches the local Qdrant collection for relevant chunks.
4. **Generation**: The retrieved context, along with the chat history and the question, is sent to Groq's Llama 3.1 model to generate a precise answer.

## 📄 License
MIT
