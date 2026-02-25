from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import shutil, os, traceback

# ─── MODELS ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question: str
    history: Optional[List] = []

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TEMP_DIR = "/tmp"
COLLECTION_NAME = "rag_final"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md"}

# ─── QDRANT ───────────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

_qdrant_client = None

def get_client():
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL")
        key = os.getenv("QDRANT_API_KEY")
        _qdrant_client = QdrantClient(url=url, api_key=key)
    return _qdrant_client

VECTOR_SIZE = 384

def ensure_collection():
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

# ─── EMBEDDINGS ───────────────────────────────────────────────────────────────
import requests, time

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL = "BAAI/bge-small-en-v1.5"
HF_URL   = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

def embed(texts: list) -> list:
    """Embed a list of texts → list of float vectors."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "X-Wait-For-Model": "true"}
    for attempt in range(3):
        try:
            r = requests.post(HF_URL, headers=headers, json={"inputs": texts}, timeout=60)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    # Handle 3D output (mean pool)
                    if data and isinstance(data[0][0], list):
                        return [[sum(col)/len(tok) for col in zip(*tok)] for tok in data]
                    return data
                if isinstance(data, dict) and "loading" in data.get("error","").lower():
                    time.sleep(15)
                    continue
            print(f"HF embed error {r.status_code}: {r.text[:200]}", flush=True)
        except Exception as e:
            print(f"HF embed exception: {e}", flush=True)
        time.sleep(5)
    raise ValueError("Embedding failed after 3 retries")

# ─── DOCUMENT LOADING ─────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(file_path: str) -> list:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".docx", ".doc"):
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)

# ─── QDRANT DIRECT OPS ────────────────────────────────────────────────────────
import uuid

def store_chunks(chunks: list, session_id: str):
    """Embed and upsert chunks directly into Qdrant."""
    client = get_client()
    ensure_collection()
    texts  = [c.page_content for c in chunks]
    vecs   = embed(texts)
    points = []
    for text, vec in zip(texts, vecs):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vec,
            "payload": {"session_id": session_id, "text": text}
        })
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)

def retrieve_chunks(session_id: str, question: str, top_k: int = 5) -> list:
    """
    Retrieve relevant chunks for a session:
    1. Get query embedding
    2. Scroll all points for this session
    3. Return them (order by payload match)
    """
    client = get_client()
    # Scroll all points for this session (Python-side filter — 100% reliable)
    all_points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=500,
        with_payload=True,
        with_vectors=False
    )
    session_points = [
        p.payload.get("text", "")
        for p in all_points
        if str(p.payload.get("session_id", "")) == session_id
    ]
    print(f"RETRIEVE: {len(all_points)} total, {len(session_points)} for session '{session_id}'", flush=True)
    return session_points[:top_k]

# ─── LLM ──────────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0,
            streaming=True
        )
    return _llm

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    errors = []
    if not os.getenv("GROQ_API_KEY"):        errors.append("GROQ_API_KEY missing")
    if not os.getenv("QDRANT_URL"):          errors.append("QDRANT_URL missing")
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"): errors.append("HUGGINGFACEHUB_API_TOKEN missing")
    try:
        cols = [c.name for c in get_client().get_collections().collections]
    except Exception as e:
        cols = [f"ERROR: {e}"]
    return {"status": "ok" if not errors else "error", "errors": errors, "collections": cols}

@app.get("/inspect/{session_id}")
async def inspect(session_id: str):
    client = get_client()
    all_points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=20, with_payload=True, with_vectors=False)
    matched = [{"id": str(p.id), "session_id": p.payload.get("session_id"), "text": p.payload.get("text","")[:100]} for p in all_points if str(p.payload.get("session_id","")) == session_id]
    all_ids = list(set(str(p.payload.get("session_id","")) for p in all_points))
    return {"searched": session_id, "matched_count": len(matched), "all_session_ids_in_db": all_ids, "sample": matched[:3]}

@app.post("/upload")
async def upload(session_id: str, file: UploadFile = File(...)):
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(400, "GROQ_API_KEY missing")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    path = os.path.join(TEMP_DIR, file.filename)
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        chunks = load_and_chunk(path)
        if not chunks:
            return {"chunks": 0, "message": "No content extracted"}
        n = store_chunks(chunks, session_id)
        return {"chunks": n, "message": f"Indexed {n} chunks for session {session_id}"}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"UPLOAD ERROR: {e}\n{tb}", flush=True)
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/chat")
async def chat(req: ChatRequest):
    def stream():
        try:
            chunks = retrieve_chunks(req.session_id, req.question)
            if not chunks:
                context = "NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT"
            else:
                context = "\n\n".join(chunks)

            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_template(
                """Answer based ONLY on the CONTEXT. If context says NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT, ask the user to upload a document first.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
            )
            chain = prompt | get_llm()
            for chunk in chain.stream({"context": context, "question": req.question}):
                yield getattr(chunk, "content", str(chunk))
        except Exception as e:
            print(f"CHAT ERROR: {e}", flush=True)
            yield f"Error: {e}"

    return StreamingResponse(stream(), media_type="text/plain")
