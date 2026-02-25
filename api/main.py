from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import shutil, os, traceback, uuid, time

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ─── ENV ──────────────────────────────────────────────────────────────────────
TEMP_DIR   = "/tmp"
GROQ_KEY   = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
HF_TOKEN   = os.getenv("HUGGINGFACEHUB_API_TOKEN")
SUPPORTED  = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md"}

COLLECTION  = "rag_sessions"
VECTOR_SIZE = 1   # Dummy single-dimension vector — we ONLY use payload for retrieval

# ─── QDRANT CLIENT ────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

_qclient = None

def qclient():
    global _qclient
    if _qclient is None:
        _qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
    return _qclient

def ensure_collection():
    c = qclient()
    cols = [x.name for x in c.get_collections().collections]
    if COLLECTION not in cols:
        c.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

# ─── DOCUMENT PROCESSING ──────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(path: str) -> list:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext in (".docx", ".doc"):
        loader = Docx2txtLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8", autodetect_encoding=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)

# ─── STORE (No real embeddings — payload-only retrieval) ──────────────────────
def store_chunks(chunks: list, session_id: str) -> int:
    ensure_collection()
    points = []
    for chunk in chunks:
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.0],          # Dummy vector — retrieval uses payload scan
            payload={"session_id": session_id, "text": chunk.page_content}
        ))
    qclient().upsert(collection_name=COLLECTION, points=points)
    print(f"STORED: {len(points)} chunks for session '{session_id}'", flush=True)
    return len(points)

# ─── RETRIEVE (Python-side scan — 100% reliable) ──────────────────────────────
def retrieve_chunks(session_id: str) -> list:
    all_pts, _ = qclient().scroll(
        collection_name=COLLECTION,
        limit=500, with_payload=True, with_vectors=False
    )
    chunks = [
        p.payload.get("text", "")
        for p in all_pts
        if str(p.payload.get("session_id", "")) == session_id and p.payload.get("text")
    ]
    print(f"RETRIEVED: {len(chunks)} chunks for session '{session_id}' out of {len(all_pts)} total", flush=True)
    return chunks

# ─── LLM ──────────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            groq_api_key=GROQ_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0,
            streaming=True
        )
    return _llm

# ─── REQUEST MODEL ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question: str
    history: Optional[List] = []

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    errors = []
    if not GROQ_KEY:   errors.append("GROQ_API_KEY missing")
    if not QDRANT_URL: errors.append("QDRANT_URL missing")
    if not QDRANT_KEY: errors.append("QDRANT_API_KEY missing")
    try:
        cols = [c.name for c in qclient().get_collections().collections]
    except Exception as e:
        cols = [f"ERROR: {e}"]
    return {"status": "ok" if not errors else "error", "errors": errors, "collections": cols}


@app.get("/inspect/{session_id}")
async def inspect(session_id: str):
    all_pts, _ = qclient().scroll(
        collection_name=COLLECTION, limit=200, with_payload=True, with_vectors=False
    )
    matched = [
        {"id": str(p.id), "text_preview": p.payload.get("text","")[:120]}
        for p in all_pts
        if str(p.payload.get("session_id","")) == session_id
    ]
    all_ids = list(set(str(p.payload.get("session_id","")) for p in all_pts))
    return {
        "searched_session": session_id,
        "matched_count": len(matched),
        "all_session_ids": all_ids,
        "sample": matched[:3]
    }


@app.post("/upload")
async def upload(session_id: str, file: UploadFile = File(...)):
    if not GROQ_KEY or not QDRANT_URL:
        raise HTTPException(400, "Missing required environment variables")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    path = os.path.join(TEMP_DIR, file.filename)
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chunks = load_and_chunk(path)
        if not chunks:
            return {"chunks": 0, "message": "No content extracted from file"}

        n = store_chunks(chunks, str(session_id))
        return {"chunks": n, "message": f"Successfully indexed {n} chunks"}

    except Exception as e:
        tb = traceback.format_exc()
        print(f"UPLOAD ERROR:\n{tb}", flush=True)
        raise HTTPException(500, f"Upload failed: {str(e)}")
    finally:
        try: os.remove(path)
        except: pass


@app.post("/chat")
async def chat(req: ChatRequest):
    def stream():
        try:
            session_id = str(req.session_id)
            chunks = retrieve_chunks(session_id)

            if chunks:
                context = "\n\n---\n\n".join(chunks[:8])  # Use up to 8 chunks
            else:
                context = "NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT"

            prompt = ChatPromptTemplate.from_template(
                """You are a helpful AI assistant. Answer the QUESTION using only the CONTEXT below.
If the context says NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT, politely tell the user to upload a document first.

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
