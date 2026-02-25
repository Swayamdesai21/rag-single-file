from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, AsyncGenerator
import shutil, os, traceback, uuid, asyncio

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ─── ENV ──────────────────────────────────────────────────────────────────────
TEMP_DIR   = "/tmp"
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_KEY = os.getenv("QDRANT_API_KEY", "")
SUPPORTED  = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md"}

COLLECTION  = "rag_sessions"
VECTOR_SIZE = 1  # Dummy vector — retrieval is payload-based

# ─── QDRANT ───────────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

def new_qclient():
    """Create a fresh Qdrant client — never cached to avoid stale connections."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, timeout=30)

def ensure_collection(client: QdrantClient):
    cols = [c.name for c in client.get_collections().collections]
    if COLLECTION not in cols:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print(f"QDRANT: Created collection '{COLLECTION}'", flush=True)

# ─── DOCUMENT LOADING ─────────────────────────────────────────────────────────
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

# ─── STORE ────────────────────────────────────────────────────────────────────
def store_chunks(chunks: list, session_id: str) -> int:
    client = new_qclient()
    ensure_collection(client)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.0],
            payload={"session_id": str(session_id), "text": c.page_content}
        )
        for c in chunks if c.page_content.strip()
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"STORE: Saved {len(points)} chunks for session='{session_id}'", flush=True)
    return len(points)

# ─── RETRIEVE ─────────────────────────────────────────────────────────────────
def retrieve_chunks(session_id: str) -> list:
    """
    Always creates a fresh Qdrant client.
    Scrolls the collection and filters by session_id in Python.
    """
    sid = str(session_id)
    try:
        client = new_qclient()
        # Check collection exists
        cols = [c.name for c in client.get_collections().collections]
        print(f"RETRIEVE: Collections available: {cols}", flush=True)
        if COLLECTION not in cols:
            print(f"RETRIEVE: Collection '{COLLECTION}' not found!", flush=True)
            return []

        all_pts, _ = client.scroll(
            collection_name=COLLECTION,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        all_ids = list(set(str(p.payload.get("session_id","?")) for p in all_pts))
        print(f"RETRIEVE: Total points={len(all_pts)}, all session_ids={all_ids}", flush=True)

        matched = [
            p.payload.get("text","")
            for p in all_pts
            if str(p.payload.get("session_id","")) == sid
        ]
        print(f"RETRIEVE: Found {len(matched)} chunks for session='{sid}'", flush=True)
        return [m for m in matched if m.strip()]

    except Exception as e:
        print(f"RETRIEVE ERROR: {traceback.format_exc()}", flush=True)
        return []

# ─── LLM ──────────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def build_llm():
    return ChatGroq(
        groq_api_key=GROQ_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0,
        streaming=True
    )

# ─── REQUEST MODEL ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question: str
    history: Optional[List] = []

# ─── HEALTH ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    errors = []
    if not GROQ_KEY:   errors.append("GROQ_API_KEY missing")
    if not QDRANT_URL: errors.append("QDRANT_URL missing")
    if not QDRANT_KEY: errors.append("QDRANT_API_KEY missing")
    try:
        cols = [c.name for c in new_qclient().get_collections().collections]
    except Exception as e:
        cols = [f"ERROR: {e}"]
    return {"status": "ok" if not errors else "error", "errors": errors, "collections": cols}

# ─── DEBUG ────────────────────────────────────────────────────────────────────
@app.get("/inspect/{session_id}")
async def inspect(session_id: str):
    client = new_qclient()
    cols = [c.name for c in client.get_collections().collections]
    if COLLECTION not in cols:
        return {"error": f"Collection '{COLLECTION}' does not exist", "available": cols}
    
    all_pts, _ = client.scroll(
        collection_name=COLLECTION, limit=500,
        with_payload=True, with_vectors=False
    )
    matched = [
        {"id": str(p.id), "preview": p.payload.get("text","")[:120]}
        for p in all_pts
        if str(p.payload.get("session_id","")) == session_id
    ]
    all_ids = list(set(str(p.payload.get("session_id","")) for p in all_pts))
    return {
        "searched_session": session_id,
        "matched_count": len(matched),
        "all_session_ids_in_db": all_ids,
        "sample": matched[:3]
    }

# ─── UPLOAD ────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(session_id: str, file: UploadFile = File(...)):
    if not GROQ_KEY or not QDRANT_URL:
        raise HTTPException(400, "Missing environment variables")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    path = os.path.join(TEMP_DIR, file.filename)
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        chunks = load_and_chunk(path)
        if not chunks:
            return {"chunks": 0, "message": "No content extracted"}
        n = store_chunks(chunks, str(session_id))
        return {"chunks": n, "message": f"Successfully indexed {n} chunks for session {session_id}"}
    except Exception as e:
        print(f"UPLOAD ERROR:\n{traceback.format_exc()}", flush=True)
        raise HTTPException(500, f"Upload failed: {str(e)}")
    finally:
        try: os.remove(path)
        except: pass

# ─── CHAT ─────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = str(req.session_id)

    # Retrieve OUTSIDE the generator to catch errors early
    chunks = retrieve_chunks(session_id)
    context = "\n\n---\n\n".join(chunks[:8]) if chunks else "NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT"

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant. Answer the QUESTION using only the CONTEXT.
If CONTEXT is 'NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT', tell user to upload a document first.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    )
    llm = build_llm()
    chain = prompt | llm

    async def token_stream() -> AsyncGenerator[str, None]:
        try:
            async for chunk in chain.astream({"context": context, "question": req.question}):
                yield getattr(chunk, "content", str(chunk))
        except Exception as e:
            print(f"STREAM ERROR: {e}", flush=True)
            yield f"\n[Error: {e}]"

    return StreamingResponse(token_stream(), media_type="text/plain")
