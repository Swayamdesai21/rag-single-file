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
        cols = [c.name for c in client.get_collections().collections]
        if COLLECTION not in cols:
            print(f"RETRIEVE: Collection '{COLLECTION}' not found!", flush=True)
            return []

        all_pts, _ = client.scroll(
            collection_name=COLLECTION,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )

        matched = [
            p.payload.get("text","")
            for p in all_pts
            if str(p.payload.get("session_id","")) == sid
        ]
        matched = [m for m in matched if m.strip()]
        print(f"RETRIEVE: Found {len(matched)} chunks for session='{sid}'", flush=True)
        return matched

    except Exception as e:
        print(f"RETRIEVE ERROR: {traceback.format_exc()}", flush=True)
        return []

# ─── LLM ──────────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import re

# Common English stop-words to ignore when extracting keywords
_STOP = {
    "a","an","the","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","must","can",
    "in","on","at","to","for","of","and","or","but","if","with","that","this",
    "what","which","who","how","when","where","me","my","i","you","your","it",
    "its","give","tell","explain","show","list","describe","about","please","need",
    "want","find","get","information","details","related","regarding","subject",
    "course","their","there","then","than","they","them","these","those","from"
}

def rank_chunks_by_relevance(chunks: list, question: str, top_k: int = 12) -> list:
    """
    Score every chunk by keyword frequency and return the top_k most relevant.
    Uses BM25-style term-frequency scoring — no external API needed.
    """
    # Extract meaningful keywords from the question
    words = re.findall(r'\b\w+\b', question.lower())
    keywords = [w for w in words if w not in _STOP and len(w) > 2]

    if not keywords:
        print("RANK: No meaningful keywords — returning first chunks", flush=True)
        return chunks[:top_k]

    def score(chunk: str) -> float:
        text = chunk.lower()
        total = 0.0
        for kw in keywords:
            count = text.count(kw)
            if count > 0:
                # TF-style: more occurrences = higher score, diminishing returns
                total += 1 + (count * 2)
        return total

    scored = [(score(c), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Log top matches for debugging
    top_scores = [(round(s, 1), c[:60]) for s, c in scored[:3]]
    print(f"RANK: keywords={keywords}, top_scores={top_scores}", flush=True)

    # Return chunks that scored > 0 first; fall back to top_k if all score 0
    ranked = [c for s, c in scored if s > 0]
    if not ranked:
        print("RANK: No keyword matches — returning first chunks as fallback", flush=True)
        return chunks[:top_k]

    return ranked[:top_k]


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

@app.get("/debug-retrieve/{session_id}")
async def debug_retrieve(session_id: str):
    """Calls retrieve_chunks directly and returns JSON — proves if function works."""
    chunks = retrieve_chunks(session_id)
    return {
        "session_id": session_id,
        "chunks_found": len(chunks),
        "first_chunk_preview": chunks[0][:200] if chunks else None
    }


# ─── UPLOAD TEXT (client-side extracted — no file size limit) ──────────────────
class TextUploadRequest(BaseModel):
    session_id: str
    text: str
    filename: str = "document.txt"

@app.post("/upload-text")
async def upload_text(req: TextUploadRequest):
    """
    Accepts text extracted client-side (e.g., by PDF.js in the browser).
    No request body size limit for the PDF itself — only the text is sent.
    Works for any size PDF.
    """
    if not GROQ_KEY or not QDRANT_URL:
        raise HTTPException(400, "Missing environment variables")
    if not req.text or len(req.text.strip()) < 10:
        raise HTTPException(400, "No meaningful text content provided")

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        raw_chunks = splitter.split_text(req.text)
        if not raw_chunks:
            return {"chunks": 0, "message": "No content to index"}

        # Build pseudo-documents
        from langchain_core.documents import Document
        docs = [Document(page_content=c) for c in raw_chunks]

        n = store_chunks(docs, str(req.session_id))
        return {"chunks": n, "message": f"Successfully indexed {n} chunks for session {req.session_id}"}

    except Exception as e:
        print(f"UPLOAD-TEXT ERROR:\n{traceback.format_exc()}", flush=True)
        raise HTTPException(500, f"Text indexing failed: {str(e)}")


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

    # 1. Retrieve ALL chunks for this session from Qdrant
    all_chunks = retrieve_chunks(session_id)
    print(f"CHAT: session='{session_id}' total_chunks={len(all_chunks)}", flush=True)

    # 2. Handle no-docs case BEFORE calling LLM — avoids hallucination
    if not all_chunks:
        async def no_docs_stream() -> AsyncGenerator[str, None]:
            yield "It looks like no document has been uploaded for this chat session. Please upload a document using the **Upload & Index** button, then ask your question again."
        return StreamingResponse(no_docs_stream(), media_type="text/plain")

    # 3. Rank chunks by keyword relevance to this specific question
    #    For large PDFs (200+ pages), this finds the right page instead of returning first 8.
    top_chunks = rank_chunks_by_relevance(all_chunks, req.question, top_k=12)
    context = "\n\n---\n\n".join(top_chunks)
    print(f"CHAT: Using {len(top_chunks)} ranked chunks for context", flush=True)

    # 4. Build prompt with only relevant context
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant. Use ONLY the CONTEXT below to answer the QUESTION.
Be specific and cite the exact information from the context.
If the answer is not in the context, say "I couldn't find this in the uploaded document."

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




@app.post("/debug-chat")
async def debug_chat(req: ChatRequest):
    """Non-streaming version that returns JSON showing what retrieve_chunks found."""
    session_id = str(req.session_id)
    chunks = retrieve_chunks(session_id)
    return {
        "session_id": session_id,
        "chunks_found": len(chunks),
        "context_preview": chunks[0][:300] if chunks else None,
        "will_answer": len(chunks) > 0
    }

