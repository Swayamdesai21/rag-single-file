from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, AsyncGenerator
import shutil, os, traceback, uuid, re, math, requests
from collections import defaultdict, Counter

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
HF_TOKEN   = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
SUPPORTED  = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md"}

# New hybrid collection — real 384-dim embeddings + payload text for BM25
COLLECTION   = "rag_hybrid"
VECTOR_SIZE  = 384          # BAAI/bge-small-en-v1.5 output dimension
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
EMBED_URL    = f"https://router.huggingface.co/hf-inference/models/{EMBED_MODEL}"
EMBED_BATCH  = 24           # chunks per HF API call (safe batch size)

# ─── QDRANT ───────────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
)

def new_qclient():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, timeout=30)

def ensure_collection(client: QdrantClient):
    cols = [c.name for c in client.get_collections().collections]
    if COLLECTION not in cols:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print(f"QDRANT: Created collection '{COLLECTION}'", flush=True)

# ─── SENTENCE TRANSFORMER EMBEDDINGS (HF API) ─────────────────────────────────
def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Call HF inference API (BAAI/bge-small-en-v1.5) to get 384-dim embeddings.
    Returns a list of embedding vectors.
    """
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "X-Wait-For-Model": "true",
    }
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    resp = requests.post(EMBED_URL, json=payload, headers=headers, timeout=45)
    resp.raise_for_status()
    return resp.json()  # list of 384-float lists

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed all texts in safe batches, returns flattened list of vectors."""
    all_vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        vecs = embed_batch(batch)
        all_vectors.extend(vecs)
        print(f"EMBED: batch {i//EMBED_BATCH + 1} done ({len(batch)} texts)", flush=True)
    return all_vectors

def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_batch([text])[0]

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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    return splitter.split_documents(docs)

# ─── STORE ────────────────────────────────────────────────────────────────────
def store_chunks(chunks: list, session_id: str) -> int:
    """
    Embed all chunks via Sentence Transformer (HF API) and store in Qdrant
    with real 384-dim vectors for semantic similarity search later.
    """
    sid = str(session_id)
    texts = [c.page_content for c in chunks if c.page_content.strip()]
    if not texts:
        return 0

    print(f"STORE: Embedding {len(texts)} chunks for session='{sid}'...", flush=True)
    vectors = embed_texts(texts)

    client = new_qclient()
    ensure_collection(client)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={"session_id": sid, "text": texts[i]}
        )
        for i in range(len(texts))
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"STORE: Saved {len(points)} chunks for session='{sid}'", flush=True)
    return len(points)

# ─── BM25 KEYWORD SCORING ─────────────────────────────────────────────────────
_STOP = {
    "a","an","the","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","must","can",
    "in","on","at","to","for","of","and","or","but","if","with","that","this",
    "what","which","who","how","when","where","me","my","i","you","your","it",
    "its","give","tell","explain","show","list","describe","about","please","need",
    "want","find","get","information","details","related","regarding","from","their"
}

def bm25_score(chunks: list[str], query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """
    Standard BM25 scoring for each chunk against the query.
    Returns a list of scores (one per chunk).
    """
    def tokenize(t):
        return [w for w in re.findall(r'\b\w+\b', t.lower()) if w not in _STOP and len(w) > 2]

    tok_chunks = [tokenize(c) for c in chunks]
    query_terms = tokenize(query)
    N = len(chunks)

    # IDF
    df = defaultdict(int)
    for tc in tok_chunks:
        for term in set(tc):
            df[term] += 1
    idf = {
        term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
        for term in query_terms
    }

    avg_len = sum(len(tc) for tc in tok_chunks) / max(N, 1)

    scores = []
    for tc in tok_chunks:
        tf_map = Counter(tc)
        dl = len(tc)
        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            idf_val = idf.get(term, 0)
            score += idf_val * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_len, 1)))
        scores.append(score)

    return scores

# ─── HYBRID RETRIEVE (Semantic + BM25 + RRF) ─────────────────────────────────
def hybrid_retrieve(session_id: str, query: str, top_k: int = 20) -> list[str]:
    """
    1. Embed query → Qdrant vector search (semantic, filtered by session_id)
    2. Scroll all session chunks → BM25 keyword scoring
    3. RRF fusion of both ranked lists → return top_k most relevant chunks
    """
    sid = str(session_id)
    client = new_qclient()

    # ── Check collection ──────────────────────────────────────────────────────
    cols = [c.name for c in client.get_collections().collections]
    if COLLECTION not in cols:
        print(f"HYBRID: Collection '{COLLECTION}' not found", flush=True)
        return []

    # ── Semantic: embed query + Qdrant search ─────────────────────────────────
    try:
        q_vec = embed_query(query)
        sem_hits = client.search(
            collection_name=COLLECTION,
            query_vector=q_vec,
            query_filter=Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=sid))]),
            limit=300,
            with_payload=True,
        )
        # Map text → semantic rank
        sem_ranks: dict[str, int] = {}
        sem_texts: list[str] = []
        for rank, hit in enumerate(sem_hits):
            t = hit.payload.get("text", "")
            if t.strip():
                sem_ranks[t] = rank
                sem_texts.append(t)
        print(f"HYBRID: Semantic hits={len(sem_texts)}", flush=True)
    except Exception as e:
        print(f"HYBRID: Semantic search failed: {e} — falling back to BM25 only", flush=True)
        sem_ranks = {}
        sem_texts = []

    # ── Keyword: BM25 over all session chunks ────────────────────────────────
    all_pts, _ = client.scroll(
        collection_name=COLLECTION, limit=10000,
        with_payload=True, with_vectors=False
    )
    all_texts = [
        p.payload.get("text", "")
        for p in all_pts
        if str(p.payload.get("session_id", "")) == sid and p.payload.get("text", "").strip()
    ]
    print(f"HYBRID: Total session chunks={len(all_texts)}", flush=True)

    if not all_texts:
        return []

    bm25_scores_list = bm25_score(all_texts, query)
    # Sort by BM25 descending → assign ranks
    bm25_ranked = sorted(range(len(all_texts)), key=lambda i: bm25_scores_list[i], reverse=True)
    bm25_ranks: dict[str, int] = {all_texts[i]: rank for rank, i in enumerate(bm25_ranked)}

    # ── RRF Fusion ───────────────────────────────────────────────────────────
    K = 60  # Standard RRF constant
    all_unique = list(dict.fromkeys(sem_texts + all_texts))  # preserve order, remove dupes

    rrf = {}
    for text in all_unique:
        sem_r = sem_ranks.get(text, len(all_unique))
        bm25_r = bm25_ranks.get(text, len(all_unique))
        rrf[text] = 1 / (K + sem_r) + 1 / (K + bm25_r)

    ranked = sorted(all_unique, key=lambda t: rrf[t], reverse=True)

    # Log top-3 for diagnostics
    for i, t in enumerate(ranked[:3]):
        print(f"HYBRID: Rank {i+1} | RRF={rrf[t]:.4f} | preview: {t[:60]}", flush=True)

    return ranked[:top_k]

# ─── LLM ──────────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def build_llm():
    return ChatGroq(
        groq_api_key=GROQ_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0,
        streaming=True,
        max_tokens=4000,
    )

# ─── REQUEST MODELS ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question: str
    history: Optional[List] = []

class TextUploadRequest(BaseModel):
    session_id: str
    text: str
    filename: str = "document.txt"

# ─── HEALTH ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    errors = []
    if not GROQ_KEY:   errors.append("GROQ_API_KEY missing")
    if not QDRANT_URL: errors.append("QDRANT_URL missing")
    if not QDRANT_KEY: errors.append("QDRANT_API_KEY missing")
    if not HF_TOKEN:   errors.append("HUGGINGFACEHUB_API_TOKEN missing")
    try:
        cols = [c.name for c in new_qclient().get_collections().collections]
    except Exception as e:
        cols = [f"ERROR: {e}"]
    return {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "collections": cols,
        "embed_model": EMBED_MODEL,
        "retrieval": "hybrid (BM25 + SentenceTransformer + RRF)"
    }

# ─── INSPECT / DEBUG ────────────────────────────────────────────────────────────
@app.post("/debug-chat")
async def debug_chat(req: ChatRequest):
    """
    Runs the full hybrid retrieval pipeline but instead of calling the LLM,
    returns the top 20 chunks with their exact semantic row ranks, BM25 ranks,
    and final RRF scores so we can debug why specific texts are missed.
    """
    sid = str(req.session_id)
    query = req.question

    client = new_qclient()
    sem_error = None
    try:
        q_vec = embed_query(query)
        sem_hits = client.search(
            collection_name=COLLECTION,
            query_vector=q_vec,
            query_filter=Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=sid))]),
            limit=300,
            with_payload=True,
        )
        sem_ranks = {hit.payload.get("text", ""): rank for rank, hit in enumerate(sem_hits) if hit.payload.get("text", "").strip()}
    except Exception as e:
        sem_ranks = {}
        sem_error = str(e)

    all_pts, _ = client.scroll(
        collection_name=COLLECTION, limit=10000,
        with_payload=True, with_vectors=False
    )
    all_texts = [p.payload.get("text", "") for p in all_pts if str(p.payload.get("session_id", "")) == sid and p.payload.get("text", "").strip()]

    bm25_scores_list = bm25_score(all_texts, query)
    bm25_ranked = sorted(range(len(all_texts)), key=lambda i: bm25_scores_list[i], reverse=True)
    bm25_ranks = {all_texts[i]: rank for rank, i in enumerate(bm25_ranked)}
    bm25_raw = {all_texts[i]: bm25_scores_list[i] for i in range(len(all_texts))}

    K = 60
    all_unique = list(dict.fromkeys(list(sem_ranks.keys()) + all_texts))
    rrf = {}
    for text in all_unique:
        sem_r = sem_ranks.get(text, len(all_unique))
        bm25_r = bm25_ranks.get(text, len(all_unique))
        rrf[text] = 1 / (K + sem_r) + 1 / (K + bm25_r)

    ranked = sorted(all_unique, key=lambda t: rrf[t], reverse=True)

    debug_results = []
    for t in ranked[:40]:
        debug_results.append({
            "text": t,
            "rrf_score": rrf[t],
            "semantic_rank": sem_ranks.get(t, -1),
            "bm25_rank": bm25_ranks.get(t, -1),
            "bm25_raw": bm25_raw.get(t, 0)
        })

    return {
        "session_id": sid,
        "query": query,
        "total_chunks_scrolled": len(all_texts),
        "semantic_error": sem_error,
        "results": debug_results
    }


@app.get("/debug-sessions")
async def debug_sessions():
    """Returns a list of all unique session IDs currently in the DB"""
    client = new_qclient()
    try:
        all_pts, _ = client.scroll(
            collection_name=COLLECTION, limit=10000,
            with_payload=True, with_vectors=False
        )
        sessions = list(set(str(p.payload.get("session_id", "")) for p in all_pts if p.payload.get("session_id")))
        return {"sessions": sessions}
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug-qclient")
async def debug_qclient():
    from qdrant_client import __version__ as qd_version
    client = new_qclient()
    return {
        "version": qd_version,
        "type": str(type(client)),
        "dir": dir(client)
    }


# ─── UPLOAD FILE (legacy — kept for DOCX/PPTX) ────────────────────────────────
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

# ─── UPLOAD TEXT (client-side extracted — no file size limit) ─────────────────
@app.post("/upload-text")
async def upload_text(req: TextUploadRequest):
    """
    Accepts text extracted client-side by PDF.js / JSZip.
    Chunks, embeds (Sentence Transformer via HF API), and stores in Qdrant.
    """
    if not GROQ_KEY or not QDRANT_URL:
        raise HTTPException(400, "Missing environment variables")
    if not req.text or len(req.text.strip()) < 10:
        raise HTTPException(400, "No meaningful text content provided")
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        raw_chunks = splitter.split_text(req.text)
        if not raw_chunks:
            return {"chunks": 0, "message": "No content to index"}

        from langchain_core.documents import Document
        docs = [Document(page_content=c) for c in raw_chunks]

        n = store_chunks(docs, str(req.session_id))
        return {"chunks": n, "message": f"Successfully indexed {n} chunks for session {req.session_id}"}
    except Exception as e:
        print(f"UPLOAD-TEXT ERROR:\n{traceback.format_exc()}", flush=True)
        raise HTTPException(500, f"Text indexing failed: {str(e)}")

# ─── CHAT ─────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = str(req.session_id)

    # Hybrid retrieval: Sentence Transformer semantic + BM25 keyword + RRF fusion
    top_chunks = hybrid_retrieve(session_id, req.question, top_k=20)
    print(f"CHAT: session='{session_id}' hybrid_chunks={len(top_chunks)}", flush=True)

    if not top_chunks:
        async def no_docs_stream() -> AsyncGenerator[str, None]:
            yield "It looks like no document has been uploaded for this chat session. Please upload a document using the **Upload & Index** button, then ask your question again."
        return StreamingResponse(no_docs_stream(), media_type="text/plain")

    context = "\n\n---\n\n".join(top_chunks)

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
