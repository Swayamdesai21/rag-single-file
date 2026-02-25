from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from app.embeddings import get_embedding_model
from app.vector_store import get_vector_store
from app.llm import get_llm
from app.config import TOP_K, COLLECTION_NAME

# Global instances
_embedding_model = None
_vector_store = None
_llm = None
_rag_chain = None

def get_shared_components():
    global _embedding_model, _vector_store, _llm
    if _embedding_model is None:
        _embedding_model = get_embedding_model()
        _vector_store = get_vector_store(_embedding_model)
        _llm = get_llm()
    return _embedding_model, _vector_store, _llm

def format_docs(docs):
    if not docs:
        return "NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT"
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(history):
    if not history:
        return ""
    formatted = []
    for msg in history:
        role = getattr(msg, "role", msg.get("role") if isinstance(msg, dict) else "user")
        content = getattr(msg, "content", msg.get("content") if isinstance(msg, dict) else str(msg))
        formatted.append(f"{role.capitalize()}: {content}")
    return "\n".join(formatted)


def get_all_session_docs(vs, session_id: str) -> list:
    """
    Directly scans Qdrant collection and returns ALL docs for this session.
    Uses Python-side matching — no filter bugs, no type mismatches.
    No extra embedding calls — just a scan.
    """
    matched = []
    try:
        points, _ = vs.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=True,
            with_vectors=False
        )

        print(f"SCAN: {len(points)} total points in '{COLLECTION_NAME}'", flush=True)

        for pt in points:
            p = pt.payload
            # Retrieve session_id from wherever it might be stored
            stored_id = (
                str(p.get("session_id", "")) or
                str(p.get("metadata", {}).get("session_id", ""))
            )
            if stored_id == session_id:
                content = p.get("page_content", "")
                if content:
                    matched.append(Document(page_content=content, metadata={"session_id": session_id}))

        print(f"SCAN: {len(matched)} docs matched session_id='{session_id}'", flush=True)

        if not matched and points:
            # Log what's actually in the DB to help diagnose mismatch
            sample = points[0].payload
            sample_id = (
                sample.get("session_id") or
                sample.get("metadata", {}).get("session_id")
            )
            print(f"MISMATCH: DB has session_id='{sample_id}', searched for '{session_id}'", flush=True)

    except Exception as e:
        print(f"ERROR during scroll: {e}", flush=True)

    return matched[:TOP_K]


def build_rag_pipeline():
    emb, vs, llm = get_shared_components()

    prompt = ChatPromptTemplate.from_template(
        """You are a professional AI assistant. Answer based ONLY on the CONTEXT below.
If CONTEXT is 'NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT', ask the user to upload a document first.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

ANSWER:"""
    )

    def get_dynamic_context(input_data):
        session_id = str(input_data.get("session_id", "default"))
        question = input_data["question"]
        print(f"RAG: session='{session_id}' | question='{question}'", flush=True)

        # Use direct Python-side scan — bypasses all Qdrant filter issues
        docs = get_all_session_docs(vs, session_id)
        return format_docs(docs)

    return ({
        "context": RunnableLambda(get_dynamic_context),
        "question": RunnableLambda(lambda x: x["question"]),
        "chat_history": RunnableLambda(lambda x: format_chat_history(x.get("chat_history", []))),
    } | prompt | llm)


def reset_rag_pipeline():
    global _rag_chain, _embedding_model, _vector_store, _llm
    _rag_chain = None
    _embedding_model = None
    _vector_store = None
    _llm = None


def answer_question(question: str, chat_history: list = None, session_id: str = "default"):
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = build_rag_pipeline()

    try:
        for chunk in _rag_chain.stream({
            "session_id": session_id,
            "question": question,
            "chat_history": chat_history or [],
        }):
            yield getattr(chunk, "content", str(chunk))
    except Exception as e:
        print(f"STREAM ERROR: {e}", flush=True)
        yield f"Error: {e}"
