from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from app.embeddings import get_embedding_model
from app.vector_store import get_vector_store
from app.llm import get_llm
from app.config import TOP_K
from qdrant_client.http import models as rest

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
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def format_chat_history(history):
    if not history: return ""
    formatted = []
    for msg in history:
        role = getattr(msg, "role", msg.get("role") if isinstance(msg, dict) else "user")
        content = getattr(msg, "content", msg.get("content") if isinstance(msg, dict) else str(msg))
        formatted.append(f"{role.capitalize()}: {content}")
    return "\n".join(formatted)

def build_rag_pipeline():
    emb, vs, llm = get_shared_components()
    
    prompt = ChatPromptTemplate.from_template(
        """You are a professional AI assistant. Answer the question using the provided context.
If the context is 'NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT', politely tell the user to upload a file.

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
        
        print(f"DEBUG: RAG Step - Session: {session_id}, Question: {question}", flush=True)

        try:
            # BROAD PYTHON-SIDE FILTERING (Extremely Reliable)
            # We fetch more documents than needed and filter them in Python to avoid Qdrant filter mismatches
            all_docs = vs.similarity_search(question, k=20) 
            
            # Filter docs for this session
            filtered_docs = []
            for doc in all_docs:
                doc_session = str(doc.metadata.get("session_id", ""))
                # Also check nested if present
                if not doc_session and "metadata" in doc.metadata:
                    doc_session = str(doc.metadata["metadata"].get("session_id", ""))
                
                if doc_session == session_id:
                    filtered_docs.append(doc)
            
            # If still nothing, try a broad "Retrieve All" for this session
            if not filtered_docs:
                print(f"DEBUG: No similarity matches. Trying Python-side session sweep...", flush=True)
                # Scroll the collection
                points, _ = vs.client.scroll(
                    collection_name=vs.collection_name,
                    limit=100,
                    with_payload=True,
                    with_vectors=False
                )
                
                for p in points:
                    p_session = str(p.payload.get("session_id") or p.payload.get("metadata", {}).get("session_id", ""))
                    if p_session == session_id:
                        from langchain_core.documents import Document
                        filtered_docs.append(Document(
                            page_content=p.payload.get("page_content", ""),
                            metadata=p.payload.get("metadata", p.payload)
                        ))
                
                # Keep only top bits
                filtered_docs = filtered_docs[:TOP_K]

            print(f"DEBUG: Final retrieval found {len(filtered_docs)} docs for session {session_id}", flush=True)
            return format_docs(filtered_docs)
            
        except Exception as e:
            print(f"ERROR in context retrieval: {str(e)}", flush=True)
            return "NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT"

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
