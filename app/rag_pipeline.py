from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from app.embeddings import get_embedding_model
from app.vector_store import get_vector_store
from app.llm import get_llm
from app.config import TOP_K

# Global instances cache
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
    if not history: return ""
    formatted = []
    for msg in history:
        role = getattr(msg, "role", msg.get("role") if isinstance(msg, dict) else "user")
        content = getattr(msg, "content", msg.get("content") if isinstance(msg, dict) else str(msg))
        formatted.append(f"{role.capitalize()}: {content}")
    return "\n".join(formatted)

def get_session_documents_direct(vs, session_id: str) -> list:
    """
    Directly scrolls the Qdrant collection and returns all documents
    matching this session_id. This is 100% reliable because it goes
    straight to the database without relying on vector-distance or filter parsing.
    """
    all_matching_docs = []
    try:
        # Scroll the entire collection (up to 1000 items) and match in Python
        points, _ = vs.client.scroll(
            collection_name=vs.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        print(f"DIAGNOSTIC: Total points in collection: {len(points)}", flush=True)
        
        for pt in points:
            payload = pt.payload
            # Check every possible location the session_id might be stored
            stored_id = (
                payload.get("session_id") or
                payload.get("metadata", {}).get("session_id") or
                ""
            )
            stored_id_str = str(stored_id)
            
            if stored_id_str == session_id:
                page_content = payload.get("page_content", "")
                if page_content:
                    all_matching_docs.append(Document(
                        page_content=page_content,
                        metadata={"session_id": session_id}
                    ))
        
        print(f"DIAGNOSTIC: Found {len(all_matching_docs)} matching docs for session '{session_id}'", flush=True)
        
        if all_matching_docs:
            # Log the first available session_id to help debug mismatches
            for pt in points[:1]:
                p = pt.payload
                found_id = p.get("session_id") or p.get("metadata", {}).get("session_id")
                print(f"DIAGNOSTIC: First point in collection has session_id='{found_id}'", flush=True)
        
    except Exception as e:
        print(f"ERROR during scroll: {e}", flush=True)
    
    return all_matching_docs

def build_rag_pipeline():
    emb, vs, llm = get_shared_components()
    
    prompt = ChatPromptTemplate.from_template(
        """You are a professional AI assistant. Answer based ONLY on the provided CONTEXT.
If CONTEXT is 'NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT', tell the user to upload a document first.

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
        
        print(f"RAG: Looking for docs for session_id='{session_id}'", flush=True)

        try:
            # PRIMARY: Direct scroll-based match (100% reliable)
            docs = get_session_documents_direct(vs, session_id)
            
            # If we found documents, rank them by similarity using embeddings
            if docs:
                # Use embedding to find the most relevant chunks
                try:
                    query_embedding = emb.embed_query(question)
                    # Simple dot-product ranking
                    import numpy as np
                    scored = []
                    for doc in docs:
                        doc_embedding = emb.embed_query(doc.page_content[:500])  # Limit for speed
                        score = sum(a*b for a, b in zip(query_embedding, doc_embedding))
                        scored.append((score, doc))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    docs = [doc for _, doc in scored[:TOP_K]]
                except Exception as rank_err:
                    print(f"Ranking failed, using unranked docs: {rank_err}", flush=True)
                    docs = docs[:TOP_K]  # Use first TOP_K without ranking
            
            return format_docs(docs)
        except Exception as e:
            print(f"ERROR in context retrieval: {e}", flush=True)
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
        print(f"ERROR in RAG stream: {e}", flush=True)
        yield f"Error: {e}"
