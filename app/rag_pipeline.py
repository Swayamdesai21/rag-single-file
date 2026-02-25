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
        print("DEBUG: Initializing Shared RAG Components...", flush=True)
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

def build_rag_pipeline():
    emb, vs, llm = get_shared_components()
    
    prompt = ChatPromptTemplate.from_template(
        """You are a professional AI assistant. Answer the question based ONLY on the context.
If the context says 'NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT', tell the user to upload a file first.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

ANSWER:"""
    )

    def get_dynamic_context(input_data):
        session_id_val = input_data.get("session_id", "default")
        session_id = str(session_id_val)
        question = input_data["question"]
        
        print(f"DEBUG: Retrieval Attempt - Session: {session_id}, Question: {question}", flush=True)

        try:
            # DESPERATE FILTERING:
            # 1. Try 'metadata.session_id' as string
            # 2. Try 'metadata.session_id' as integer (if numeric)
            # 3. Try top-level 'session_id'
            
            conditions = [
                rest.FieldCondition(key="metadata.session_id", match=rest.MatchValue(value=session_id)),
                rest.FieldCondition(key="session_id", match=rest.MatchValue(value=session_id))
            ]
            
            # If the session_id is a numeric string, also try matching as integer
            if session_id.isdigit():
                try:
                    num_id = int(session_id)
                    conditions.append(rest.FieldCondition(key="metadata.session_id", match=rest.MatchValue(value=num_id)))
                    conditions.append(rest.FieldCondition(key="session_id", match=rest.MatchValue(value=num_id)))
                except: pass

            qdrant_filter = rest.Filter(should=conditions)
            
            # Use k=TOP_K
            docs = vs.similarity_search(question, k=TOP_K, filter=qdrant_filter)
            print(f"DEBUG: Step 1 - Found {len(docs)} docs for {session_id}", flush=True)
            
            # FATAL FALLBACK: If nothing found with similarity, try finding ANY doc for this session
            if not docs:
                print(f"DEBUG: Step 2 - Zero matches, trying 'ANY DOC' fallback...", flush=True)
                # Search with empty string and k=3 to get at least something
                docs = vs.similarity_search(" ", k=3, filter=qdrant_filter)
                print(f"DEBUG: Step 2 - Fallback found {len(docs)} docs", flush=True)
            
            # FINAL CHECK: If still nothing, check if the collection is even populated
            if not docs:
                all_points = vs.client.scroll(collection_name=vs.collection_name, limit=1)[0]
                if not all_points:
                    print(f"CRITICAL: Collection {vs.collection_name} is COMPLETELY EMPTY!", flush=True)
                else:
                    first_payload = all_points[0].payload
                    print(f"CRITICAL: Found other data ID={all_points[0].id}, Payload keys: {list(first_payload.keys())}", flush=True)
                    print(f"CRITICAL: Session mismatch. Database has session {first_payload.get('metadata', {}).get('session_id')} but searched for {session_id}", flush=True)

            return format_docs(docs)
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
    print("DEBUG: Full Pipeline Reset performed.", flush=True)

def answer_question(question: str, chat_history: list = None, session_id: str = "default"):
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = build_rag_pipeline()
    
    try:
        # Pass data through chain
        for chunk in _rag_chain.stream({
            "session_id": session_id,
            "question": question,
            "chat_history": chat_history or [],
        }):
            yield getattr(chunk, "content", str(chunk))
    except Exception as e:
        print(f"STREAM ERROR: {e}", flush=True)
        yield f"Error: {e}"
