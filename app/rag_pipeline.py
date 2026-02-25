from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from app.embeddings import get_embedding_model
from app.vector_store import get_vector_store
from app.llm import get_llm
from app.config import TOP_K
from qdrant_client.http import models as rest

# Global instances for lazy initialization
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
    # Join documents with clear separators
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
        """You are a highly capable AI assistant. Answer the question using the provided context.
If the context is empty or says 'NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT', politely explain that the user needs to upload a document first.

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
        print(f"DEBUG: Retrieval Attempt - Session: {session_id}, Question: {question}", flush=True)

        try:
            # SUPER BROAD FILTER: Search everywhere session_id might be hidden
            qdrant_filter = rest.Filter(
                should=[
                    rest.FieldCondition(key="session_id", match=rest.MatchValue(value=session_id)),
                    rest.FieldCondition(key="metadata.session_id", match=rest.MatchValue(value=session_id)),
                    rest.FieldCondition(key="metadata.metadata.session_id", match=rest.MatchValue(value=session_id))
                ]
            )
            
            # Step 1: Try strict filtered search
            docs = vs.similarity_search(question, k=TOP_K, filter=qdrant_filter)
            print(f"DEBUG: Standard search found {len(docs)} docs", flush=True)
            
            # Step 2: Fallback - Search for everything in this session (ignore query relevance)
            if not docs or len(docs) == 0:
                print(f"DEBUG: No docs for this session found via similarity. Trying 'Retrieve All' for session...", flush=True)
                # Query with empty space to match everything, but keep the session filter
                docs = vs.similarity_search(" ", k=5, filter=qdrant_filter)
                print(f"DEBUG: 'Retrieve All' fallback found {len(docs)} docs", flush=True)

            # Step 3: Extreme Diagnostic - Search the WHOLE collection (no session filter)
            # Only do this to log what's actually in the DB if we are still getting 0
            if not docs or len(docs) == 0:
                print("DEBUG: EXTREME CHECK - Searching WHOLE collection without filters...", flush=True)
                all_data = vs.similarity_search(" ", k=1)
                if all_data:
                    found_payload = all_data[0].metadata
                    print(f"DIAGNOSTIC: Found data for a DIFFERENT session! Stored ID is: {found_payload.get('session_id')}", flush=True)
                else:
                    print("DIAGNOSTIC: The collection is COMPLETELY EMPTY!", flush=True)

            return format_docs(docs)
        except Exception as e:
            print(f"ERROR in context retrieval: {e}", flush=True)
            return f"Error connecting to knowledge base: {str(e)}"

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
        print(f"ERROR in RAG chain: {e}", flush=True)
        yield f"Error generating response: {e}"
