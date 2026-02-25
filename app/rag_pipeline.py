from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from app.embeddings import get_embedding_model
from app.vector_store import get_vector_store
from app.llm import get_llm
from app.config import TOP_K
from qdrant_client.http import models as rest

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

def build_rag_pipeline():
    emb, vs, llm = get_shared_components()
    
    prompt = ChatPromptTemplate.from_template(
        """You are a professional AI assistant. Answer the question using the provided context.
If no context is provided, explain that the user needs to upload a document first.

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
        
        print(f"DIAGNOSTIC: Searching for session_id='{session_id}'", flush=True)

        try:
            # BROAD FILTER: Check everywhere the ID could be stored
            qdrant_filter = rest.Filter(
                should=[
                    rest.FieldCondition(key="metadata.session_id", match=rest.MatchValue(value=session_id)),
                    rest.FieldCondition(key="session_id", match=rest.MatchValue(value=session_id))
                ]
            )
            
            # Step 1: Perform the filtered search
            docs = vs.similarity_search(question, k=TOP_K, filter=qdrant_filter)
            print(f"DIAGNOSTIC: Found {len(docs)} documents for this session.", flush=True)
            
            # Step 2: If nothing found, try matching by integer ID as a backup
            if not docs and session_id.isdigit():
                try:
                    num_id = int(session_id)
                    qdrant_filter_int = rest.Filter(
                        should=[
                            rest.FieldCondition(key="metadata.session_id", match=rest.MatchValue(value=num_id)),
                            rest.FieldCondition(key="session_id", match=rest.MatchValue(value=num_id))
                        ]
                    )
                    docs = vs.similarity_search(question, k=TOP_K, filter=qdrant_filter_int)
                    if docs:
                        print(f"DIAGNOSTIC: Found {len(docs)} docs using INTEGER ID fallback.", flush=True)
                except Exception: pass

            # Step 3: DESPERATE LOGGING - Log what's actually in there
            if not docs:
                print("DIAGNOSTIC: COLLECTION SCAN - Dumping first available point...", flush=True)
                res = vs.client.scroll(collection_name=vs.collection_name, limit=1, with_payload=True)[0]
                if res:
                    payload = res[0].payload
                    print(f"DIAGNOSTIC: Actual Session ID in DB: {payload.get('metadata', {}).get('session_id')}", flush=True)

            return format_docs(docs)
        except Exception as e:
            print(f"ERROR: Retrieval failed: {e}", flush=True)
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
        print(f"ERROR: Chat streaming failed: {e}", flush=True)
        yield f"Thinking failed: {e}"
