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
    """Lazily initialize RAG components to prevent crash on import."""
    global _embedding_model, _vector_store, _llm
    if _embedding_model is None:
        _embedding_model = get_embedding_model()
    if _vector_store is None:
        _vector_store = get_vector_store(_embedding_model)
    if _llm is None:
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
        """You are a highly capable AI assistant. Your goal is to provide accurate, well-structured, and easy-to-read answers based ONLY on the provided context.

GUIDELINES:
1. Use **bold text** for key terms, names, or important concepts.
2. Use bullet points or numbered lists if there are multiple steps, features, or items.
3. Use headers (###) to separate different sections of a long answer.
4. If the context is 'NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT', politely ask the user to upload a document.
5. Keep the tone professional yet helpful.

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
        print(f"DEBUG: Retrieval for session_id: {session_id}, question: {question}", flush=True)

        try:
            # Use strict filter
            qdrant_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.session_id",
                        match=rest.MatchValue(value=session_id)
                    )
                ]
            )
            
            docs = vs.similarity_search(question, k=TOP_K, filter=qdrant_filter)
            print(f"DEBUG: Found {len(docs)} documents with filter", flush=True)
            
            if not docs:
                all_session_docs = vs.similarity_search("", k=1, filter=qdrant_filter)
                if all_session_docs:
                    print(f"DEBUG: Found documents for this session, but none matched the query well.", flush=True)
                else:
                    print(f"DEBUG: Absolutely no documents found for session_id: {session_id}", flush=True)
            
            return format_docs(docs)
        except Exception as e:
            print(f"Error in context retrieval: {e}", flush=True)
            return "NO_DOCUMENTS_UPLOADED_FOR_THIS_CHAT"

    return ({
        "context": RunnableLambda(get_dynamic_context),
        "question": RunnableLambda(lambda x: x["question"]),
        "chat_history": RunnableLambda(lambda x: format_chat_history(x.get("chat_history", []))),
    } | prompt | llm)

def reset_rag_pipeline():
    global _rag_chain
    _rag_chain = None

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
