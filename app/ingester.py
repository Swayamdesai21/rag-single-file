from app.document_loader import load_document
from app.chunker import chunk_documents
from app.embeddings import get_embedding_model
from app.vector_store import get_vector_store
import os

def ingest_file(file_path: str, session_id: str, recreate: bool = False):
    """
    Ingests a file into the Qdrant vector store with ultra-simple metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Load Documents
    documents = load_document(file_path)
    if not documents:
        return 0

    # 2. Add session_id as a simple string
    for doc in documents:
        doc.metadata = {"session_id": str(session_id)}

    # 3. Chunk Documents
    chunks = chunk_documents(documents)
    if not chunks:
        return 0
    
    # 4. Clean up metadata for all chunks
    for chunk in chunks:
        chunk.metadata = {"session_id": str(session_id)}

    # 5. Get Models & Store
    embedding_model = get_embedding_model()
    vector_store = get_vector_store(embedding_model, recreate=recreate)

    # 6. Add to Vector Store
    print(f"DEBUG: Manually adding {len(chunks)} chunks for session {session_id}", flush=True)
    vector_store.add_documents(chunks)

    return len(chunks)
