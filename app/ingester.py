from app.document_loader import load_document
from app.chunker import chunk_documents
from app.embeddings import get_embedding_model
from app.vector_store import get_vector_store
import os

def ingest_file(file_path: str, session_id: str, recreate: bool = False):
    """
    Ingests a file (PDF, DOCX, PPTX, TXT, MD) into the local Qdrant vector store.
    Associated with a specific session_id for isolation.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Load Documents
    documents = load_document(file_path)
    
    if not documents:
        return 0

    # 2. Add session_id to metadata for filtering
    print(f"DEBUG: Ingesting file for session_id: {session_id}", flush=True)
    for doc in documents:
        doc.metadata["session_id"] = session_id


    # 3. Chunk Documents
    chunks = chunk_documents(documents)

    if not chunks:
        return 0
    
    if len(chunks) > 0:
        print(f"DEBUG: First chunk metadata: {chunks[0].metadata}", flush=True)



    # 4. Get Models & Store
    embedding_model = get_embedding_model()
    # We set recreate=False so we don't wipe out other sessions' data
    vector_store = get_vector_store(embedding_model, recreate=recreate)

    # 5. Add to Vector Store
    vector_store.add_documents(chunks)

    return len(chunks)

