from app.embeddings import get_embedding_model
from app.vector_store import get_vector_store
from langchain_core.documents import Document

import time

def test_flow():
    embedding = get_embedding_model()
    session_id = f"test_{int(time.time())}"
    
    print(f"--- Step 1: Ingesting for {session_id} ---")
    vs = get_vector_store(embedding, recreate=False)
    doc = Document(page_content="The secret code is 123456.", metadata={"session_id": session_id})
    vs.add_documents([doc])
    print("Ingestion done.")
    
    # We don't close the client because it's a singleton in our app
    
    print(f"\n--- Step 2: Retrieving for {session_id} ---")
    # Simulate a new request
    vs_new = get_vector_store(embedding, recreate=False)
    
    # Non-filtered search
    print("Searching without filter...")
    results_no_filter = vs_new.similarity_search("secret code", k=5)
    print(f"Found {len(results_no_filter)} docs without filter.")
    for r in results_no_filter:
        print(f" - Metadata: {r.metadata}")
        
    if results_no_filter:
        print("\nSearching WITH filter...")
        # Note: we need to use the filter structure retriever.py uses
        from qdrant_client.http import models as rest
        filter_obj = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="session_id",
                    match=rest.MatchValue(value=session_id)
                )
            ]
        )
        results_filter = vs_new.similarity_search("secret code", k=5, filter=filter_obj)
        print(f"Found {len(results_filter)} docs with filter.")

if __name__ == "__main__":
    test_flow()
