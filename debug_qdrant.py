from qdrant_client import QdrantClient
from pathlib import Path

QDRANT_PATH = Path("qdrant_data")
COLLECTION_NAME = "local_rag"

def inspect_qdrant():
    client = QdrantClient(path=str(QDRANT_PATH))
    
    try:
        collections = client.get_collections().collections
        print(f"Collections: {[c.name for c in collections]}")
        
        if not collections:
            print("No collections found.")
            return

        # Get some points
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_payload=True,
            with_vectors=False,
        )
        
        print(f"\nFound {len(points)} points in {COLLECTION_NAME}:")
        for i, point in enumerate(points):
            print(f"\n--- Point {i} (ID: {point.id}) ---")
            print(f"Payload: {point.payload}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_qdrant()
