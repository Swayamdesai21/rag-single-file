from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from app.config import COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_DIMENSION

VECTOR_SIZE = EMBEDDING_DIMENSION
QDRANT_PATH = Path("qdrant_data")

_client_instance = None

def get_qdrant_client():
    global _client_instance
    if _client_instance is None:
        if QDRANT_URL:
            _client_instance = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            _client_instance = QdrantClient(path=str(QDRANT_PATH))
    return _client_instance


def get_vector_store(embedding_model, recreate: bool = False):
    client = get_qdrant_client()
    
    try:
        collections = [c.name for c in client.get_collections().collections]
    except Exception:
        collections = []

    if recreate or COLLECTION_NAME not in collections:
        if COLLECTION_NAME in collections:
            client.delete_collection(collection_name=COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
        # Removed explicit metadata_payload_key to return to standard LangChain behavior
    )
