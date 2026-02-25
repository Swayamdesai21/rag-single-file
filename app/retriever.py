from qdrant_client.http import models as rest
from app.config import TOP_K

def get_retriever(vector_store, session_id: str = None):
    search_kwargs = {"k": TOP_K}
    
    # if session_id:
    #     # Correctly construct the Qdrant filter object
    #     # LangChain's QdrantVectorStore puts metadata fields inside a 'metadata' nested object
    #     search_kwargs["filter"] = rest.Filter(
    #         must=[
    #             rest.FieldCondition(
    #                 key="metadata.session_id", 
    #                 match=rest.MatchValue(value=session_id)
    #             )
    #         ]
    #     )



    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
