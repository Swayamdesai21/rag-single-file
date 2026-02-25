from langchain_huggingface import HuggingFaceEmbeddings
from app.config import HF_EMBEDDING_MODEL

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )



