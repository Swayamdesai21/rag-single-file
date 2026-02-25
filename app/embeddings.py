from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from app.config import HF_EMBEDDING_MODEL, HUGGINGFACEHUB_API_TOKEN

def get_embedding_model():
    if HUGGINGFACEHUB_API_TOKEN:
        print("DEBUG: Using Hugging Face Inference API for embeddings", flush=True)
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=HUGGINGFACEHUB_API_TOKEN,
            model_name=HF_EMBEDDING_MODEL
        )
    else:
        print("DEBUG: Using Local Hugging Face embeddings (Warning: This will fail on Vercel without a token)", flush=True)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=HF_EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except ImportError:
            raise ImportError("HuggingFaceHUB_API_TOKEN not found and local 'langchain-huggingface' is not installed.")
