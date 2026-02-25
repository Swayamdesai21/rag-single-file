from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from app.config import HF_EMBEDDING_MODEL, HUGGINGFACEHUB_API_TOKEN
from typing import List
import time

class SafeHFInferenceEmbeddings(HuggingFaceInferenceAPIEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Retry logic for 'model loading' and clear error reporting."""
        max_retries = 3
        for i in range(max_retries):
            try:
                res = super().embed_documents(texts)
                # If the result is a dict (like {'error': '...'}), this will help us see it
                if isinstance(res, dict):
                    error_msg = res.get("error", str(res))
                    if "currently loading" in error_msg.lower() and i < max_retries - 1:
                        print(f"DEBUG: HF Model is loading (attempt {i+1})...", flush=True)
                        time.sleep(5)
                        continue
                    raise ValueError(f"Hugging Face API Error: {error_msg}")
                
                # Ensure it's a list and not empty
                if not isinstance(res, list) or len(res) == 0:
                    raise ValueError(f"Invalid response format from HF API: {res}")
                
                return res
            except Exception as e:
                # Catch the specific KeyError: 0 which happens when it returns a dict
                if "0" in str(e) and i < max_retries - 1:
                    print(f"DEBUG: HF API returned unexpected format (possibly loading), retrying...", flush=True)
                    time.sleep(5)
                    continue
                raise e
        return []

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def get_embedding_model():
    if HUGGINGFACEHUB_API_TOKEN:
        print("DEBUG: Using Safe Hugging Face Inference API wrapper", flush=True)
        return SafeHFInferenceEmbeddings(
            api_key=HUGGINGFACEHUB_API_TOKEN,
            model_name=HF_EMBEDDING_MODEL
        )
    else:
        print("DEBUG: Using Local HF embeddings (Fallback)", flush=True)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=HF_EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except ImportError:
            raise ImportError("HuggingFaceHUB_API_TOKEN missing and 'langchain-huggingface' not installed.")
