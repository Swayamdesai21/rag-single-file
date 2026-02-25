import os
import requests
import time
from typing import List
from langchain_core.embeddings import Embeddings
from app.config import HF_EMBEDDING_MODEL, HUGGINGFACEHUB_API_TOKEN

class SafeHFInferenceEmbeddings(Embeddings):
    """
    A robust, manual implementation of Hugging Face Inference API embeddings.
    Inherits from LangChain Embeddings base class for strict type compatibility.
    """
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        # Using the standard Inference API endpoint structure
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        max_retries = 3
        for i in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=20
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and "error" in data:
                        error_msg = data.get("error", "")
                        if "loading" in error_msg.lower() and i < max_retries - 1:
                            time.sleep(10) # Wait longer for loading
                            continue
                        raise ValueError(f"HF API Error: {error_msg}")
                
                # If we get a 503 (Model loading) or other transient errors
                if response.status_code in [503, 504, 502] and i < max_retries - 1:
                    time.sleep(10)
                    continue
                    
                if response.status_code != 200:
                    raise ValueError(f"HF API Failed ({response.status_code}): {response.text[:200]}")

            except Exception as e:
                if i < max_retries - 1:
                    time.sleep(5)
                    continue
                raise e
        
        return []

    def embed_query(self, text: str) -> List[float]:
        res = self.embed_documents([text])
        if res and len(res) > 0:
            return res[0]
        return []

def get_embedding_model():
    if HUGGINGFACEHUB_API_TOKEN:
        return SafeHFInferenceEmbeddings(
            api_key=HUGGINGFACEHUB_API_TOKEN,
            model_name=HF_EMBEDDING_MODEL
        )
    else:
        # Fallback for local
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
