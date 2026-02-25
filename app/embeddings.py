import os
import requests
import time
from typing import List
from langchain_core.embeddings import Embeddings
from app.config import HF_EMBEDDING_MODEL, HUGGINGFACEHUB_API_TOKEN

class SafeHFInferenceEmbeddings(Embeddings):
    """
    A robust, manual implementation of Hugging Face Inference API embeddings.
    Strictly uses the new 'router' endpoint mandated by Hugging Face.
    """
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        # Updated to the new mandatory router endpoint format
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        max_retries = 3
        for i in range(max_retries):
            try:
                print(f"DEBUG: Calling HF Router API for {len(texts)} texts...", flush=True)
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and "error" in data:
                        error_msg = data.get("error", "")
                        if "loading" in error_msg.lower() and i < max_retries - 1:
                            print(f"DEBUG: Model is loading, waiting 15s (attempt {i+1})...", flush=True)
                            time.sleep(15)
                            continue
                        raise ValueError(f"HF API Error: {error_msg}")
                    else:
                        raise ValueError(f"Unexpected HF response format: {data}")
                
                # Handle common transient errors
                if response.status_code in [503, 504, 502, 429] and i < max_retries - 1:
                    print(f"DEBUG: HF API Status {response.status_code}, retrying in 15s...", flush=True)
                    time.sleep(15)
                    continue
                    
                if response.status_code != 200:
                    raise ValueError(f"HF API Failed ({response.status_code}): {response.text[:200]}")

            except requests.exceptions.Timeout:
                if i < max_retries - 1:
                    print("DEBUG: HF API Timeout, retrying...", flush=True)
                    time.sleep(5)
                    continue
                raise ValueError("HF API Request timed out. The model might be too large or the server is busy.")
            except Exception as e:
                if i < max_retries - 1:
                    print(f"DEBUG: HF API Exception: {e}, retrying...", flush=True)
                    time.sleep(10)
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
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
