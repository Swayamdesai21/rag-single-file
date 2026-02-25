import os
import requests
import time
from typing import List
from langchain_core.embeddings import Embeddings
from app.config import HF_EMBEDDING_MODEL, HUGGINGFACEHUB_API_TOKEN

class SafeHFInferenceEmbeddings(Embeddings):
    """
    A robust implementation for generating embeddings via Hugging Face Inference API.
    Specifically designed to handle the 'feature-extraction' task.
    """
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Wait-For-Model": "true",
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Ensure texts are strings
        texts = [str(t) for t in texts]

        max_retries = 3
        for i in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": texts},
                    timeout=60 # Long timeout for cold starts
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if isinstance(data, list):
                        # Ensure we have a 2D list
                        if len(data) > 0:
                            if isinstance(data[0], list):
                                if len(data[0]) > 0 and isinstance(data[0][0], list):
                                    # Handle 3D tensor [batch, tokens, dim] with mean pooling
                                    pooled = []
                                    for doc in data:
                                        dim = len(doc[0])
                                        mean_vec = [sum(col)/len(doc) for col in zip(*doc)]
                                        pooled.append(mean_vec)
                                    return pooled
                                return data # Already [batch, dim]
                                
                    elif isinstance(data, dict) and "error" in data:
                        error_msg = data.get("error", "")
                        if "loading" in error_msg.lower() and i < max_retries - 1:
                            print("DEBUG: Model loading, waiting 15s...", flush=True)
                            time.sleep(15)
                            continue
                        raise ValueError(f"HF API Error: {error_msg}")

                if response.status_code in [500, 502, 503, 504] and i < max_retries - 1:
                    print(f"DEBUG: HF API Status {response.status_code}, retrying...", flush=True)
                    time.sleep(10)
                    continue
                    
                raise ValueError(f"HF API Failed ({response.status_code}): {response.text[:200]}")

            except Exception as e:
                if i < max_retries - 1:
                    time.sleep(5)
                    continue
                raise e
        
        return []

    def embed_query(self, text: str) -> List[float]:
        res = self.embed_documents([str(text)])
        return res[0] if res else []

def get_embedding_model():
    if HUGGINGFACEHUB_API_TOKEN:
        return SafeHFInferenceEmbeddings(
            api_key=HUGGINGFACEHUB_API_TOKEN,
            model_name=HF_EMBEDDING_MODEL
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
