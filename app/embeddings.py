import os
import requests
import time
from typing import List
from langchain_core.embeddings import Embeddings
from app.config import HF_EMBEDDING_MODEL, HUGGINGFACEHUB_API_TOKEN

class SafeHFInferenceEmbeddings(Embeddings):
    """
    A robust implementation for generating embeddings via Hugging Face Inference API.
    Specifically designed to handle the 'feature-extraction' task and bypass
    the 'sentence-similarity' pipeline errors.
    """
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        # Using the explicit feature-extraction task endpoint to avoid similarity pipeline errors
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Wait-For-Model": "true"
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        max_retries = 3
        for i in range(max_retries):
            try:
                # The payload for feature-extraction is simply the list of strings
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": texts},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Feature extraction can return a 3D list [batch, tokens, dim] 
                    # OR a 2D list [batch, dim] depending on the model/pooling.
                    # We need to ensure we return a 2D list of averages if it's 3D.
                    if isinstance(data, list):
                        if len(data) > 0 and isinstance(data[0], list):
                            # Check if it's the 3D case (hidden states for each token)
                            if len(data[0]) > 0 and isinstance(data[0][0], list):
                                # Perform simple mean pooling if model doesn't do it
                                pooled_embeddings = []
                                for doc_states in data:
                                    # Average across the token dimension (axis 1)
                                    dim = len(doc_states[0])
                                    mean_vec = [sum(col) / len(doc_states) for col in zip(*doc_states)]
                                    pooled_embeddings.append(mean_vec)
                                return pooled_embeddings
                            return data # Already 2D [batch, dim]
                    
                    elif isinstance(data, dict) and "error" in data:
                        error_msg = data.get("error", "")
                        if "loading" in error_msg.lower() and i < max_retries - 1:
                            print(f"DEBUG: Model is loading, waiting 20s...", flush=True)
                            time.sleep(20)
                            continue
                        raise ValueError(f"HF API Error: {error_msg}")

                # Handle specific failure cases
                if response.status_code == 400 and "similarity" in response.text.lower():
                    # If it STILL tries similarity, the model metadata is likely forcing it.
                    # Switching models in config.py is the best solution for this.
                    raise ValueError(f"HF Model Configuration Conflict: This model is defaulting to Similarity instead of Embeddings. Please use BAAI/bge-small-en-v1.5.")

                if response.status_code in [503, 504, 502, 429] and i < max_retries - 1:
                    time.sleep(15)
                    continue
                    
                if response.status_code != 200:
                    raise ValueError(f"HF API Failed ({response.status_code}): {response.text[:200]}")

            except Exception as e:
                if i < max_retries - 1:
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
