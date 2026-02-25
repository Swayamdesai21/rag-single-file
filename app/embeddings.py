import os
import requests
import time
from typing import List
from app.config import HF_EMBEDDING_MODEL, HUGGINGFACEHUB_API_TOKEN

class SafeHFInferenceEmbeddings:
    """
    A robust, manual implementation of Hugging Face Inference API embeddings.
    This avoids issues with library-specific URL construction and provides
    better error handling for Vercel/Serverless environments.
    """
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        # The new router endpoint as requested by the HF error message
        self.api_url = f"https://router.huggingface.co/models/{model_name}"
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
                
                # Check for 404 (wrong model named/url) or other errors
                if response.status_code == 404:
                    # Fallback to the older endpoint if the router fails with 404
                    fallback_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
                    print(f"DEBUG: Router 404, trying fallback URL: {fallback_url}", flush=True)
                    response = requests.post(
                        fallback_url,
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
                            print(f"DEBUG: Model loading, retry {i+1}...", flush=True)
                            time.sleep(5)
                            continue
                        raise ValueError(f"HF API Error: {error_msg}")
                    else:
                        raise ValueError(f"Unexpected HF response format: {data}")
                
                # If we get a non-200 but it's not a loading error
                print(f"DEBUG: HF API Status {response.status_code}, Body: {response.text[:200]}", flush=True)
                
                if i < max_retries - 1:
                    time.sleep(2)
                    continue
                
                raise ValueError(f"HF API Failed (Status {response.status_code}): {response.text[:200]}")

            except requests.exceptions.JSONDecodeError:
                if i < max_retries - 1:
                    print("DEBUG: JSON Decode Error (likely empty response), retrying...", flush=True)
                    time.sleep(2)
                    continue
                raise ValueError("HF API returned non-JSON response. Check your API Token and Model accessibility.")
            except Exception as e:
                if i < max_retries - 1:
                    time.sleep(2)
                    continue
                raise e
        
        return []

    def embed_query(self, text: str) -> List[float]:
        res = self.embed_documents([text])
        if res and len(res) > 0:
            return res[0]
        return []

    # These help LangChain treat this as a valid embedding object
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

def get_embedding_model():
    if HUGGINGFACEHUB_API_TOKEN:
        print("DEBUG: Using Manual Robust HF Inference implementation", flush=True)
        return SafeHFInferenceEmbeddings(
            api_key=HUGGINGFACEHUB_API_TOKEN,
            model_name=HF_EMBEDDING_MODEL
        )
    else:
        print("DEBUG: Fallback to local (Will fail on Vercel)", flush=True)
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
