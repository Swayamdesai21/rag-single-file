import os
from langchain_groq import ChatGroq
from app.config import GROQ_LLM_MODEL, GROQ_API_KEY

def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_LLM_MODEL,
        temperature=0.2
    )

