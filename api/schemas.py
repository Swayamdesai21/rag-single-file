from pydantic import BaseModel
from typing import List

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    question: str
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str
