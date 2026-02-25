from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import traceback
from app.rag_pipeline import answer_question, reset_rag_pipeline
from app.ingester import ingest_file
from app.config import check_config
from api.schemas import ChatRequest, ChatResponse
from fastapi.responses import StreamingResponse

router = APIRouter()

# Use /tmp for Vercel/Serverless environments
TEMP_DIR = "/tmp" if os.getenv("VERCEL") else "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md"}

@router.get("/health")
async def health():
    config_errors = check_config()
    return {
        "status": "ok" if not config_errors else "unconfigured",
        "config_errors": config_errors,
        "environment": "vercel" if os.getenv("VERCEL") else "local"
    }

@router.post("/chat")
async def chat(req: ChatRequest):
    print(f"DEBUG: Chat request: {req.question} (session: {req.session_id})", flush=True)
    config_errors = check_config()
    if config_errors:
        raise HTTPException(status_code=400, detail=f"Configuration error: {', '.join(config_errors)}")
        
    try:
        def stream_answer():
            yield from answer_question(
                question=req.question,
                chat_history=req.history,
                session_id=req.session_id
            )
        return StreamingResponse(stream_answer(), media_type="text/plain")
    except Exception as e:
        print(f"ERROR in /chat: {e}\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def initiate_upload(session_id: str, file: UploadFile = File(...)):
    print(f"DEBUG: Upload request for session: {session_id}, file: {file.filename}", flush=True)
    
    config_errors = check_config()
    if config_errors:
        raise HTTPException(status_code=400, detail=f"Configuration error: {', '.join(config_errors)}")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    file_path = os.path.join(TEMP_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        num_chunks = ingest_file(file_path, session_id=session_id, recreate=False)
        
        if num_chunks == 0:
            raise Exception("No content could be extracted from the file. The file might be corrupted or empty.")

        reset_rag_pipeline()

        return {
            "message": "File processed successfully",
            "filename": file.filename,
            "chunks": num_chunks
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"ERROR in /upload: {e}\n{error_trace}", flush=True)
        # Return a more descriptive error if possible
        error_msg = str(e)
        if "Authentication" in error_msg or "401" in error_msg:
            error_msg = "Authentication failed. Please check your API keys (Groq, Qdrant, or Hugging Face)."
        elif "Timeout" in error_msg:
            error_msg = "The request timed out. This often happens on Vercel's free tier with large files. Try a smaller file."
        
        raise HTTPException(status_code=500, detail=error_msg)
