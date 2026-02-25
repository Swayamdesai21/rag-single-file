from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import traceback
import json
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
        "environment": "vercel" if os.getenv("VERCEL") else "local",
        "temp_writable": os.access(TEMP_DIR, os.W_OK)
    }

@router.post("/chat")
async def chat(req: ChatRequest):
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
        err_msg = f"CHAT_ERROR: {str(e)}\n{traceback.format_exc()}"
        print(err_msg, flush=True)
        raise HTTPException(status_code=500, detail=err_msg)

@router.post("/upload")
async def initiate_upload(session_id: str, file: UploadFile = File(...)):
    config_errors = check_config()
    if config_errors:
        raise HTTPException(status_code=400, detail=f"Configuration error: {', '.join(config_errors)}")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {ext}"
        )

    file_path = os.path.join(TEMP_DIR, file.filename)
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest
        num_chunks = ingest_file(file_path, session_id=session_id, recreate=False)
        
        if num_chunks == 0:
            return {
                "message": "Warning: No content extracted",
                "filename": file.filename,
                "chunks": 0
            }

        reset_rag_pipeline()

        return {
            "message": "File processed successfully",
            "filename": file.filename,
            "chunks": num_chunks
        }
    except Exception as e:
        err_msg = f"UPLOAD_ERROR: {str(e)}\n{traceback.format_exc()}"
        print(err_msg, flush=True)
        raise HTTPException(status_code=500, detail=err_msg)
