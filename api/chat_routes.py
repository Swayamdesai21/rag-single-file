from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from app.rag_pipeline import answer_question, reset_rag_pipeline
from app.ingester import ingest_file
from api.schemas import ChatRequest, ChatResponse
from fastapi.responses import StreamingResponse

router = APIRouter()

# Use /tmp for Vercel/Serverless environments
TEMP_DIR = "/tmp" if os.getenv("VERCEL") else "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md"}

@router.post("/chat")
async def chat(req: ChatRequest):
    print(f"DEBUG: Chat request: {req.question} (session: {req.session_id})", flush=True)
    try:
        def stream_answer():
            yield from answer_question(
                question=req.question,
                chat_history=req.history,
                session_id=req.session_id
            )
        return StreamingResponse(stream_answer(), media_type="text/plain")
    except Exception as e:
        print(f"ERROR in /chat: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def initiate_upload(session_id: str, file: UploadFile = File(...)):
    print(f"DEBUG: Upload request for session: {session_id}, file: {file.filename}", flush=True)
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
            raise Exception("No content could be extracted from the file.")

        reset_rag_pipeline()

        return {
            "message": "File processed successfully",
            "filename": file.filename,
            "chunks": num_chunks
        }
    except Exception as e:
        print(f"ERROR in /upload: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
