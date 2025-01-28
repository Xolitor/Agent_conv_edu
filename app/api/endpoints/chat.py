# api/chat.py
"""
Routes FastAPI pour le chatbot
Inclut les endpoints du TP1 et du TP2
"""
from fastapi import APIRouter, HTTPException, Body
from models.chat import ChatRequest, ChatResponse, ChatRequestWithCourseData, ChatRequestTP1
from services.llm_service import LLMService
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File
from typing import List
import shutil
from pathlib import Path
router = APIRouter()
llm_service = LLMService()

#################### endpoint pour le chatbot de base ####################


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Nouvel endpoint du TP2 avec gestion de session"""
    try:
        response = await llm_service.generate_response(
            message=request.message,
            session_id=request.session_id
        )
        return ChatResponse(response=response)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize", response_model=ChatResponse)
async def chat(request: ChatRequestTP1) -> ChatResponse:
    """Nouvel endpoint permettant de tester le Sequencing Chain"""
    try:
        response = await llm_service.generate_response_sequencing(
            message=request.message,
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#################### endpoints pour gestion de l'historique des conversations ####################

@router.get("/history/{session_id}")
async def get_history(session_id: str) -> List[Dict[str, str]]:
    """Récupération de l'historique d'une conversation"""
    try:
        return await llm_service.get_conversation_history(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/sessions", response_model=List[str])
async def get_sessions() -> List[str]:
    """Retrieve all session IDs."""
    try:
        return await llm_service.get_all_sessions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{session_id}", response_model=bool)
async def delete_history(session_id: str) -> bool:
    """Delete a specific conversation."""
    try:
        return await llm_service.delete_conversation(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#################### endpoints pour gestion du rag, discussiona avec rag ####################

from fastapi import FastAPI, UploadFile, File
from typing import List
import shutil
from pathlib import Path

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process multiple files for RAG"""
    upload_dir = Path("./uploads")
    upload_dir.mkdir(exist_ok=True)
    
    saved_files = []
    try:
        for file in files:
            file_path = upload_dir / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
            
        # Process the files with RAG service
        await llm_service.rag_service.process_files(saved_files)
        
        return {"message": f"Successfully processed {len(saved_files)} files"}
    finally:
        # Clean up uploaded files
        for file_path in saved_files:
            file_path.unlink(missing_ok=True)

@router.post("/index/documents")
async def index_documents(
    texts: List[str] = Body(...),
    clear_existing: bool = Body(False)
) -> dict:
    """
    Endpoint pour indexer des documents
    
    Args:
        texts: Liste des textes à indexer
        clear_existing: Si True, supprime l'index existant avant d'indexer
    """
    try:
        await llm_service.rag_service.load_and_index_texts(texts, clear_existing)
        return {"message": "Documents indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete/all/documents")
async def clear_documents() -> dict:
    """Endpoint pour supprimer tous les documents indexés"""
    try:
        llm_service.rag_service.close()
        llm_service.rag_service.clear()
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag", response_model=ChatResponse)
async def chat_rag(request: ChatRequest) -> ChatResponse:
    """Endpoint de chat utilisant le RAG"""
    try:
        response = await llm_service.generate_response_rag_service(
            message=request.message,
            session_id=request.session_id,
            use_rag=True
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
