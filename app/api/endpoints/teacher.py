from fastapi import APIRouter, HTTPException, Body
from models.chat import ChatRequest, ChatResponse, ChatRequestWithCourseData
from services.llm_service import LLMService
from typing import Dict, List
from services.mongo_service import MongoService
from fastapi import Depends

router = APIRouter()
llm_service = LLMService()

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

@router.post("/{teacher_id}/chat", response_model=ChatResponse)
async def chat_with_teacher(teacher_id: str, request: ChatRequest):
        try :
            response = await llm_service.generate_teacher_response(
            teacher_id = teacher_id,
            message=request.message,
            session_id=request.session_id
            )
            return ChatResponse(response=response)
        except Exception as e:
                print(e)
                raise HTTPException(status_code=500, detail=str(e))
        