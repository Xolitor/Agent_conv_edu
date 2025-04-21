"""
Endpoints for teacher-specific functionality
"""
from fastapi import APIRouter, HTTPException, Body, Depends
from models.chat import ChatRequest, ChatResponse
from models.teacher import Teacher, TeacherResponse, TeacherListResponse
from services.llm_service import LLMService
from typing import Dict, List, Optional
from asyncio.log import logger

router = APIRouter()
llm_service = LLMService()
mongo_service = llm_service.mongo_services

# Helper function to get or create a session
async def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get an existing session or create a new one if none exists"""
    if not session_id:
        session_id = await mongo_service.create_conversation()
    return session_id

@router.get("/list", response_model=TeacherListResponse)
async def list_teachers():
    """Get all available teachers"""
    try:
        teachers = await mongo_service.teacher_manager.get_all_teachers()
        # Convert to Teacher models
        teacher_models = [Teacher(**teacher) for teacher in teachers]
        return TeacherListResponse(teachers=teacher_models, count=len(teacher_models))
    except Exception as e:
        logger.error(f"List teachers error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{teacher_id}", response_model=TeacherResponse)
async def get_teacher(teacher_id: str):
    """Get a specific teacher by ID"""
    try:
        teacher_data = await mongo_service.get_teacher(teacher_id)
        if not teacher_data:
            raise HTTPException(status_code=404, detail=f"Teacher {teacher_id} not found")
        return TeacherResponse(teacher=Teacher(**teacher_data))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get teacher error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{teacher_id}/chat", response_model=ChatResponse)
async def chat_with_teacher(
    teacher_id: str, 
    request: ChatRequest
):
    """
    Chat with a specific teacher personality
    """
    try:
        # Ensure we have a valid session
        session_id = await get_or_create_session(request.session_id)
        
        # First save the user message to conversation history
        await mongo_service.save_message(
            session_id, 
            "user", 
            request.message,
            metadata={"teacher_id": teacher_id}
        )
        
        # Generate response
        response = await llm_service.generate_response(
            teacher_id=teacher_id,
            message=request.message,
            session_id=session_id
        )
        
        # Save the assistant's response
        await mongo_service.save_message(
            session_id,
            "assistant",
            response,
            metadata={"teacher_id": teacher_id}
        )
        
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Teacher chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
