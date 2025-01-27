from fastapi import APIRouter, HTTPException, Body
from models.chat import ChatRequest, ChatResponse, ChatRequestWithCourseData
from services.llm_service import LLMService
from typing import Dict, List
from services.mongo_service import MongoService
from fastapi import Depends

router = APIRouter()
llm_service = LLMService()

@router.post("/{teacher_id}/chat", response_model=ChatResponse)
async def chat_with_teacher(teacher_id: str, request: ChatRequest, mongo_service: MongoService = Depends()):
    teacher = await mongo_service.teachers.find_one({"teacher_id": teacher_id})
    if not teacher:
        raise HTTPException(status_code=404, detail=f"Teacher with ID {teacher_id} not found.")
    
    response = await llm_service.generate_teacher_response(
        teacher_id = teacher_id,
        message=request.message,
        session_id=request.session_id
    )
    return ChatResponse(response=response)