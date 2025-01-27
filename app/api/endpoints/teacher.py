from fastapi import APIRouter, HTTPException, Body
from models.chat import ChatRequest, ChatResponse, ChatRequestWithCourseData
from services.llm_service import LLMService
from typing import Dict, List

router = APIRouter()
llm_service = LLMService()

@router.post("/{teacher_id}/chat", response_model=ChatResponse)
async def chat_with_teacher(teacher_id: str, request: ChatRequest):
        response = await llm_service.generate_teacher_response(
            teacher_id = teacher_id,
            user_message=request.message,
            session_id=request.session_id
        )
        return ChatResponse(response=response)