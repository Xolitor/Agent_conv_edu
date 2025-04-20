"""
Endpoints for conversation history management
"""
from fastapi import APIRouter, HTTPException
from models.conversation import MessageHistoryResponse
from services.llm_service import LLMService
from typing import List
from asyncio.log import logger

router = APIRouter()
llm_service = LLMService()

@router.get("/history/{session_id}", response_model=List[MessageHistoryResponse])
async def get_history(session_id: str):
    """
    Retrieve conversation history for a specific session
    """
    try:
        return await llm_service.get_conversation_history(session_id)
    except Exception as e:
        logger.error(f"Get history error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/sessions", response_model=List[str])
async def get_sessions() -> List[str]:
    """
    Retrieve all session IDs
    """
    try:
        return await llm_service.get_all_sessions()
    except Exception as e:
        logger.error(f"Get sessions error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{session_id}", response_model=bool)
async def delete_conversation(session_id: str) -> bool:
    """
    Delete a specific conversation
    """
    try:
        return await llm_service.delete_conversation(session_id)
    except Exception as e:
        logger.error(f"Delete conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
