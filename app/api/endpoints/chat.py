# api/chat.py
"""
Routes FastAPI pour le chatbot
"""
from datetime import datetime
from fastapi import APIRouter, HTTPException, Body, UploadFile, File
from models.conversation import MessageHistoryResponse
from models.chat import ChatRequest, ChatResponse
from services.llm_service import LLMService
from typing import Dict, List, Optional
from pathlib import Path
router = APIRouter()
import hashlib
from asyncio.log import logger
from bson.json_util import dumps, loads

llm_service = LLMService()
mongo_service = llm_service.mongo_services

#################### endpoint pour le chatbot de base ####################

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint supporting regular, and RAG responses
    """
    try:
        # Ensure there's a valid session ID (either provided or generated)
        session_id = request.session_id
        if not session_id:
            # Generate a new session ID and create the conversation
            session_id = await mongo_service.create_conversation()
            # Update the request object with the new session ID
            request.session_id = session_id
        
        # First save the user message to conversation history
        await mongo_service.save_message(
            session_id, 
            "user", 
            request.message,
            metadata={"teacher_id": request.teacher_id if request.teacher_id else None}
        )
        
        # Generate response
        response = await llm_service.generate_response(
            message=request.message,
            session_id=session_id,
            use_rag=request.use_rag if hasattr(request, 'use_rag') else False
        )
        
        # Save the assistant's response
        metadata = {}
        if hasattr(request, 'use_rag') and request.use_rag:
            metadata["use_rag"] = True
            
        await mongo_service.save_message(
            session_id,
            "assistant",
            response,
            metadata=metadata if metadata else None
        )
        
        # Create a response object that includes the session_id
        chat_response = ChatResponse(response=response)
        # Return the session_id in the response headers or as part of response
        return chat_response
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))