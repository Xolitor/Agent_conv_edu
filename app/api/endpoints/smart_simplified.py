"""
Endpoints for smart routing functionality
"""
from fastapi import APIRouter, HTTPException, Body
from models.chat import ChatRequest, ChatResponse
from services.llm_service import LLMService
from typing import Dict, Any, Optional
from asyncio.log import logger
import traceback

router = APIRouter()
llm_service = LLMService()

@router.post("/chat", response_model=ChatResponse)
async def smart_chat(request: ChatRequest) -> ChatResponse:
    """
    Smart chat endpoint that automatically routes to the appropriate functionality
    """
    try:
        # Ensure there's a valid session ID (either provided or generated)
        session_id = request.session_id
        if not session_id:
            # Generate a new session ID
            session_id = await llm_service.mongo_services.create_conversation()
            # Update the request object with the new session ID
            request.session_id = session_id
        
        # Use the smart router
        result = await llm_service.smart_chat(
            message=request.message,
            session_id=session_id
        )
        
        # Return the response
        return ChatResponse(response=result["response"])
        
    except Exception as e:
        # Get detailed stack trace
        stack_trace = traceback.format_exc()
        error_message = f"Smart chat error: {str(e)}\n\nStack trace: {stack_trace}"
        logger.error(error_message)
        
        # Return a more helpful error message
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your request: {str(e)}. Please check server logs for details."
        )

@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_query(
    request: ChatRequest,
    return_full_analysis: bool = Body(False)
) -> Dict[str, Any]:
    """
    Analyze a query without executing it, to see how it would be routed
    """
    try:
        # Use the router service directly
        result = await llm_service.router_service.route_query(
            query=request.message,
            session_id=request.session_id
        )
        
        if return_full_analysis:
            return result
        else:
            return {
                "route": result["route"],
                "reasoning": result.get("reasoning", ""),
                "action": result.get("action", "")
            }
            
    except Exception as e:
        logger.error(f"Query analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))