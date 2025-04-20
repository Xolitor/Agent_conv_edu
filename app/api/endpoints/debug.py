"""
Debug endpoints for troubleshooting and development
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from services.llm_service import LLMService
from asyncio.log import logger

router = APIRouter()
llm_service = LLMService()
mongo_service = llm_service.mongo_services

@router.get("/conversation/{session_id}")
async def debug_conversation(session_id: str) -> Dict[str, Any]:
    """
    Debug endpoint to diagnose conversation structure issues
    """
    try:
        result = await mongo_service.conversation_manager.debug_conversation(session_id)
        return result
    except Exception as e:
        logger.error(f"Debug conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/repair-conversations")
async def repair_all_conversations() -> Dict[str, Any]:
    """
    Attempt to repair all conversations by ensuring messages arrays exist
    """
    try:
        # Get all conversations
        cursor = mongo_service.conversation_manager.collection.find({})
        conversations = await cursor.to_list(length=None)
        
        repaired = 0
        already_valid = 0
        
        # Check each conversation
        for conversation in conversations:
            if "messages" not in conversation:
                # Add messages array if missing
                await mongo_service.conversation_manager.collection.update_one(
                    {"_id": conversation["_id"]},
                    {"$set": {"messages": []}}
                )
                repaired += 1
            else:
                already_valid += 1
                
        return {
            "status": "success",
            "total_processed": len(conversations),
            "repaired": repaired,
            "already_valid": already_valid
        }
    except Exception as e:
        logger.error(f"Repair conversations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-message/{session_id}")
async def create_test_message(session_id: str) -> Dict[str, Any]:
    """
    Create a test message to verify message saving
    """
    try:
        # Create a test user message
        user_saved = await mongo_service.conversation_manager.save_message(
            session_id=session_id,
            role="user",
            content="This is a test message",
            metadata={"test": True}
        )
        
        # Create a test assistant message
        assistant_saved = await mongo_service.conversation_manager.save_message(
            session_id=session_id,
            role="assistant",
            content="This is a test response",
            metadata={"test": True}
        )
        
        # Check conversation structure
        debug_info = await mongo_service.conversation_manager.debug_conversation(session_id)
        
        return {
            "user_message_saved": user_saved,
            "assistant_message_saved": assistant_saved,
            "conversation_debug": debug_info
        }
    except Exception as e:
        logger.error(f"Test message error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
