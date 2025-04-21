"""
Core LLM service that integrates specialized components
Acts as the main entry point for LLM functionality
"""
import os
from asyncio.log import logger
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from services.llm_service.session_manager import SessionManager
from services.llm_service.response_generator import ResponseGenerator
from services.llm_service.exercise_manager import ExerciseManager
from services.llm_service.router_service import RouterService
from services.mongo_service import MongoDBService

class LLMService:
    """
    Unified LLM service integrating various specialized components
    """
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY n'est pas définie")
        
        # Initialize MongoDB service
        self.mongo_services = MongoDBService()
        
        # Configure base LLM
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            api_key=api_key
        )
        
        # Initialize specialized components in proper order
        # (components that depend on other components come later)
        self.session_manager = SessionManager(self.mongo_services)
        self.response_generator = ResponseGenerator(self.llm, self.mongo_services, self.session_manager)
        self.exercise_manager = ExerciseManager(self.llm, self.mongo_services)
        
        # Initialize router last as it depends on other components
        self.router_service = RouterService(
            llm=self.llm,
            mongo_service=self.mongo_services,
            response_generator=self.response_generator,
            exercise_manager=self.exercise_manager
        )
        
        logger.info("LLM Service initialized")
        
    #################### Smart routing ####################
    
    async def smart_chat(self,
                        message: str,
                        session_id: Optional[str] = None,
                        teacher_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Smart chat that automatically routes to the appropriate functionality
        
        Args:
            message: User's message
            session_id: Optional session ID
            teacher_id: Optional teacher ID
            
        Returns:
            Dict containing response and metadata
        """
        # First, save the user message to conversation history if session_id exists
        if session_id:
            await self.mongo_services.save_message(
                session_id, 
                "user", 
                message,
                metadata={"teacher_id": teacher_id if teacher_id else None}
            )
        else:
            # Create a new session if none provided
            session_id = await self.mongo_services.create_conversation()
            await self.mongo_services.save_message(
                session_id,
                "user",
                message,
                metadata={"teacher_id": teacher_id if teacher_id else None}
            )
        
        # Use router to determine the best handler
        result = await self.router_service.route_query(
            query=message,
            session_id=session_id,
            teacher_id=teacher_id
        )
        
        # Save the assistant's response
        if session_id:
            metadata = {
                "route": result.get("route"),
                "teacher_id": teacher_id if teacher_id else None
            }
            
            if "action" in result:
                metadata["action"] = result["action"]
                
            await self.mongo_services.save_message(
                session_id,
                "assistant",
                result["response"],
                metadata=metadata
            )
        
        # Add session_id to result for convenience
        result["session_id"] = session_id
            
        return result
    
    #################### Session and conversation management ####################
    
    async def create_new_conversation(self) -> str:
        """Creates a new conversation and returns a unique ID"""
        return await self.session_manager.create_new_conversation()
    
    async def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Retrieves conversation history for a session"""
        return await self.session_manager.get_conversation_history(session_id)
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Deletes a conversation by session ID"""
        return await self.session_manager.delete_conversation(session_id)
    
    async def get_all_sessions(self) -> List[str]:
        """Retrieves all session IDs"""
        return await self.session_manager.get_all_sessions()
    
    #################### Response generation ####################
    
    async def generate_response(self,
                              message: str,
                              session_id: Optional[str] = None,
                              teacher_id: Optional[str] = None,
                              use_rag: bool = False) -> str:
        """Unified response generation method"""
        # If no session_id provided, get one from the session manager
        if not session_id:
            session_context = await self.session_manager._ensure_session(None)
            session_id = session_context.session_id
        
        return await self.response_generator.generate_response(
            message=message,
            session_id=session_id,
            teacher_id=teacher_id,
            use_rag=use_rag
        )
    
    #################### Exercise management ####################
    
    async def generate_exercise(self, *args, **kwargs):
        """Generate exercises based on subject and parameters"""
        return await self.exercise_manager.generate_exercise(*args, **kwargs)
    
    async def evaluate_answer(self, *args, **kwargs):
        """Evaluate a student's answer to an exercise"""
        return await self.exercise_manager.evaluate_answer(*args, **kwargs)
