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
        
        # Initialize specialized components
        self.session_manager = SessionManager(self.mongo_services)
        self.response_generator = ResponseGenerator(self.llm, self.mongo_services, self.session_manager)
        self.exercise_manager = ExerciseManager(self.llm, self.mongo_services)
        
        # Keep only the chains needed for sequencing demo
        self.main_prompt = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant utile et concis en expliquant avec des exemples de jeux vidéos."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        self.bullet_points_chain = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant qui ajoute des jetons à la fin du texte."),
            ("human", "Résumé sous forme de points clés : {text}")
        ]) | self.llm

        self.one_liner_chain = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant qui ajoute un résumé en une phrase à la fin du texte."),
            ("human", "Résumé en une phrase : {text}")
        ]) | self.llm
        
        logger.info("LLM Service initialized")
    
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
    
    async def generate_response_sequencing(self, message: str, session_id: str = "") -> str:
        """Generate a comprehensive response with multiple processing steps"""
        main_chain = self.main_prompt | self.llm
        main_response = (await main_chain.ainvoke({
            "history":  [],
            "question": message
        })).content

        bullet_points_response = (await self.bullet_points_chain.ainvoke({
            "text": main_response
        })).content

        one_liner_response = (await self.one_liner_chain.ainvoke({
            "text": bullet_points_response
        })).content

        return one_liner_response
    
    #################### Exercise management ####################
    
    async def generate_exercise(self, *args, **kwargs):
        """Generate exercises based on subject and parameters"""
        return await self.exercise_manager.generate_exercise(*args, **kwargs)
    
    async def evaluate_answer(self, *args, **kwargs):
        """Evaluate a student's answer to an exercise"""
        return await self.exercise_manager.evaluate_answer(*args, **kwargs)
