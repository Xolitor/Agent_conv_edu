"""
Core MongoDB service that integrates specialized database components.
Acts as the main entry point for database operations.
"""
import os
import logging
import uuid  # Add this import at the top
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_openai import OpenAIEmbeddings
from core.config import settings
import threading
from typing import Any, Dict, List, Optional

from services.mongo_service.conversation_manager import ConversationManager
from services.mongo_service.rag_manager import RAGManager
from services.mongo_service.exercise_manager import ExerciseManager

logging.basicConfig(level=logging.DEBUG)

class MongoDBService:
    """
    Unified MongoDB service handling multiple data domains through specialized managers
    """
    def __init__(self):
        """Initialize the MongoDB service with all specialized managers"""
        self.client = AsyncIOMotorClient(settings.mongodb_uri)
        self.db = self.client[settings.database_name]
        
        # Collection references
        self.conversations = self.db[settings.collection_name]
        self.teachers = self.db[settings.teachers_database]
        self.rag_collection = self.db[settings.rag_database_name]
        self.exercises = self.db[settings.exercises_database]
        
        # Shared resources
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.lock = threading.Lock()
        
        # Initialize specialized managers
        self.conversation_manager = ConversationManager(self.db, self.conversations)
        self.rag_manager = RAGManager(self.db, self.rag_collection, self.embeddings, self.lock)
        self.exercise_manager = ExerciseManager(self.db, self.exercises)
        
        logging.info("MongoDB service initialized")
    
    #############################################
    # Connection and shared database operations #
    #############################################
    
    async def close(self):
        """Close the MongoDB connection"""
        self.client.close()
        logging.debug("MongoDB connection closed.")
        
    def clear(self) -> None:
        """Clears the RAG collection"""
        self.rag_manager.clear_collection()
    
    #######################################
    # Conversation management delegation  #
    #######################################
    
    async def seed_teachers(self, teachers_data):
        """Seed teachers collection with initial data"""
        return await self.teacher_manager.seed_teachers(teachers_data)
    
    async def get_teacher(self, teacher_id: str) -> Optional[Dict]:
        """Get teacher by ID"""
        return await self.teacher_manager.get_teacher(teacher_id)
    
    async def save_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save a message to a conversation"""
        return await self.conversation_manager.save_message(session_id, role, content, metadata)
    
    async def create_conversation(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation with optional session ID.
        
        Args:
            session_id: Optional session ID. If not provided, a new UUID is generated.
            
        Returns:
            The session ID (either provided or generated)
        """
        if not session_id:
            # Generate a unique session ID
            session_id = f"session-{uuid.uuid4()}"
        
        # Create the conversation with the session ID
        await self.conversation_manager.create_conversation(session_id)
        return session_id
    
    async def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history"""
        return await self.conversation_manager.get_conversation_history(session_id)
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation"""
        return await self.conversation_manager.delete_conversation(session_id)
    
    async def get_all_sessions(self) -> List[str]:
        """Get all session IDs"""
        return await self.conversation_manager.get_all_sessions()
    
    ###################################
    # RAG Vector operations delegation #
    ###################################
    
    async def verify_index(self) -> bool:
        """Verify vector search index exists"""
        return await self.rag_manager.verify_index()
    
    def clear_rag_collection(self) -> None:
        """Clear the RAG collection"""
        self.rag_manager.clear_collection()
    
    async def process_file(self, file) -> List[str]:
        """Process uploaded file and extract text chunks"""
        return await self.rag_manager.process_file(file)
    
    async def get_document_count(self) -> int:
        """Get document count in RAG collection"""
        return await self.rag_manager.get_document_count()
    
    async def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        return await self.rag_manager.similarity_search(query, k)
    
    async def add_texts_to_vectorstore(self, texts: List[str], metadata: Optional[dict] = None):
        """Add texts to vector store"""
        return await self.rag_manager.add_texts_to_vectorstore(texts, metadata)
    
    ###################################
    # Exercise operations delegation  #
    ###################################
    
    async def save_exercise(self, exercise_data: dict[str, Any]) -> str:
        """Save an exercise to the database"""
        return await self.exercise_manager.save_exercise(exercise_data)

    async def get_exercise(self, exercise_id: str) -> Optional[Dict]:
        """Retrieve an exercise by ID"""
        return await self.exercise_manager.get_exercise(exercise_id)

    async def get_exercises_by_subject(self, subject: str, limit: int = 10) -> List[Dict]:
        """Get exercises for a subject"""
        return await self.exercise_manager.get_exercises_by_subject(subject, limit)
