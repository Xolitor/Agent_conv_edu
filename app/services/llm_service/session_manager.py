"""
Session management component for LLM service
Handles conversation history, sessions, and persistence
"""
import uuid
from asyncio.log import logger
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from services.memory import InMemoryHistory
from langchain_core.chat_history import BaseChatMessageHistory

@dataclass
class SessionContext:
    """Represents a chat session context"""
    session_id: str
    history: List[Dict[str, str]]
    metadata: Dict[str, Any] = None

class SessionManager:
    """
    Manages conversation sessions, history, and persistence
    """
    def __init__(self, mongo_service):
        """Initialize with MongoDB service for persistence"""
        self.mongo_service = mongo_service
        self.conversation_store = {}
        logger.info("Session Manager initialized")
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retrieves or creates the in-memory history for a session"""
        if session_id not in self.conversation_store:
            logger.debug(f"Creating new history for session {session_id}")
            self.conversation_store[session_id] = InMemoryHistory()
        return self.conversation_store[session_id]
    
    def cleanup_inactive_sessions(self):
        """Cleans up inactive sessions"""
        current_time = datetime.now()
        for session_id, history in list(self.conversation_store.items()):
            if hasattr(history, 'is_active') and not history.is_active():
                del self.conversation_store[session_id]
                logger.debug(f"Cleaned up inactive session {session_id}")
    
    async def create_new_conversation(self) -> str:
        """Creates a new conversation and generates a unique ID"""
        session_id = f"session_{uuid.uuid4()}"
        # The MongoDB service now handles the case when session_id is None
        return await self.mongo_service.create_conversation(session_id)
    
    async def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Retrieves conversation history from MongoDB and initializes memory"""
        history = await self.mongo_service.get_conversation_history(session_id)
        self.conversation_store[session_id] = InMemoryHistory()
        self.conversation_store[session_id].add_messages(history)
        return history
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Deletes a conversation by session ID"""
        if session_id in self.conversation_store:
            del self.conversation_store[session_id]
        return await self.mongo_service.delete_conversation(session_id)
    
    async def get_all_sessions(self) -> List[str]:
        """Retrieves all session IDs"""
        return await self.mongo_service.get_all_sessions()
    
    async def _ensure_session(self, session_id: Optional[str] = None) -> SessionContext:
        """Creates or retrieves a session context"""
        if not session_id:
            # Let the MongoDB service generate the session ID
            session_id = await self.mongo_service.create_conversation()
            self.conversation_store[session_id] = InMemoryHistory()
            return SessionContext(session_id=session_id, history=[])
            
        if session_id not in self.conversation_store:
            history = await self.mongo_service.get_conversation_history(session_id)
            self.conversation_store[session_id] = InMemoryHistory()
            self.conversation_store[session_id].add_messages(history)
            return SessionContext(session_id=session_id, history=history)
            
        return SessionContext(
            session_id=session_id,
            history=self.conversation_store[session_id].messages
        )
