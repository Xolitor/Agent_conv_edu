from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import List, Dict, Optional
from models.conversation import Conversation, Message
from core.config import settings

class MongoService:
    def __init__(self):
        self.client = AsyncIOMotorClient(settings.mongodb_uri)
        self.db = self.client[settings.database_name]
        self.conversations = self.db[settings.collection_name]
        
    async def save_message(self, session_id: str, role: str, content: str) -> bool:
        """Sauvegarde un nouveau message dans une conversation"""
        message = Message(role=role, content=content)
        
        result = await self.conversations.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": message.model_dump()},
                "$set": {"updated_at": datetime.utcnow()},
                "$setOnInsert": {"created_at": datetime.utcnow()}
            },
            
            upsert=True
        )
        
        return result.modified_count > 0 or result.upserted_id is not None
    
    async def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Récupère l'historique d'une conversation"""
        conversation = await self.conversations.find_one({"session_id": session_id})
        if conversation:
            messages = conversation.get("messages", [])
            for message in messages:
                if "timestamp" in message and isinstance(message["timestamp"], datetime):
                    message["timestamp"] = message["timestamp"].isoformat()
            return messages
        return []
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Supprime une conversation"""
        result = await self.conversations.delete_one({"session_id": session_id})
        return result.deleted_count > 0
    
    async def get_all_sessions(self) -> List[str]:
        """Récupère tous les IDs de session triés du plus récent au plus vieux"""
        cursor = self.conversations.find({}, {"session_id": 1}).sort("updated_at", -1)
        sessions = await cursor.to_list(length=None)
        return [session["session_id"] for session in sessions]
    
    async def create_conversation(self, session_id: str, user_id: str) -> bool:
        """Crée une nouvelle conversation"""
        conversation = {
            "session_id": session_id,
            "user_id": user_id,
            "messages": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await self.conversations.insert_one(conversation)
        return result.inserted_id is not None
    
    
    ## Non fonctionnel A FAIRE
    async def find_course(self, query: Dict) -> Optional[Dict]:
        """Trouve un cours dans la base de données"""
        return await self.db.courses.find_one(query)
    
    async def create_teacher(self, teacher_data: dict) -> bool:
        result = await self.db.teachers.insert_one(teacher_data)
        return result.inserted_id is not None
    
    async def get_teacher(self, teacher_id: str) -> Optional[dict] :
        return await self.db.teachers.find_one({"teacher_id": teacher_id})

