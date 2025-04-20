"""
Manages conversations and message history in MongoDB
"""
from asyncio.log import logger
from datetime import datetime
from typing import Any, Dict, List, Optional
from models.conversation import Message

class ConversationManager:
    """Handles conversation storage and retrieval"""
    
    def __init__(self, db, collection):
        """Initialize with database and collection"""
        self.db = db
        self.collection = collection
        logger.info("Conversation Manager initialized")
    
    async def save_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save a new message in a conversation"""
        # Create a message document directly (don't use Pydantic model conversion which might cause issues)
        message_dict = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        }
    
        # Add metadata if provided
        if metadata:
            message_dict["metadata"] = metadata
        
        try:
            # Check if conversation exists first
            conversation = await self.collection.find_one({"session_id": session_id})
            
            if conversation:
                # Update existing conversation - make sure messages array exists
                if "messages" not in conversation:
                    # Add messages array if it doesn't exist
                    await self.collection.update_one(
                        {"_id": conversation["_id"]},
                        {"$set": {"messages": []}}
                    )
                
                # Now push the new message
                result = await self.collection.update_one(
                    {"session_id": session_id},
                    {
                        "$push": {"messages": message_dict},
                        "$set": {"updated_at": datetime.utcnow()}
                    }
                )
            else:
                # Create new conversation with initial message
                result = await self.collection.insert_one({
                    "session_id": session_id,
                    "messages": [message_dict],  # Initialize with array containing first message
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                })
                return result.inserted_id is not None
            
            return result.modified_count > 0 or result.upserted_id is not None
        
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
            return False
    
    async def create_conversation(self, session_id: str) -> bool:
        """
        Create a new conversation with the given session ID.
        
        Args:
            session_id: The session ID for the conversation (should be provided)
            
        Returns:
            True if the conversation was created successfully, False otherwise
        """
        conversation = {
            "session_id": session_id,
            "messages": [],  # Explicitly initialize with empty array
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        try:
            # Check if conversation already exists
            existing = await self.collection.find_one({"session_id": session_id})
            if existing:
                # If exists but has no messages array, add it
                if "messages" not in existing:
                    await self.collection.update_one(
                        {"_id": existing["_id"]},
                        {"$set": {"messages": []}}
                    )
                return True
                
            # Otherwise create new conversation
            result = await self.collection.insert_one(conversation)
            return result.inserted_id is not None
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            return False
    
    async def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history"""
        conversation = await self.collection.find_one({"session_id": session_id})
        formatted_messages = []
        if conversation:
            messages = conversation.get("messages", [])
            
            for msg in messages:
                # Convert datetime to string
                if "timestamp" in msg and isinstance(msg["timestamp"], datetime):
                    msg["timestamp"] = msg["timestamp"].isoformat()
                    
                # Include all fields
                formatted_message = {
                    "role": msg.get("role", ""),
                    "content": msg.get("content", "")
                }
                
                # Only include metadata if it exists
                if "metadata" in msg and msg["metadata"] is not None:
                    formatted_message["metadata"] = msg["metadata"]
                
                formatted_messages.append(formatted_message)
            
        return formatted_messages
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation"""
        result = await self.collection.delete_one({"session_id": session_id})
        return result.deleted_count > 0
    
    async def get_all_sessions(self) -> List[str]:
        """Get all session IDs sorted from newest to oldest"""
        cursor = self.collection.find({}, {"session_id": 1}).sort("updated_at", -1)
        sessions = await cursor.to_list(length=None)
        return [session["session_id"] for session in sessions]

    async def debug_conversation(self, session_id: str) -> Dict[str, Any]:
        """Debug a conversation to check its structure"""
        try:
            conversation = await self.collection.find_one({"session_id": session_id})
            if not conversation:
                return {"error": "Conversation not found"}
            
            # Check if messages array exists
            if "messages" not in conversation:
                return {"error": "Messages array does not exist"}
            
            # Return the structure of the conversation
            return {
                "session_id": session_id,
                "messages_count": len(conversation["messages"]),
                "created_at": conversation.get("created_at"),
                "updated_at": conversation.get("updated_at"),
                "messages_structure": [msg for msg in conversation["messages"]]
            }
        except Exception as e:
            logger.error(f"Error debugging conversation: {str(e)}")
            return {"error": str(e)}