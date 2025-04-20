"""
Manages exercise data in MongoDB
"""
from asyncio.log import logger
from typing import Any, Dict, List, Optional
from bson import ObjectId

class ExerciseManager:
    """Handles exercise data operations"""
    
    def __init__(self, db, collection):
        """Initialize with database and collection"""
        self.db = db
        self.collection = collection
        logger.info("Exercise Manager initialized")
    
    async def save_exercise(self, exercise_data: Dict[str, Any]) -> str:
        """Save an exercise to the database"""
        result = await self.collection.insert_one(exercise_data)
        return str(result.inserted_id)
    
    async def get_exercise(self, exercise_id: str) -> Optional[Dict]:
        """Retrieve an exercise by ID"""
        try:
            result = await self.db.exercises.find_one({"_id": ObjectId(exercise_id)})
            return result
        except Exception as e:
            logger.error(f"Failed to retrieve exercise: {str(e)}")
            return None
    
    async def get_exercises_by_subject(self, subject: str, limit: int = 10) -> List[Dict]:
        """Get exercises for a specific subject"""
        cursor = self.db.exercises.find({"subject": subject}).sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)
    
    async def update_exercise(self, exercise_id: str, update_data: Dict) -> bool:
        """Update an exercise"""
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(exercise_id)},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update exercise: {str(e)}")
            return False
    
    async def delete_exercise(self, exercise_id: str) -> bool:
        """Delete an exercise"""
        try:
            result = await self.collection.delete_one({"_id": ObjectId(exercise_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete exercise: {str(e)}")
            return False
