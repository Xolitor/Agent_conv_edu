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
    
    async def save_hint(self, hint_data: Dict[str, Any]) -> str:
        """Save a hint to the database"""
        try:
            result = await self.db.exercise_hints.insert_one(hint_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to save hint: {str(e)}")
            raise
    
    async def get_hints_by_exercise(self, exercise_id: str) -> List[Dict]:
        """Get all hints associated with an exercise"""
        try:
            cursor = self.db.exercise_hints.find({"exercise_id": exercise_id}).sort("created_at", -1)
            return await cursor.to_list(length=100)
        except Exception as e:
            logger.error(f"Failed to retrieve hints: {str(e)}")
            return []
    
    async def get_hint(self, hint_id: str) -> Optional[Dict]:
        """Retrieve a specific hint by ID"""
        try:
            result = await self.db.exercise_hints.find_one({"_id": ObjectId(hint_id)})
            return result
        except Exception as e:
            logger.error(f"Failed to retrieve hint: {str(e)}")
            return None
    
    async def get_solutions(self, exercise_id: str) -> Optional[Dict]:
        """Get solutions for a specific exercise"""
        try:
            exercise = await self.get_exercise(exercise_id)
            if exercise and "solutions" in exercise:
                return exercise["solutions"]
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve solutions: {str(e)}")
            return None
    
    async def save_evaluation(self, evaluation_data: Dict[str, Any]) -> str:
        """Save an exercise evaluation to the database"""
        try:
            result = await self.db.exercise_evaluations.insert_one(evaluation_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to save evaluation: {str(e)}")
            raise
    
    async def get_evaluation(self, evaluation_id: str) -> Optional[Dict]:
        """Retrieve a specific evaluation by ID"""
        try:
            result = await self.db.exercise_evaluations.find_one({"_id": ObjectId(evaluation_id)})
            return result
        except Exception as e:
            logger.error(f"Failed to retrieve evaluation: {str(e)}")
            return None
    
    async def get_evaluations_by_exercise(self, exercise_id: str) -> List[Dict]:
        """Get all evaluations for a specific exercise"""
        try:
            cursor = self.db.exercise_evaluations.find({"exercise_id": exercise_id}).sort("created_at", -1)
            return await cursor.to_list(length=100)
        except Exception as e:
            logger.error(f"Failed to retrieve evaluations: {str(e)}")
            return []
    
    async def get_evaluations_by_session(self, session_id: str) -> List[Dict]:
        """Get all evaluations for a specific session"""
        try:
            cursor = self.db.exercise_evaluations.find({"session_id": session_id}).sort("created_at", -1)
            return await cursor.to_list(length=100)
        except Exception as e:
            logger.error(f"Failed to retrieve evaluations: {str(e)}")
            return []
