"""
Manages teacher data in MongoDB
"""
from asyncio.log import logger
from typing import Dict, List, Optional
from pymongo import UpdateOne
from models.teacher import Teacher

class TeacherManager:
    """Handles teacher data operations"""
    
    def __init__(self, db, collection):
        """Initialize with database and collection"""
        self.db = db
        self.collection = collection
        logger.info("Teacher Manager initialized")
    
    async def seed_teachers(self, teachers_data: List[Teacher]):
        """Populates the teachers collection with initial data if it is empty."""
        operations = []
        for teacher in teachers_data:
            operations.append(
                UpdateOne(
                    {"teacher_id": teacher.teacher_id},  
                    {"$set": teacher.model_dump()},      
                    upsert=True
                )
            )
        if operations:
            await self.collection.bulk_write(operations)
            logger.info(f"Seeded {len(operations)} teachers")
    
    async def get_teacher(self, teacher_id: str) -> Optional[Dict]:
        """Get teacher by ID"""
        return await self.collection.find_one({"teacher_id": teacher_id})
    
    async def create_teacher(self, teacher: Teacher) -> bool:
        """Create a new teacher"""
        result = await self.collection.insert_one(teacher.model_dump())
        return result.inserted_id is not None
    
    async def update_teacher(self, teacher_id: str, update_data: Dict) -> bool:
        """Update a teacher's data"""
        result = await self.collection.update_one(
            {"teacher_id": teacher_id},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    async def delete_teacher(self, teacher_id: str) -> bool:
        """Delete a teacher"""
        result = await self.collection.delete_one({"teacher_id": teacher_id})
        return result.deleted_count > 0
    
    async def get_all_teachers(self) -> List[Dict]:
        """Get all teachers"""
        cursor = self.collection.find({})
        return await cursor.to_list(length=None)
