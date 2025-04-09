from fastapi import APIRouter
from api.endpoints import exercises, smart, teacher, chat

router = APIRouter()

router.include_router(
    chat.router, 
    prefix="/chat", 
    #tags=["chat"]
)

router.include_router(
    exercises.router, 
    prefix="/exercises", 
    #tags=["Exercises"]
)

router.include_router(
    smart.router, 
    prefix="/smart", 
    #tags=["SmartChat"]
)

router.include_router(
    teacher.router, 
    prefix="/teacher", 
    #tags=["Teacher"]
)