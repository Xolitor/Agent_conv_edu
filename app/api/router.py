from fastapi import APIRouter
from api.endpoints import chat_claude, exercises, smart

router = APIRouter()

router.include_router(
    chat_claude.router, 
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