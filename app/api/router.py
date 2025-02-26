from fastapi import APIRouter
from api.endpoints import chat, student, teacher, chat_claude

router = APIRouter()

router.include_router(
    chat_claude.router, 
    prefix="/chat", 
    #tags=["chat"]
)

router.include_router(
    student.router, 
    prefix="/student", 
    #tags=["Student"]
)

router.include_router(
    teacher.router, 
    prefix="/teacher", 
    #tags=["Teacher"]
)