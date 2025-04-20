from fastapi import APIRouter
from api.endpoints import chat, conversation, teacher, exercise, rag, debug, smart

router = APIRouter()

# Chat functionality
router.include_router(
    chat.router, 
    prefix="/chat", 
    tags=["Chat"]
)

# Conversation management
router.include_router(
    conversation.router, 
    prefix="/conversation", 
    tags=["Conversation"]
)

# Teacher-specific functionality
router.include_router(
    teacher.router, 
    prefix="/teacher", 
    tags=["Teacher"]
)

# Exercise functionality
router.include_router(
    exercise.router, 
    prefix="/exercise", 
    tags=["Exercise"]
)

# RAG (Retrieval Augmented Generation)
router.include_router(
    rag.router, 
    prefix="/rag", 
    tags=["RAG"]
)

# Smart chat (unified experience)
router.include_router(
    smart.router, 
    prefix="/smart", 
    tags=["Smart"]
)

# Debug endpoints
router.include_router(
    debug.router, 
    prefix="/debug", 
    tags=["Debug"]
)