from fastapi import APIRouter
from api.endpoints import chat, conversation, exercise, rag, smart_simplified

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
    smart_simplified.router, 
    prefix="/smart", 
    tags=["Smart"]
)