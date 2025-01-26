# models/chat.py
"""
Modèles Pydantic pour la validation des données
Inclut les modèles du TP1 et les nouveaux modèles pour le TP2
"""
from typing import Dict, List, Optional
from pydantic import BaseModel

class ChatResponse(BaseModel):
    """Réponse standard du chatbot"""
    response: str

class ChatRequestWithContext(BaseModel):
    """Requête avec contexte de conversation du TP1"""
    message: str
    context: Optional[List[Dict[str, str]]] = []

class ChatRequest(BaseModel):
    """Requête de base pour une conversation sans contexte"""
    message: str
    session_id: Optional[str] = ""  # Ajouté pour supporter les deux versions
    
class ChatMessage(BaseModel):
    """Structure d'un message individuel dans l'historique"""
    role: str  # "user" ou "assistant"
    content: str
    
# ---- Modèles du TP1 ----
class ChatRequestTP1(BaseModel):
    """Requête de base pour une conversation sans contexte"""
    message: str

# class ChatHistory(BaseModel):
#     """Collection de messages formant une conversation"""
#     messages: List[ChatMessage]
    
# # endpoint du projet
    
class ExerciseRequest(BaseModel):
    """Requête pour un quiz"""
    course_id: str  
    answers: Dict[str, str] 

class ChatRequestWithCourseData(BaseModel):
    """Requête avec données de cours pour le RAG"""
    message: str  
    course_id: str  