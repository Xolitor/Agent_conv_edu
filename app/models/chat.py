# models/chat.py
"""
Modèles Pydantic pour la validation des données
Inclut les modèles du TP1 et les nouveaux modèles pour le TP2
"""
from typing import Dict, List, Optional
from pydantic import BaseModel

#################### Réponse standard du chatbot ####################


class ChatResponse(BaseModel):
    """Réponse standard du chatbot"""
    response: str

#################### Requête de base pour une conversation ####################

class ChatRequest(BaseModel):
    """Requête de base pour une conversation sans contexte"""
    message: str
    session_id: Optional[str] = ""  # Ajouté pour supporter la gestion de session
    teacher_id: Optional[str] = None  # ID de l'enseignant
    use_rag: Optional[bool] = False  # Utiliser la RAG ou pas
    
class ChatMessage(BaseModel):
    """Structure d'un message individuel dans l'historique"""
    role: str  # "user" ou "assistant"
    content: str
