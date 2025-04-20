# models/chat.py
"""
Modèles Pydantic pour la validation des données
Inclut les modèles du TP1 et les nouveaux modèles pour le TP2
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime

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

#################### Modèles pour RAG (Retrieval Augmented Generation) ####################
class DocumentChunk(BaseModel):
    """Représente un morceau de document retourné par une recherche vectorielle"""
    text: str
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None

class RAGResponse(BaseModel):
    """Réponse d'une requête RAG contenant à la fois la réponse et les chunks utilisés"""
    answer: str
    chunks: Optional[List[DocumentChunk]] = None
    metadata: Optional[Dict[str, Any]] = None

class RAGUploadResponse(BaseModel):
    """Résultat du traitement d'un fichier pour RAG"""
    filename: str
    status: str
    chunks: Optional[int] = None
    error: Optional[str] = None

class RAGUploadResult(BaseModel):
    """Résultat global du traitement de fichiers pour RAG"""
    processed_files: List[RAGUploadResponse]

#################### Modèles pour l'analyse d'intention ####################
class IntentResult(BaseModel):
    """Résultat de l'analyse d'intention d'un message utilisateur"""
    intent: str
    parameters: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None
    is_exercise_request: Optional[bool] = None
