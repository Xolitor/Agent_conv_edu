# services/memory.py
"""
Gestion de la mémoire des conversations
"""
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from typing import List

class InMemoryHistory(BaseChatMessageHistory):
    """
    Implémentation simple du stockage en mémoire de l'historique des conversations.
    Pour un environnement de production, considérer une solution persistante comme Redis.
    """
    def __init__(self, *args) -> None:
        self.messages: List[BaseMessage] = args if args else []
    
    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Ajoute une série de messages à l'historique"""
        self.messages.extend(messages)
    
    def clear(self) -> None:
        """Réinitialise l'historique de la conversation"""
        self.messages = []

    async def aget_messages(self) -> List[BaseMessage]:
        """Récupère l'historique des messages de façon asynchrone"""
        return self.messages.copy()