# services/llm_service.py
"""
Service principal gérant les interactions avec le modèle de langage
Compatible avec les fonctionnalités du TP1 et du TP2
"""
import asyncio
from asyncio.log import logger
import logging
import uuid
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import LLM
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from services.memory import InMemoryHistory
import os
from typing import Any, List, Dict, Optional
from services.mongo_service import MongoService
from datetime import datetime
from services.rag_mongo_services import RAGServiceMongo
from services.rag_service import RAGService
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union

@dataclass
class SessionContext:
    """Represents a chat session context"""
    session_id: str
    history: List[Dict[str, str]]
    metadata: Dict[str, Any] = None

class LLMService:
    """
    Service LLM unifié supportant à la fois les fonctionnalités du TP1 et du TP2
    """
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.mongo_service = MongoService()
        self.rag_mongo_service = RAGServiceMongo()
        if not api_key:
            raise ValueError("OPENAI_API_KEY n'est pas définie")
        
        # Configuration commune
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            api_key=api_key
        )
        
        print("Initialisation du service LLM")
        self.conversation_store = {}

        #################### Configuration de plusieurs chaînes de traitement ####################
        
        self.main_prompt  = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant utile et concis en expliquant avec des exemples de jeux vidéos."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])       
        
        self.bullet_points_chain = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant qui ajoute des jetons à la fin du texte."),
            ("human", "Résumé sous forme de points clés : {text}")
        ]) | self.llm

        self.one_liner_chain = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant qui ajoute un résumé en une phrase à la fin du texte."),
            ("human", "Résumé en une phrase : {text}")
        ]) | self.llm
        

        #################### Configuration du service RAG ####################
        
        
        self.rag_service = RAGService()
    
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "{context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ]) | self.llm

        self.rag_chain_with_history = RunnableWithMessageHistory(
            self.rag_prompt,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )
        
        #################### Chaine qui gère l'historique ####################
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant utile et concis qui retourne ses réponses en format Markdown. Répondez toujours avec un formatage clair, en utilisant des titres, des listes."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        #################### Chaine qui gère l'historique ####################
        
        # self.prompt = ChatPromptTemplate.from_messages([
        #     ("system", "Vous êtes un assistant utile et concis qui retourne ses réponses en format Markdown. Répondez toujours avec un formatage clair, en utilisant des titres, des listes."),
        #     MessagesPlaceholder(variable_name="history"),
        #     ("human", "{question}")
        # ]) 
        
        self.chain = self.prompt | self.llm
        
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )
        
        #################### Chaine qui gère l'historique Agent (Teacher) ####################

        self.teacher_prompt = ChatPromptTemplate.from_messages([
            ("system", "{teacher_style}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ]) | self.llm

        self.teacher_chain_with_history = RunnableWithMessageHistory(
            self.teacher_prompt,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    
    #################### Méthodes pour gérer l'historique, les sessions et les conversations ####################
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Récupère ou crée l'historique pour une session donnée"""
        if session_id not in self.conversation_store:
            print(f"Création de l'historique pour la session {session_id}")
            self.conversation_store[session_id] = InMemoryHistory()
        return self.conversation_store[session_id]
    
    def cleanup_inactive_sessions(self):
        """Nettoie les sessions inactives"""
        current_time = datetime.now()
        for session_id, history in list(self.conversation_store.items()):
            if not history.is_active():
                del self.conversation_store[session_id]
        
    async def create_new_conversation(self, user_id: str) -> str:
        """Crée une nouvelle conversation et génère un ID unique."""
        session_id = f"session_{uuid.uuid4()}"
        
        await self.mongo_service.create_conversation(session_id, user_id)
        return session_id
        
    
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Récupère l'historique depuis MongoDB et initialise la mémoire"""
        # Attendre les données de MongoDB
        history = await self.mongo_service.get_conversation_history(session_id)
        # Initialiser EnhancedMemoryHistory avec les données récupérées
        self.conversation_store[session_id] = InMemoryHistory()
        self.conversation_store[session_id].add_messages(history)

        return history
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation by session ID."""
        return await self.mongo_service.delete_conversation(session_id)

    async def get_all_sessions(self) -> List[str]:
        """Retrieve all session IDs."""
        return await self.mongo_service.get_all_sessions()
    
    
    
    #################### Méthodes pour  générer des réponses en fonction du type de endpoint utilisé ####################

    
    async def _ensure_session(self, session_id: Optional[str] = None) -> SessionContext:
        """Creates or retrieves a session context"""
        if not session_id:
            session_id = f"session_{uuid.uuid4()}"
            await self.mongo_service.create_conversation(session_id, "anonymous")
            self.conversation_store[session_id] = InMemoryHistory()
            return SessionContext(session_id=session_id, history=[])
            
        if session_id not in self.conversation_store:
            history = await self.mongo_service.get_conversation_history(session_id)
            self.conversation_store[session_id] = InMemoryHistory()
            self.conversation_store[session_id].add_messages(history)
            return SessionContext(session_id=session_id, history=history)
            
        return SessionContext(
            session_id=session_id,
            history=self.conversation_store[session_id].messages
        )

    async def _save_interaction(self, 
                              session: SessionContext, 
                              user_message: str, 
                              assistant_response: str):
        """Save interaction to database and memory"""
        await self.mongo_service.save_message(session.session_id, "user", user_message)
        await self.mongo_service.save_message(session.session_id, "assistant", assistant_response)
        
        if session.session_id not in self.conversation_store:
            self.conversation_store[session.session_id] = InMemoryHistory()
        
        self.conversation_store[session.session_id].add_user_message(user_message)
        self.conversation_store[session.session_id].add_ai_message(assistant_response)

    async def generate_response(self,
                              message: str,
                              session_id: Optional[str] = None,
                              context: Optional[List[Dict[str, str]]] = None,
                              teacher_id: Optional[str] = None,
                              use_rag: bool = False) -> str:
        """Unified response generation method"""
        try:
            session = await self._ensure_session(session_id)
            
            # Prepare the base messages
            messages = []
            
            # Add appropriate system message
            if teacher_id:
                teacher_data = await self.mongo_service.get_teacher(teacher_id)
                if not teacher_data:
                    raise ValueError(f"Teacher {teacher_id} not found")
                messages.append(SystemMessage(content=teacher_data["prompt_instructions"]))
            elif use_rag:
                # Get relevant documents for RAG
                relevant_docs = await self.rag_mongo_service.similarity_search(message)
                if relevant_docs:
                    rag_context = "\n\n".join(doc["text"] for doc in relevant_docs)
                    messages.append(SystemMessage(content=self.rag_system_prompt + rag_context))
            else:
                messages.append(SystemMessage(content=self.default_system_prompt))

            # Add conversation history if exists
            for msg in session.history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

            # Add the current message
            messages.append(HumanMessage(content=message))

            # Generate response
            response = await self.llm.agenerate([messages])
            response_text = response.generations[0][0].text

            # Save interaction
            await self._save_interaction(session, message, response_text)

            return response_text

        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    # Properties for different system prompts
    @property
    def default_system_prompt(self) -> str:
        return """Vous êtes un assistant utile et concis qui retourne ses réponses en format Markdown. 
        Répondez toujours avec un formatage clair, en utilisant des titres, des listes."""

    @property
    def rag_system_prompt(self) -> str:
        return """Tu es un assistant pédagogue qui répond uniquement en se basant sur le contexte fourni. 
        Utilise un format Markdown clair et structuré. Contexte fourni:\n\n"""

    # Remove or mark as deprecated the old methods
    async def generate_teacher_response(self, *args, **kwargs):
        """Deprecated: Use generate_response with teacher_id parameter instead"""
        raise DeprecationWarning("Use generate_response with teacher_id parameter")

    async def generate_response_rag_service(self, *args, **kwargs):
        """Deprecated: Use generate_response with use_rag=True instead"""
        raise DeprecationWarning("Use generate_response with use_rag=True")

    async def generate_answer_rag_v2(self, *args, **kwargs):
        """Deprecated: Use generate_response with use_rag=True instead"""
        raise DeprecationWarning("Use generate_response with use_rag=True")
    
    # Cette méthode a été crée pour s'entrainer à utiliser le Sequencing Chain  
    async def generate_response_sequencing(self, message: str, session_id: str = "") -> str:
        """
        Generate a comprehensive response with multiple processing steps

        Args:
            message (str): User's input message
            session_id (str, optional): Session identifier for conversation context

        Returns:
            str: Processed response
        """
        main_chain = self.main_prompt | self.llm
        main_response = (await main_chain.ainvoke({
            "history":  [],
            "question": message
        })).content

        bullet_points_response = (await self.bullet_points_chain.ainvoke({
            "text": main_response
        })).content

        one_liner_response = (await self.one_liner_chain.ainvoke({
            "text": bullet_points_response
        })).content

        return one_liner_response
            