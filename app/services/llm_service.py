# services/llm_service.py
"""
Service principal gérant les interactions avec le modèle de langage
Compatible avec les fonctionnalités du TP1 et du TP2
"""
import logging
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from services.memory import InMemoryHistory
import os
from typing import List, Dict, Optional
from services.mongo_service import MongoService
from services.rag_mongo_services import RAGServiceMongo
from services.rag_service import RAGService

class LLMService:
    """
    Service LLM unifié supportant à la fois les fonctionnalités du TP1 et du TP2
    """
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.mongo_service = MongoService()
        if not api_key:
            raise ValueError("OPENAI_API_KEY n'est pas définie")
        
        # Configuration commune
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            api_key=api_key
        )
        
        # Configuration pour le TP2
        self.conversation_store = {}
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant utile et concis."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        self.bullet_points_chain = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant qui transforme un texte en points clés concis."),
            ("human", "Résumé sous forme de points clés : {text}")
        ]) | self.llm

        self.one_liner_chain = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant qui fournit un résumé d'une ligne."),
            ("human", "Résumé en une phrase : {text}")
        ]) | self.llm
        
        self.chain = self.prompt | self.llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant utile et concis."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        # Configuration du gestionnaire d'historique
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )

        # Ajout du service RAG
        self.rag_service = RAGService()
        
        # Mise à jour du prompt pour inclure le contexte RAG
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant utile et concis. "
                      "Utilisez le contexte fourni pour répondre aux questions."),
            ("system", "Contexte : {context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Récupère ou crée l'historique pour une session donnée"""
        if session_id not in self.conversation_store:
            self.conversation_store[session_id] = InMemoryHistory()
        return self.conversation_store[session_id]

    async def generate_response(self, 
                              message: str, 
                              context: Optional[List[Dict[str, str]]] = None,
                              session_id: Optional[str] = None) -> str:
        if session_id:
            response = await self.chain_with_history.ainvoke(
                {
                    "question": message,
                    
                },
                config={"configurable": {"session_id": session_id}}
            )
            return response.content
        response_text = response.content
        await self.mongo_service.save_message(session_id, "user", message)
        await self.mongo_service.save_message(session_id, "assistant", response_text)
        
    async def generate_response_with_rag(
        self, 
        message: str, 
        context: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        use_rag: bool = False
    ) -> str:
        """
        Generate a response, optionally using Retrieval-Augmented Generation (RAG).
        
        Args:
            message: The user message.
            context: Optional conversation history.
            session_id: The session ID for tracking history.
            use_rag: Whether to use RAG for response generation.
        
        Returns:
            The generated response as a string.
        """
        rag_context = ""
        if use_rag:
            try:
                # Use the new RAGServiceMongo for similarity search
                relevant_docs = await self.rag_service.similarity_search(message)
                rag_context = "\n\n".join(relevant_docs)
            except Exception as e:
                logging.error(f"RAG retrieval failed: {e}")
        
        if session_id:
            # Use chain with history for conversational context
            response = await self.chain_with_history.ainvoke(
                {
                    "question": message,
                    "context": rag_context
                },
                config={"configurable": {"session_id": session_id}}
            )
            return response.content
        else:
            # Direct response generation without session tracking
            messages = [
                SystemMessage(content="Vous êtes un assistant utile et concis.")
            ]
            
            if rag_context:
                messages.append(SystemMessage(content=f"Contexte : {rag_context}"))
            
            if context:
                for msg in context:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
            
            messages.append(HumanMessage(content=message))
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text
        
            
    async def generate_response_rag_mongo_v2(
        self, 
        message: str, 
        context: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        use_rag: bool = False
    ) -> str:
        """
        Generate a response, optionally using Retrieval-Augmented Generation (RAG).
        
        Args:
            message: The user message.
            context: Optional conversation history.
            session_id: The session ID for tracking history.
            use_rag: Whether to use RAG for response generation.
        
        Returns:
            The generated response as a string.
        """
        rag_context = ""
        if use_rag:
            try:
                # Use the new RAGServiceMongo for similarity search
                relevant_docs = await self.rag_service.similarity_search(message)
                rag_context = "\n\n".join(relevant_docs)
            except Exception as e:
                logging.error(f"RAG retrieval failed: {e}")
        
        if session_id:
            # Use chain with history for conversational context
            response = await self.chain_with_history.ainvoke(
                {
                    "question": message,
                    "context": rag_context
                },
                config={"configurable": {"session_id": session_id}}
            )
            return response.content
        else:
            # Direct response generation without session tracking
            messages = [
                SystemMessage(content="Vous êtes un assistant utile et concis.")
            ]
            
            if rag_context:
                messages.append(SystemMessage(content=f"Contexte : {rag_context}"))
            
            if context:
                for msg in context:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
            
            messages.append(HumanMessage(content=message))
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text

    
    
    async def generate_exercise_from_context(self, 
                                            message: str, 
                                            session_id: Optional[str] = None) -> Dict:
        """Génerer un exercice à partir d'un contexte de conversation"""
        try:
            response = await self.chain_with_history.ainvoke(
                {"question": message},
                config={"configurable": {"session_id": session_id}}
            )
            
            response_text = response.content
            await self.mongo_service.save_message(session_id, "user", message)
            await self.mongo_service.save_message(session_id, "assistant", response_text)
            
            return response_text
        except Exception as e:
            raise Exception(f"Error generating exercise: {str(e)}")

    async def get_course_data(self, course_id: str) -> List[Dict[str, str]]:
        try:
            course_data = self.mongo_service.find_course({"course_id": course_id})
            if not course_data:
                raise Exception(f"Il n'y a pas de cours pour le cours ID: {course_id}")

            return course_data.get("lessons", [])
        except Exception as e:
            raise Exception(f"Error retrieving course data: {str(e)}")
    
    # async def generate_response(self, 
    #                           message: str, 
    #                           context: Optional[List[Dict[str, str]]] = None,
    #                           session_id: Optional[str] = None,
    #                           use_rag: bool = False) -> str:
    #     """Méthode mise à jour pour supporter le RAG"""
    #     rag_context = ""
    #     if use_rag and self.rag_service.vector_store:
    #         relevant_docs = await self.rag_service.similarity_search(message)
    #         rag_context = "\n\n".join(relevant_docs)
        
    #     if session_id:
    #         response = await self.chain_with_history.ainvoke(
    #             {
    #                 "question": message,
    #                 "context": rag_context
    #             },
    #             config={"configurable": {"session_id": session_id}}
    #         )
    #         return response.content
    #     else:
    #         messages = [
    #             SystemMessage(content="Vous êtes un assistant utile et concis.")
    #         ]
            
    #         if rag_context:
    #             messages.append(SystemMessage(
    #                 content=f"Contexte : {rag_context}"
    #             ))
            
    #         if context:
    #             for msg in context:
    #                 if msg["role"] == "user":
    #                     messages.append(HumanMessage(content=msg["content"]))
    #                 elif msg["role"] == "assistant":
    #                     messages.append(AIMessage(content=msg["content"]))
            
    #         messages.append(HumanMessage(content=message))
    #         response = await self.llm.agenerate([messages])
    #         return response.generations[0][0].text

    async def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Récupère l'historique depuis MongoDB"""
        return await self.mongo_service.get_conversation_history(session_id)
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation by session ID."""
        return await self.mongo_service.delete_conversation(session_id)

    async def get_all_sessions(self) -> List[str]:
        """Retrieve all session IDs."""
        return await self.mongo_service.get_all_sessions()

