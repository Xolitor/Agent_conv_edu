# services/llm_service.py
"""
Service principal gérant les interactions avec le modèle de langage
Compatible avec les fonctionnalités du TP1 et du TP2
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from services.memory import InMemoryHistory
import os
from typing import List, Dict, Optional
from services.mongo_service import MongoService

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
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Récupère ou crée l'historique pour une session donnée"""
        if session_id not in self.conversation_store:
            self.conversation_store[session_id] = InMemoryHistory()
        return self.conversation_store[session_id]

    async def generate_response(self, 
                              message: str, 
                              context: Optional[List[Dict[str, str]]] = None,
                              session_id: Optional[str] = None) -> str:
        """
        Méthode unifiée pour générer des réponses
        Supporte les deux modes : avec contexte (TP1) et avec historique (TP2)
        """
        if session_id:
            # Mode TP2 avec historique
            response = await self.chain_with_history.ainvoke(
                {"question": message},
                config={"configurable": {"session_id": session_id}}
            )
            
            response_text = response.content
            await self.mongo_service.save_message(session_id, "user", message)
            await self.mongo_service.save_message(session_id, "assistant", response_text)
        else:
            # Mode TP1 avec contexte explicite
            messages = [SystemMessage(content="Vous êtes un assistant utile et concis.")]
            
            if context:
                for msg in context:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                        
                # Ajout du nouveau message
                messages.append(HumanMessage(content=message))
                # Génération de la réponse
                response = await self.llm.agenerate([messages])
                response_text = response.generations[0][0].text
            else:
            # Générer une réponse sans contexte spécifique
                response = await self.llm.agenerate([[SystemMessage(content="Vous êtes un assistant utile et concis."), 
                                                    HumanMessage(content=message)]])
                response_text = response.generations[0][0].text
            
            # messages.append(HumanMessage(content=message))
            # response = await self.llm.agenerate([messages])
        
        return response_text
            
            # await self.mongo_service.save_message(session_id, "user", message)
            # await self.mongo_service.save_message(session_id, "assistant", response_text)
            # return response_text

            # return response.generations[0][0].text
            
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Récupère l'historique depuis MongoDB"""
        return await self.mongo_service.get_conversation_history(session_id)


    # def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
    #     """Récupère l'historique d'une conversation spécifique"""
    #     if session_id in self.conversation_store:
    #         history = self.conversation_store[session_id].messages
    #         return [
    #             {
    #                 "role": "user" if isinstance(msg, HumanMessage) else "assistant",
    #                 "content": msg.content
    #             }
    #             for msg in history
    #         ]
    #     return []
    


