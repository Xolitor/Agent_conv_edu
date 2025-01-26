# services/llm_service.py
"""
Service principal gérant les interactions avec le modèle de langage
Compatible avec les fonctionnalités du TP1 et du TP2
"""
import logging
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import LLM
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

from services.memory import InMemoryHistory
import os
from typing import List, Dict, Optional
from services.mongo_service import MongoService
from services.rag_mongo_services import RAGServiceMongo

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
        
        self.conversation_store = {}
        
        #################### Configuration de plusieurs chaînes de traitement ####################
        
        self.explanation_chain  = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful and concise educational assistant."),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{question}")
        ]) | self.llm

        self.exercise_chain = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a creative educational assistant."),
            HumanMessage(content="Generate 3 exercises based on this topic: {topic}.")
        ]) | self.llm
        
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
        
        
        self.rag_service = RAGServiceMongo()
    
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant utile et concis. "
                      "Utilisez le contexte fourni pour répondre aux questions."),
            ("system", "Contexte : {context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
                #################### Chaine qui gère l'historique ####################
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant utile et concis qui retourne ses réponses en format Markdown. Répondez toujours avec un formatage clair, en utilisant des titres, des listes."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ]) 
        self.chain = self.prompt | self.llm
        
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
        
    async def create_new_conversation(self, user_id: str) -> str:
        """Crée une nouvelle conversation et génère un ID unique."""
        session_id = f"session_{uuid.uuid4()}"
        await self.mongo_service.create_conversation(session_id, user_id)
        return session_id
        
    async def generate_response_sequencing(self, message: str, session_id: str = "") -> str:
        """
        Generate a comprehensive response with multiple processing steps

        Args:
            message (str): User's input message
            session_id (str, optional): Session identifier for conversation context

        Returns:
            str: Processed response
        """
        # Generate main response using the main prompt
        main_chain = self.main_prompt | self.llm
        main_response = (await main_chain.ainvoke({
            "history":  [],
            "question": message
        })).content

        # Generate bullet points from the main response
        bullet_points_response = (await self.bullet_points_chain.ainvoke({
            "text": main_response
        })).content

        # Generate a one-liner summary
        one_liner_response = (await self.one_liner_chain.ainvoke({
            "text": bullet_points_response
        })).content

        # Return the main response with additional processing
        return one_liner_response
            
    def generate_explanation(self, topic: str, reference: str) -> str:
        """
        Generates an explanation for the given topic with optional personalization.
        :param topic: The topic to explain.
        :param reference: A personalized reference (e.g., video games or pop culture).
        :return: A string explanation.
        """
        question = f"Explain {topic} using examples from {reference}."
        response = self.explanation_chain({"question": question})
        return response

    def generate_exercises(self, topic: str) -> str:
        """
        Generates exercises based on the given topic.
        :param topic: The topic for which exercises are generated.
        :return: A list of exercises as a string.
        """
        response = self.exercise_chain.run({"topic": topic})
        return response
            
    async def generate_response_rag_mongo(
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
            await self.mongo_service.save_message(session_id, "user", message)
            await self.mongo_service.save_message(session_id, "assistant", response.content)
            
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
    
    async def generate_response(self, 
                            message: str, 
                            context: Optional[List[Dict[str, str]]] = None,
                            session_id: Optional[str] = None) -> str:
        """
        Méthode unifiée pour générer des réponses
        Supporte les deux modes : avec contexte (TP1) et avec historique (TP2)
        """
        if session_id:
            response = await self.chain_with_history.ainvoke(
                {"question": message},
                config={"configurable": {"session_id": session_id}}
            )
            
            response_text = response.content
            await self.mongo_service.save_message(session_id, "user", message)
            await self.mongo_service.save_message(session_id, "assistant", response_text)
        else:
            # Mode TP1 avec contexte explicite
            messages = [SystemMessage(content="Vous êtes un assistant utile et concis qui retourne ses réponses en format Markdown. Répondez toujours avec un formatage clair, en utilisant des titres, des listes.")]
            
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
                response = await self.llm.agenerate(HumanMessage(content=message))
                response_text = response.generations[0][0].text
        
        return response_text
    
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Récupère l'historique depuis MongoDB"""
        return await self.mongo_service.get_conversation_history(session_id)
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation by session ID."""
        return await self.mongo_service.delete_conversation(session_id)

    async def get_all_sessions(self) -> List[str]:
        """Retrieve all session IDs."""
        return await self.mongo_service.get_all_sessions()
