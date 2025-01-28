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
from datetime import datetime
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
        
        print("Initialisation du service LLM")
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
        
        
        self.rag_service = RAGService()
    
        # self.rag_prompt = ChatPromptTemplate.from_messages([
        #     ("system", "Vous êtes un assistant utile et concis. "
        #               "Utilisez le contexte fourni pour répondre aux questions."),
        #     ("system", "Contexte : {context}"),
        #     MessagesPlaceholder(variable_name="history"),
        #     ("human", "{question}")
        # ])
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
                config={"configurable": {"session_id": session_id}},
                history=self.conversation_store[session_id]
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
                config={"configurable": {"session_id": session_id}},
                history=self.conversation_store[session_id]
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
    
    async def generate_teacher_response(self, teacher_id: str, message: str, session_id: str) -> str:
        teacher_data = await self.mongo_service.get_teacher(teacher_id)
        if not teacher_data:
            raise ValueError(f"Le professeur {teacher_id} n'existe pas")
        
        teacher_style = teacher_data["prompt_instructions"]

        if not teacher_style or not isinstance(teacher_style, str):
            raise ValueError(f"Prompt instructions invalides pour le professeur {teacher_id}")

        # self.chain = ChatPromptTemplate.from_messages([
        #     ("system",teacher_style),
        #     MessagesPlaceholder(variable_name="history"),
        #     ("human","{question}")
        # ])


        if session_id:
            response = await self.teacher_chain_with_history.ainvoke(
                {
                    "teacher_style": teacher_style,
                    "question": message
                },
                config={"configurable": {"session_id": session_id}},
                history=self.conversation_store[session_id]
            )
            print(response)
            response_text = response.content
            await self.mongo_service.save_message(session_id, "user", message)
            await self.mongo_service.save_message(session_id, "assistant", response_text)
        else:
            response = await self.chain.ainvoke({
                "history": [],
                "question": message
            })
            response_text = response.content
        return response_text

    async def generate_response_rag_service(self, 
                              message: str, 
                              context: Optional[List[Dict[str, str]]] = None,
                              session_id: Optional[str] = None,
                              use_rag: bool = False) -> str:
        """Méthode mise à jour pour supporter le RAG"""
        rag_context = """Tu es un professeur très pédagogue et passionné, capable d’enseigner et de répondre à des questions dans divers domaines éducatifs. Que ce soit les mathématiques, l’histoire, la biologie, la littérature ou d’autres sujets scolaires, tu adaptes ton discours pour fournir des explications claires et engageantes.

        Objectifs principaux :
        Compréhension et pédagogie :
        Explique les concepts, même complexes, de façon accessible et intéressante. Utilise des exemples concrets de la vie quotidienne ou des analogies adaptées au niveau de l’utilisateur.

        Référence au contexte :
        Tu peux répondre aux questions qui font référence aux messages précédents ou à une conversation en cours. Tu es capable de construire une réponse en tenant compte de l’historique de la discussion.

        Adaptabilité :
        Que l’utilisateur soit un élève du primaire ou un adulte souhaitant approfondir ses connaissances, ajuste ton niveau de réponse pour être clair, complet et adapté à la situation.

        Règles et style de réponse :
        Langue :
        Parle toujours en français dans un style bienveillant et clair. Priorise la précision tout en restant accessible.

        Structure et clarté :
        Utilise du contenu structuré en Markdown (titres, sous-titres, listes à puces, formules en LaTeX si nécessaire) pour rendre tes réponses agréables à lire et faciles à comprendre.

        Explications détaillées mais adaptées :
        Si l’utilisateur pose une question sur un sujet, définis chaque notion importante et donne des explications pas à pas. Évite d’être trop technique, sauf si le niveau de l’utilisateur le justifie.

        Ouverture aux sujets variés :

        Tu réponds à toutes les questions éducatives avec précision SI ET SEULEMENT SI ELLES ONT UN RAPPORT AVEC LES INFORMATIONS QUI VONT SUIVRE.
        Si un sujet dépasse les limites de tes connaissances, indique-le poliment tout en guidant l’utilisateur vers des questions éducatives ou des outils pertinents.
        Reste bienveillant même pour les questions ambiguës ou complexes.
        
        Tu dois ABSOLUMENT répondre en te basant UNIQUEMENT sur les informations suivantes :\n\n"""
        if use_rag and self.rag_service.vector_store:
            relevant_docs = await self.rag_service.similarity_search(message)
            rag_context = rag_context.join(relevant_docs)

        print(rag_context)
        
        if session_id:
            response = await self.rag_chain_with_history.ainvoke(
                {
                    "question": message,
                    "context": rag_context
                },
                config={"configurable": {"session_id": session_id}},
                history=self.conversation_store[session_id]
            )
            return response.content
        else:
            messages = [
                SystemMessage(content="Vous êtes un assistant utile et concis.")
            ]
            
            if rag_context:
                messages.append(SystemMessage(
                    content=f"Contexte : {rag_context}"
                ))
            
            if context:
                for msg in context:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
            
            messages.append(HumanMessage(content=message))
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text