"""
Response generation component for LLM service
Handles chat message processing and LLM interaction
"""
from asyncio.log import logger
from fastapi import HTTPException
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Dict, List, Optional, Any

class ResponseGenerator:
    """
    Generates responses using LLM models with context management
    """
    def __init__(self, llm, mongo_service, session_manager):
        """Initialize with LLM and MongoDB service"""
        self.llm = llm
        self.mongo_service = mongo_service
        self.session_manager = session_manager
        logger.info("Response Generator initialized")
    
    @property
    def default_system_prompt(self) -> str:
        return """Vous êtes un assistant utile et concis qui retourne ses réponses en format Markdown. 
        Répondez toujours avec un formatage clair, en utilisant des titres, des listes."""

    @property
    def rag_system_prompt(self) -> str:
        return """Tu es un assistant pédagogue expert qui génère des réponses précises et utiles basées sur le contexte fourni.

                Règles fondamentales :
                1. Analyse du contexte :
                   - Base tes réponses UNIQUEMENT sur le contexte fourni
                   - Si le contexte est insuffisant, indique-le clairement
                   - Cite explicitement les parties pertinentes du contexte
                
                2. Structure de réponse :
                   - Organise ta réponse de manière logique et claire
                   - Utilise des paragraphes distincts pour chaque point important
                   - Emploie du Markdown pour améliorer la lisibilité
                
                3. Style de communication :
                   - Adopte un ton professionnel mais accessible
                   - Explique les concepts complexes simplement
                   - Utilise des exemples concrets quand c'est pertinent
                
                4. Précision et honnêteté :
                   - Ne fais pas de suppositions hors du contexte
                   - Indique clairement les limites de l'information disponible
                   - Si des informations semblent contradictoires, signale-le
                
                5. Synthèse :
                   - Commence par une réponse directe à la question
                   - Développe ensuite avec des détails pertinents
                   - Termine par une conclusion claire si nécessaire

                Contexte fourni : \n\n"""
    
    async def generate_response(self,
                              message: str,
                              session_id: Optional[str] = None,
                              teacher_id: Optional[str] = None,
                              use_rag: bool = False) -> str:
        """Unified response generation method"""
        try:
            # Use the session_manager's _ensure_session method instead
            session_context = await self.session_manager._ensure_session(session_id)
            session_id = session_context.session_id
            history = session_context.history
            
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
                relevant_docs = await self.mongo_service.similarity_search(message)
                if relevant_docs:
                    rag_context = "\n\n".join(doc["text"] for doc in relevant_docs)
                    messages.append(SystemMessage(content=self.rag_system_prompt + rag_context))
            else:
                messages.append(SystemMessage(content=self.default_system_prompt))

            # Add conversation history if exists
            for msg in history:
                if isinstance(msg, dict):
                    # Dictionary access
                    if msg.get("role") == "user":
                        messages.append(HumanMessage(content=msg.get("content", "")))
                    elif msg.get("role") == "assistant":
                        messages.append(AIMessage(content=msg.get("content", "")))
                else:
                    # Object attribute access
                    try:
                        if hasattr(msg, "role") and msg.role == "user":
                            messages.append(HumanMessage(content=msg.content))
                        elif hasattr(msg, "role") and msg.role == "assistant":
                            messages.append(AIMessage(content=msg.content))
                    except AttributeError:
                        logger.warning(f"Malformed message ignored: {msg}")

            # Add the current message
            messages.append(HumanMessage(content=message))

            # Generate response
            response = await self.llm.agenerate([messages])
            response_text = response.generations[0][0].text

            return response_text

        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
