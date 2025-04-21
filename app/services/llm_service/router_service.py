"""
Router service for directing queries to the appropriate handler
Uses LangChain's structured output for intent classification and routing
"""
from asyncio.log import logger
from typing import Dict, List, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import re
import json
import traceback
from datetime import datetime
from bson import ObjectId
from models.exercise import ExerciseType, Solution

class RouterService:
    """
    Routes user queries to the appropriate service based on content analysis
    """
    def __init__(self, llm, mongo_service, response_generator, exercise_manager):
        """Initialize with required services and components"""
        self.llm = llm
        self.mongo_service = mongo_service
        self.response_generator = response_generator
        self.exercise_manager = exercise_manager
        
        # Configure the router chain
        self.setup_router_chain()
        logger.info("Router Service initialized")
    
    def setup_router_chain(self):
        """Set up the router chain with destination chains"""
        # Define the routing prompt
        router_prompt = ChatPromptTemplate.from_template(
            """You are a routing assistant for an educational chatbot with multiple specialized functions.

            Based on the user's query, determine which of the following functions would best handle the request:

            1. general_chat: For general knowledge questions, casual conversation, explanations of concepts that don't require specialized document knowledge.
            2. rag_chat: For queries specifically about documents or courses uploaded to the system, when the user is looking for information from their materials.
            3. exercise_handler: For requests to generate exercises, evaluate answers, provide hints, or show solutions to educational problems.

            User query: {query}

            Think step by step, then select the most appropriate destination as one of: "general_chat", "rag_chat", or "exercise_handler".
            """
        )
        
        # Define the structure of the expected output
        class RouterOutput(BaseModel):
            destination: str = Field(description="The destination to route to: general_chat, rag_chat, or exercise_handler")
            reasoning: str = Field(description="Brief explanation of the routing decision")
        
        # Create the router chain with proper structured output
        self.router_chain = router_prompt | self.llm.with_structured_output(RouterOutput)
        
        # For testing and debugging
        logger.info("Router chain setup complete")
    
    async def route_query(self, 
                         query: str, 
                         session_id: Optional[str] = None,
                         teacher_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Route a query to the appropriate handler based on content
        
        Args:
            query: User's query text
            session_id: Session identifier for conversation tracking
            teacher_id: Optional teacher identifier for personalized responses
            
        Returns:
            Dictionary with response and metadata about the routing
        """
        try:
            # Log the incoming query for debugging
            logger.info(f"Processing query: '{query}' for session {session_id}")
            
            # Validate session_id early to catch issues
            if session_id:
                try:
                    # Test if the session exists
                    session_exists = await self.mongo_service.check_session_exists(session_id)
                    if not session_exists:
                        logger.warning(f"Session {session_id} does not exist. Creating a new session.")
                        session_id = await self.mongo_service.create_conversation()
                except Exception as session_error:
                    logger.error(f"Error validating session: {str(session_error)}")
                    # Create a new session if there's an error
                    session_id = await self.mongo_service.create_conversation()
                    logger.info(f"Created new session {session_id} due to validation error")
            else:
                # Create a new session if none provided
                session_id = await self.mongo_service.create_conversation()
                logger.info(f"Created new session {session_id}")
            
            # Get routing decision with better error handling
            try:
                route_result = await self.router_chain.ainvoke({"query": query})
                destination = route_result.destination
                reasoning = route_result.reasoning
                logger.info(f"Routing query to {destination}: {reasoning}")
            except Exception as routing_error:
                logger.error(f"Error in router chain: {str(routing_error)}")
                # Default to general chat if routing fails
                destination = "general_chat"
                reasoning = "Routing failed, defaulting to general chat"
                logger.info(f"Defaulting to {destination}: {reasoning}")
            
            # Process based on destination
            if destination == "general_chat" or destination == "rag_chat":
                # Handle general chat and RAG chat using the existing logic
                
                use_rag = (destination == "rag_chat")
                
                # Check if documents exist when using RAG
                if use_rag:
                    doc_count = await self.mongo_service.get_document_count()
                    if doc_count == 0:
                        # No documents, fall back to general chat but include explanation
                        response = await self.response_generator.generate_response(
                            message=query,
                            session_id=session_id,
                            teacher_id=teacher_id,
                            use_rag=False
                        )
                        return {
                            "response": "Je n'ai pas accès à des documents spécifiques pour répondre à cette question. " + response,
                            "route": "rag_chat_fallback",
                            "reasoning": "No documents available for RAG query"
                        }
                
                # Generate response using the appropriate method
                response = await self.response_generator.generate_response(
                    message=query,
                    session_id=session_id,
                    teacher_id=teacher_id,
                    use_rag=use_rag
                )
                
                return {
                    "response": response,
                    "route": destination,
                    "reasoning": reasoning
                }
                
            elif destination == "exercise_handler":
                # Analyze the exercise intent with better error handling
                try:
                    exercise_intent = await self._analyze_exercise_intent(query)
                except Exception as intent_error:
                    logger.error(f"Error analyzing exercise intent: {str(intent_error)}")
                    # Fallback to general chat with explanation
                    response = await self.response_generator.generate_response(
                        message=query,
                        session_id=session_id,
                        teacher_id=teacher_id,
                        use_rag=False
                    )
                    return {
                        "response": "J'ai du mal à comprendre votre demande concernant l'exercice. " + response,
                        "route": "exercise_handler_fallback",
                        "reasoning": f"Exercise intent analysis failed: {str(intent_error)}"
                    }
                
                # Continue with the correct exercise intent
                if exercise_intent.action == "generate":
                    # Parse exercise parameters
                    params = exercise_intent.parameters
                    
                    # Convert exercise_type to the proper enum
                    try:
                        exercise_type = params.exercise_type
                    except ValueError:
                        # Default to multiple_choice if the exercise type is invalid
                        exercise_type = ExerciseType.MULTIPLE_CHOICE
                    
                    # Generate exercise using the exercise_manager
                    exercise_response = await self.exercise_manager.generate_exercise(
                        subject=params.subject,
                        topic=params.topic,
                        exercise_type=exercise_type,
                        difficulty=params.difficulty,
                        number_of_questions=params.number_of_questions,
                        session_id=session_id,
                        teacher_id=teacher_id
                    )
                    
                    # Store exercise in database with session_id reference
                    exercise_data = {
                        "exercise": exercise_response.exercise.model_dump(),
                        "solutions": exercise_response.solutions.model_dump() if exercise_response.solutions else None,
                        "subject": params.subject,
                        "topic": params.topic,
                        "exercise_type": params.exercise_type,
                        "difficulty": params.difficulty,
                        "number_of_questions": params.number_of_questions,
                        "session_id": session_id,
                        "teacher_id": teacher_id,
                        "created_at": datetime.utcnow()
                    }
                    
                    # Save to MongoDB
                    exercise_id = await self.mongo_service.save_exercise(exercise_data)
                    exercise_id_str = str(exercise_id)
                    
                    # Format as text response
                    exercise_text = f"Voici un exercice sur {params.topic}:\n\n"
                    exercise_text += f"Instructions: {exercise_response.exercise.instructions}\n\n"
                    
                    for i, question in enumerate(exercise_response.exercise.questions):
                        exercise_text += f"Question {i+1}: {question['question']}\n"
                        if "options" in question and question["options"]:
                            for j, option in enumerate(question["options"]):
                                exercise_text += f"  {chr(65+j)}) {option}\n"
                        exercise_text += "\n"
                    
                    # Add exercise ID to response for reference
                    exercise_text += f"\nExercise ID: {exercise_id_str}"
                    
                    # Add to conversation history
                    await self.mongo_service.save_message(
                        session_id, 
                        "assistant",
                        exercise_text,
                        metadata={"type": "exercise", "exercise_id": exercise_id_str}
                    )
                    
                    return {
                        "response": exercise_text,
                        "route": "exercise_handler",
                        "action": "generate",
                        "exercise_id": exercise_id_str,
                        "reasoning": reasoning
                    }
                    
                elif exercise_intent.action == "evaluate":
                    # Extract exercise ID from parameters
                    exercise_id = exercise_intent.parameters.exercise_id
                    user_answers = exercise_intent.parameters.user_answers
                    
                    if not exercise_id:
                        return {
                            "response": "Pour évaluer vos réponses, j'ai besoin de l'ID de l'exercice. Veuillez le fournir.",
                            "route": "exercise_handler",
                            "action": "evaluate_request_id",
                            "reasoning": reasoning
                        }
                    
                    try:
                        # Fetch the exercise to check if it exists
                        exercise = await self.mongo_service.get_exercise(exercise_id)
                        if not exercise:
                            return {
                                "response": f"Je ne trouve pas d'exercice avec l'ID {exercise_id}. Veuillez vérifier l'ID.",
                                "route": "exercise_handler",
                                "action": "evaluate_error",
                                "reasoning": "Exercise not found"
                            }
                        
                        # This would typically be handled by the dedicated endpoint
                        # For chat, we just provide guidance on how to submit answers
                        return {
                            "response": f"J'ai bien reçu votre demande d'évaluation pour l'exercice {exercise_id}. Pour les évaluer correctement, veuillez utiliser l'endpoint d'évaluation dédié ou soumettre vos réponses au format suivant: 'Question 1: [votre réponse], Question 2: [votre réponse]...'",
                            "route": "exercise_handler",
                            "action": "evaluate_redirect",
                            "exercise_id": exercise_id,
                            "reasoning": reasoning
                        }
                    except Exception as e:
                        logger.error(f"Error evaluating exercise: {str(e)}")
                        return {
                            "response": f"Une erreur s'est produite lors de l'évaluation de l'exercice: {str(e)}",
                            "route": "exercise_handler",
                            "action": "evaluate_error",
                            "reasoning": str(e)
                        }
                        
                elif exercise_intent.action == "hint":
                    # Extract exercise ID and question number
                    exercise_id = exercise_intent.parameters.exercise_id
                    question_number = exercise_intent.parameters.question_number
                    
                    if not exercise_id:
                        return {
                            "response": "Pour vous donner un indice, j'ai besoin de l'ID de l'exercice. Veuillez le fournir.",
                            "route": "exercise_handler",
                            "action": "hint_request_id",
                            "reasoning": reasoning
                        }
                    
                    try:
                        # Generate hint using the exercise_manager method
                        hint = await self.exercise_manager.generate_hint(
                            exercise_id=exercise_id,
                            question_number=question_number,
                            session_id=session_id
                        )
                        
                        return {
                            "response": hint,
                            "route": "exercise_handler",
                            "action": "hint",
                            "exercise_id": exercise_id,
                            "question_number": question_number,
                            "reasoning": reasoning
                        }
                    except ValueError as ve:
                        # Handle specific value errors
                        return {
                            "response": str(ve),
                            "route": "exercise_handler",
                            "action": "hint_error",
                            "reasoning": str(ve)
                        }
                    except Exception as e:
                        logger.error(f"Error getting hint: {str(e)}")
                        return {
                            "response": f"Une erreur s'est produite lors de la génération de l'indice: {str(e)}",
                            "route": "exercise_handler",
                            "action": "hint_error",
                            "reasoning": str(e)
                        }
                
                elif exercise_intent.action == "solution":
                    # Extract exercise ID
                    exercise_id = exercise_intent.parameters.exercise_id
                    
                    if not exercise_id:
                        return {
                            "response": "Pour vous montrer la solution, j'ai besoin de l'ID de l'exercice. Veuillez le fournir.",
                            "route": "exercise_handler",
                            "action": "solution_request_id",
                            "reasoning": reasoning
                        }
                    
                    try:
                        # Get solutions using the exercise_manager method
                        solutions = await self.exercise_manager.get_solutions(
                            exercise_id=exercise_id,
                            session_id=session_id
                        )
                        
                        # The get_solutions method already formats and saves the message to the conversation history
                        # We just need to return the formatted response or access it from the conversation history
                        
                        # Get the last message from conversation history to use as response
                        history = await self.mongo_service.get_conversation_history(session_id)
                        if history and len(history) > 0:
                            last_message = history[-1]
                            response_text = last_message.get("content", "Solutions retrieved successfully.")
                        else:
                            # Fall back to a basic response if the history retrieval fails
                            response_text = "Les solutions ont été récupérées avec succès. Veuillez consulter l'historique des messages."
                        
                        return {
                            "response": response_text,
                            "route": "exercise_handler",
                            "action": "solution",
                            "exercise_id": exercise_id,
                            "reasoning": reasoning
                        }
                    except ValueError as ve:
                        # Handle specific value errors
                        return {
                            "response": str(ve),
                            "route": "exercise_handler",
                            "action": "solution_error",
                            "reasoning": str(ve)
                        }
                    except Exception as e:
                        logger.error(f"Error getting solution: {str(e)}")
                        return {
                            "response": f"Une erreur s'est produite lors de la récupération de la solution: {str(e)}",
                            "route": "exercise_handler",
                            "action": "solution_error",
                            "reasoning": str(e)
                        }
                
                else:
                    # For other exercise related actions
                    return {
                        "response": f"Vous avez demandé une action liée aux exercices ({exercise_intent.action}). Pour des fonctionnalités spécifiques comme l'évaluation, les indices, et les solutions, veuillez utiliser les endpoints dédiés.",
                        "route": "exercise_handler",
                        "action": exercise_intent.action,
                        "reasoning": reasoning
                    }
            
            else:
                # Default case - general chat
                response = await self.response_generator.generate_response(
                    message=query,
                    session_id=session_id,
                    teacher_id=teacher_id,
                    use_rag=False
                )
                return {
                    "response": response,
                    "route": "default",
                    "reasoning": "Falling back to default handler"
                }
                
        except Exception as e:
            # Get detailed stack trace
            stack_trace = traceback.format_exc()
            logger.error(f"Error routing query: {str(e)}\nStack trace: {stack_trace}")
            
            # Create fallback response with simpler approach
            try:
                # Generate simple response
                system_message = "You are a helpful assistant. The user had a question but we encountered an error processing it. Please provide a general helpful response."
                user_message = f"The user asked: '{query}'. Can you give a general response?"
                
                # Use a basic prompt format instead of invoke
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
                
                # Use direct method to generate response if available
                if hasattr(self.llm, "generate") and callable(self.llm.generate):
                    prompt = system_message + "\n\n" + user_message
                    response_obj = await self.llm.agenerate_prompt([prompt])
                    fallback_response = response_obj.generations[0][0].text
                else:
                    # Try a simple string-based approach
                    fallback_response = "Je m'excuse, mais j'ai rencontré une erreur technique. Comment puis-je vous aider autrement?"
                
                # Log the fallback response
                logger.info(f"Generated fallback response: {fallback_response[:100]}...")
                
                # Try to save error to conversation history
                try:
                    if session_id:
                        metadata = {
                            "type": "error",
                            "error": str(e)[:100],  # Truncate long errors
                            "route": "error_handler"  # Use error_handler, not error_fallback
                        }
                        await self.mongo_service.save_message(
                            session_id,
                            "assistant",
                            fallback_response,
                            metadata=metadata
                        )
                        logger.info(f"Saved error message to conversation history for session {session_id}")
                except Exception as save_error:
                    logger.error(f"Failed to save error message: {str(save_error)}")
                
                return {
                    "response": fallback_response,
                    "route": "error_handler",  # Change from error_fallback to error_handler
                    "error": str(e)
                }
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                return {
                    "response": "Je m'excuse, mais j'ai rencontré une difficulté technique. Veuillez réessayer dans quelques instants.",
                    "route": "error_handler",  # Change from error_fallback to error_handler
                    "error": f"{str(e)}. Fallback also failed: {str(fallback_error)}"
                }
    
    async def _analyze_exercise_intent(self, query: str) -> Any:
        """
        Analyze the specific exercise-related intent
        
        Args:
            query: User's query text
            
        Returns:
            ExerciseIntent object with action and parameters
        """
        # Define the structure for exercise intent analysis
        class ExerciseParameters(BaseModel):
            subject: str = Field(description="The academic subject (math, history, etc.)")
            topic: str = Field(description="The specific topic within the subject")
            exercise_type: str = Field(description="Type of exercise (multiple_choice, fill_in_blank, short_answer, code_challenge, true_false, math_problem)")
            difficulty: str = Field(description="How difficult the exercise should be (easy, medium, hard)")
            number_of_questions: int = Field(description="How many questions to generate (default 3)")
            exercise_id: Optional[str] = Field(None, description="ID of an existing exercise if applicable")
            question_number: Optional[int] = Field(None, description="Question number if referring to a specific question")
            user_answers: Optional[List[Any]] = Field(None, description="User's answers if evaluating")
        
        class ExerciseIntent(BaseModel):
            action: str = Field(description="The exercise-related action: generate, evaluate, hint, or solution")
            parameters: ExerciseParameters = Field(description="Parameters for the exercise-related action")
        
        exercise_prompt = ChatPromptTemplate.from_template(
            """Analyze this query related to educational exercises:
            
            "{query}"
            
            Determine the specific exercise-related action requested:
            - generate: User wants to create new exercises
            - evaluate: User wants answers evaluated
            - hint: User is asking for a hint
            - solution: User wants to see solutions
            
            Extract these parameters based on the action:
            
            For "generate" action:
            - subject: The academic subject (math, history, etc.)
            - topic: The specific topic within the subject
            - exercise_type: Type of exercise (multiple_choice, short_answer, etc.)
            - difficulty: How difficult the exercise should be (easy, medium, hard)
            - number_of_questions: How many questions to generate
            
            For "evaluate", "hint", and "solution" actions:
            - exercise_id: Look for an exercise ID in the format of a hexadecimal string (e.g., "65f123abc456def789abcdef")
            - question_number: If a specific question is mentioned (integer)
            - user_answers: For evaluation, try to extract the answers provided by the user
            
            If an exercise ID is not explicitly mentioned but should be required for the action, still specify the action but leave exercise_id null.
            """
        )
        
        # Parse the exercise intent with proper structured output
        result = await (
            exercise_prompt 
            | self.llm.with_structured_output(ExerciseIntent)
        ).ainvoke({"query": query})
        
        # If the action requires an exercise_id but none was found, try to extract it
        if result.action in ["evaluate", "hint", "solution"] and not result.parameters.exercise_id:
            # Try to extract exercise ID using regex pattern
            id_match = re.search(r'[0-9a-f]{24}', query)
            if id_match:
                result.parameters.exercise_id = id_match.group(0)
        
        return result