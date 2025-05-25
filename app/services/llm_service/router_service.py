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
        
        class RouterOutput(BaseModel):
            destination: str = Field(description="The destination to route to: general_chat, rag_chat, or exercise_handler")
            reasoning: str = Field(description="Brief explanation of the routing decision")
        
        self.router_chain = router_prompt | self.llm.with_structured_output(RouterOutput)
        logger.info("Router chain setup complete")
    
    async def route_query(self, 
                         query: str, 
                         session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Route a query to the appropriate handler based on content
        
        Args:
            query: User's query text
            session_id: Session identifier for conversation tracking
            
        Returns:
            Dictionary with response and metadata about the routing
        """
        try:
            logger.info(f"Processing query: '{query}' for session {session_id}")
            
            if session_id:
                try:
                    session_exists = await self.mongo_service.check_session_exists(session_id)
                    if not session_exists:
                        logger.warning(f"Session {session_id} does not exist. Creating a new session.")
                        session_id = await self.mongo_service.create_conversation()
                except Exception as session_error:
                    logger.error(f"Error validating session: {str(session_error)}")
                    session_id = await self.mongo_service.create_conversation()
                    logger.info(f"Created new session {session_id} due to validation error")
            else:
                session_id = await self.mongo_service.create_conversation()
                logger.info(f"Created new session {session_id}")
            
            try:
                route_result = await self.router_chain.ainvoke({"query": query})
                destination = route_result.destination
                reasoning = route_result.reasoning
                logger.info(f"Routing query to {destination}: {reasoning}")
            except Exception as routing_error:
                logger.error(f"Error in router chain: {str(routing_error)}")
                destination = "general_chat"
                reasoning = "Routing failed, defaulting to general chat"
                logger.info(f"Defaulting to {destination}: {reasoning}")
            
            if destination == "general_chat" or destination == "rag_chat":
                use_rag = (destination == "rag_chat")
                
                if use_rag:
                    doc_count = await self.mongo_service.get_document_count()
                    if doc_count == 0:
                        response = await self.response_generator.generate_response(
                            message=query,
                            session_id=session_id,
                            use_rag=False
                        )
                        await self.mongo_service.save_message(
                            session_id,
                            "assistant",
                            response,
                            metadata={"route": "rag_chat_fallback", "reasoning": "No documents available for RAG query"}
                        )
                        return {
                            "response": "Je n'ai pas accès à des documents spécifiques pour répondre à cette question. " + response,
                            "route": "rag_chat_fallback",
                            "reasoning": "No documents available for RAG query"
                        }
                
                response = await self.response_generator.generate_response(
                    message=query,
                    session_id=session_id,
                    use_rag=use_rag
                )
                
                await self.mongo_service.save_message(
                    session_id,
                    "assistant",
                    response,
                    metadata={"route": destination, "reasoning": reasoning}
                )
                
                return {
                    "response": response,
                    "route": destination,
                    "reasoning": reasoning
                }
                
            elif destination == "exercise_handler":
                try:
                    exercise_intent = await self._analyze_exercise_intent(query)
                except Exception as intent_error:
                    logger.error(f"Error analyzing exercise intent: {str(intent_error)}")
                    response = await self.response_generator.generate_response(
                        message=query,
                        session_id=session_id,
                        use_rag=False
                    )
                    await self.mongo_service.save_message(
                        session_id,
                        "assistant",
                        response,
                        metadata={"route": "exercise_handler_fallback", "reasoning": f"Exercise intent analysis failed: {str(intent_error)}"}
                    )
                    return {
                        "response": "J'ai du mal à comprendre votre demande concernant l'exercice. " + response,
                        "route": "exercise_handler_fallback",
                        "reasoning": f"Exercise intent analysis failed: {str(intent_error)}"
                    }
                
                if exercise_intent.action == "generate":
                    params = exercise_intent.parameters
                    
                    try:
                        exercise_type = params.exercise_type
                    except ValueError:
                        exercise_type = ExerciseType.MULTIPLE_CHOICE
                    
                    exercise_data = await self.exercise_manager.generate_exercise(
                        subject=params.subject,
                        topic=params.topic,
                        exercise_type=exercise_type,
                        difficulty=params.difficulty,
                        number_of_questions=params.number_of_questions,
                        session_id=session_id
                    )
                    
                    if not isinstance(exercise_data, dict):
                        exercise_data = exercise_data.dict()
                    
                    full_exercise_data = {
                        "exercise": exercise_data.get("exercise", {}),
                        "solutions": exercise_data.get("solutions", None),
                        "subject": params.subject,
                        "topic": params.topic,
                        "exercise_type": exercise_type,
                        "difficulty": params.difficulty,
                        "number_of_questions": params.number_of_questions,
                        "session_id": session_id,
                        "created_at": datetime.utcnow(),
                    }
                    
                    exercise_id = await self.mongo_service.save_exercise(full_exercise_data)
                    exercise_id_str = str(exercise_id)
                    
                    exercise_text = self._format_exercise_for_chat(exercise_data, exercise_id_str, params.topic)
                    
                    await self.mongo_service.save_message(
                        session_id,
                        "assistant",
                        exercise_text,
                        metadata={
                            "type": "exercise", 
                            "exercise_id": exercise_id_str,
                            "route": "exercise_handler",
                            "action": "generate"
                        }
                    )
                    
                    return {
                        "response": exercise_text,
                        "route": "exercise_handler",
                        "action": "generate",
                        "exercise_id": exercise_id_str,
                        "reasoning": reasoning
                    }
                    
                elif exercise_intent.action == "evaluate":
                    exercise_id = exercise_intent.parameters.exercise_id
                    user_answers = exercise_intent.parameters.user_answers
                    
                    if not exercise_id:
                        return {
                            "response": "Pour évaluer vos réponses, j'ai besoin de l'ID de l'exercice. Veuillez le fournir.",
                            "route": "exercise_handler",
                            "action": "evaluate_request_id",
                            "reasoning": reasoning
                        }
                    
                    if not user_answers:
                        return {
                            "response": "Pour évaluer vos réponses, veuillez les fournir au format suivant: 'Question 1: [votre réponse], Question 2: [votre réponse]...'",
                            "route": "exercise_handler",
                            "action": "evaluate_request_answers",
                            "reasoning": reasoning
                        }
                    
                    try:
                        result = await self.exercise_manager.evaluate_exercise(
                            exercise_id=exercise_id,
                            user_answers=user_answers,
                            session_id=session_id
                        )
                        
                        response_text = f"Evaluation results:\n\n"
                        response_text += f"Score: {int(float(result.get('score', 0)) * 100)}%\n\n"
                        response_text += f"{result.get('feedback', 'No feedback available.')}\n\n"
                        
                        if result.get('question_feedback'):
                            response_text += "Question feedback:\n"
                            for qf in result['question_feedback']:
                                status = "✅" if qf.get('is_correct', False) else "❌"
                                response_text += f"Q{qf.get('question_number', '?')}: {status} {qf.get('feedback', '')}\n"
                        
                        await self.mongo_service.save_message(
                            session_id,
                            "assistant",
                            response_text,
                            metadata={
                                "type": "evaluation", 
                                "exercise_id": exercise_id,
                                "evaluation_id": result.get("_id", ""),
                                "route": "exercise_handler",
                                "action": "evaluate"
                            }
                        )
                        
                        return {
                            "response": response_text,
                            "route": "exercise_handler",
                            "action": "evaluate",
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
                        hint = await self.exercise_manager.generate_hint(
                            exercise_id=exercise_id,
                            question_number=question_number,
                            session_id=session_id
                        )
                        
                        await self.mongo_service.save_message(
                            session_id,
                            "assistant",
                            hint,
                            metadata={
                                "type": "hint", 
                                "exercise_id": exercise_id,
                                "question_number": question_number,
                                "route": "exercise_handler",
                                "action": "hint"
                            }
                        )
                        
                        return {
                            "response": hint,
                            "route": "exercise_handler",
                            "action": "hint",
                            "exercise_id": exercise_id,
                            "question_number": question_number,
                            "reasoning": reasoning
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
                    exercise_id = exercise_intent.parameters.exercise_id
                    
                    if not exercise_id:
                        return {
                            "response": "Pour vous montrer la solution, j'ai besoin de l'ID de l'exercice. Veuillez le fournir.",
                            "route": "exercise_handler",
                            "action": "solution_request_id",
                            "reasoning": reasoning
                        }
                    
                    try:
                        solutions = await self.exercise_manager.get_solutions(
                            exercise_id=exercise_id,
                            session_id=session_id
                        )
                        
                        history = await self.mongo_service.get_conversation_history(session_id)
                        if history and len(history) > 0:
                            last_message = history[-1]
                            response_text = last_message.get("content", "Solutions retrieved successfully.")
                        else:
                            response_text = self._format_solutions_for_chat(solutions, exercise_id)
                            
                            await self.mongo_service.save_message(
                                session_id,
                                "assistant",
                                response_text,
                                metadata={
                                    "type": "solution", 
                                    "exercise_id": exercise_id,
                                    "route": "exercise_handler",
                                    "action": "solution"
                                }
                            )
                        
                        return {
                            "response": response_text,
                            "route": "exercise_handler",
                            "action": "solution",
                            "exercise_id": exercise_id,
                            "reasoning": reasoning
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
                    return {
                        "response": f"Vous avez demandé une action liée aux exercices ({exercise_intent.action}). Pour des fonctionnalités spécifiques comme l'évaluation, les indices, et les solutions, veuillez utiliser les endpoints dédiés.",
                        "route": "exercise_handler",
                        "action": exercise_intent.action,
                        "reasoning": reasoning
                    }
            
            else:
                response = await self.response_generator.generate_response(
                    message=query,
                    session_id=session_id,
                    use_rag=False
                )
                await self.mongo_service.save_message(
                    session_id,
                    "assistant",
                    response,
                    metadata={"route": "default", "reasoning": "Falling back to default handler"}
                )
                return {
                    "response": response,
                    "route": "default",
                    "reasoning": "Falling back to default handler"
                }
                
        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"Error routing query: {str(e)}\nStack trace: {stack_trace}")
            
            try:
                system_message = "You are a helpful assistant. The user had a question but we encountered an error processing it. Please provide a general helpful response."
                user_message = f"The user asked: '{query}'. Can you give a general response?"
                
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
                
                if hasattr(self.llm, "generate") and callable(self.llm.generate):
                    prompt = system_message + "\n\n" + user_message
                    response_obj = await self.llm.agenerate_prompt([prompt])
                    fallback_response = response_obj.generations[0][0].text
                else:
                    fallback_response = "Je m'excuse, mais j'ai rencontré une erreur technique. Comment puis-je vous aider autrement?"
                
                logger.info(f"Generated fallback response: {fallback_response[:100]}...")
                
                try:
                    if session_id:
                        metadata = {
                            "type": "error",
                            "error": str(e)[:100],
                            "route": "error_handler"
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
                    "route": "error_handler",
                    "error": str(e)
                }
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                return {
                    "response": "Je m'excuse, mais j'ai rencontré une difficulté technique. Veuillez réessayer dans quelques instants.",
                    "route": "error_handler",
                    "error": f"{str(e)}. Fallback also failed: {str(fallback_error)}"
                }
    
    def _format_exercise_for_chat(self, exercise_data: Dict, exercise_id: str, topic: str) -> str:
        """Format an exercise for display in chat"""
        exercise_text = f"Voici un exercice sur {topic}:\n\n"
        
        if "exercise" in exercise_data and "instructions" in exercise_data["exercise"]:
            exercise_text += f"Instructions: {exercise_data['exercise']['instructions']}\n\n"
        
        if "exercise" in exercise_data and "questions" in exercise_data["exercise"]:
            for i, question in enumerate(exercise_data["exercise"]["questions"]):
                exercise_text += f"Question {i+1}: {question.get('question', '')}\n"
                if "options" in question and question["options"]:
                    for j, option in enumerate(question["options"]):
                        exercise_text += f"  {chr(65+j)}) {option}\n"
                exercise_text += "\n"
        
        exercise_text += f"\nExercise ID: {exercise_id}"
        
        return exercise_text
        
    def _format_solutions_for_chat(self, solutions: Dict, exercise_id: str) -> str:
        """Format solutions for display in chat"""
        response_text = "Solutions for all questions:\n\n"
        
        if isinstance(solutions, dict) and "answers" in solutions:
            for i, answer in enumerate(solutions["answers"]):
                response_text += f"Question {i+1}:\n"
                if isinstance(answer, dict):
                    if "correct_option" in answer:
                        response_text += f"Correct option: {answer['correct_option']}\n"
                    if "correct_answer" in answer:
                        response_text += f"Correct answer: {answer['correct_answer']}\n"
                else:
                    response_text += f"{answer}\n"
                
                if "explanations" in solutions and i < len(solutions["explanations"]):
                    response_text += f"Explanation: {solutions['explanations'][i]}\n"
                
                response_text += "\n"
        
        return response_text
    
    async def _analyze_exercise_intent(self, query: str) -> Any:
        """
        Analyze the specific exercise-related intent
        
        Args:
            query: User's query text
            
        Returns:
            ExerciseIntent object with action and parameters
        """
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
        
        result = await (
            exercise_prompt 
            | self.llm.with_structured_output(ExerciseIntent)
        ).ainvoke({"query": query})
        
        if result.action in ["evaluate", "hint", "solution"] and not result.parameters.exercise_id:
            id_match = re.search(r'[0-9a-f]{24}', query)
            if id_match:
                result.parameters.exercise_id = id_match.group(0)
        
        return result