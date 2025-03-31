from asyncio.log import logger
from fastapi import APIRouter, HTTPException, Body
from models.chat import ChatRequest, ChatResponse
from models.exercise import ExerciseType, ExerciseResponse, ExerciseRequest, ExerciseContent, Solution
from services.llm_claude import LLMService
from services.mongo_services import MongoDBService
from typing import Dict, Union, Any, Optional, List
from langchain_core.messages import SystemMessage, HumanMessage
import json
import re
import uuid
from datetime import datetime
from bson import ObjectId

router = APIRouter()
llm_service = LLMService()
mongo_service = MongoDBService()

@router.post("/smart", response_model=Union[ChatResponse, ExerciseResponse])
async def smart_chat(
    request: ChatRequest, teacher_id: Optional[str] = None
) -> Union[ChatResponse, ExerciseResponse]:
    """
    Smart endpoint that handles all educational agent interactions:
    - Regular chat
    - Exercise generation
    - Answer evaluation
    - Hints
    - Solutions
    """
    try:
        # Extract common fields
        message = request.message
        session_id = request.session_id
        
        # await mongo_service.save_message(session_id, "user", message)
        
        # Analyze the user intent
        intent_result = await analyze_intent(message, session_id)
        intent = intent_result.get("intent", "chat")
        
        # Handle based on the intent
        if intent == "generate_exercise" or (intent_result.get("is_exercise_request", False)):
            # Extract exercise parameters from the result
            exercise_params = intent_result.get("parameters", {})
            
            # Generate exercise
            response = await llm_service.generate_exercise(
                subject=exercise_params.get("subject", "general"),
                topic=exercise_params.get("topic", ""),
                exercise_type=ExerciseType(exercise_params.get("exercise_type", "multiple_choice")),
                difficulty=exercise_params.get("difficulty", "medium"),
                number_of_questions=exercise_params.get("number_of_questions", 3),
                session_id=session_id,
                teacher_id=teacher_id
            )
            
            # Process the response to ensure correct data types
            if response.solutions:
                for answer in response.solutions.answers:
                    # Convert integer correct_options to strings
                    if "correct_option" in answer and isinstance(answer["correct_option"], int):
                        answer["correct_option"] = str(answer["correct_option"])
                    
                    # Convert any other integer values in lists to strings if needed
                    for key, value in answer.items():
                        if isinstance(value, list):
                            answer[key] = [str(item) if isinstance(item, int) else item for item in value]
            
            # Save the exercise with solutions to MongoDB
            exercise_data = {
                "exercise": response.exercise.model_dump(),
                "solutions": response.solutions.model_dump() if response.solutions else None,
                "subject": exercise_params.get("subject", "general"),
                "topic": exercise_params.get("topic", ""),
                "exercise_type": exercise_params.get("exercise_type", "multiple_choice"),
                "difficulty": exercise_params.get("difficulty", "medium"),
                "number_of_questions": exercise_params.get("number_of_questions", 3),
                "session_id": session_id,
                "teacher_id": teacher_id,
                "created_at": datetime.utcnow()
            }
            
            # Store in exercises collection
            exercise_id = await mongo_service.save_exercise(exercise_data)
            exercise_id_str = str(exercise_id)
            
            # Create a copy without solutions to return to the user
            user_response = ExerciseResponse(
                exercise=response.exercise,
                solutions=None  # Hide solutions
            )
            
            # Add the exercise ID to the exercise content for reference
            user_response.exercise.questions = [
                {**question, "exercise_id": exercise_id_str}
                for question in user_response.exercise.questions
            ]
            
            # Also add the exercise ID to the instructions for easy reference
            user_response.exercise.instructions += f"\n\nExercise ID: {exercise_id_str}"
            
            # Save a reference to the exercise in the conversation
            assistant_message = f"J'ai crée un exercice pour toi {exercise_params.get('topic', '')}. Exercise ID: {exercise_id_str}"
            await mongo_service.save_message(session_id, "assistant", assistant_message, 
                                           metadata={"type": "exercise", "exercise_id": exercise_id_str})
            return user_response
            
        elif intent == "evaluate_answers":
            # Extract parameters
            params = intent_result.get("parameters", {})
            exercise_id = params.get("exercise_id")
            user_answers = params.get("user_answers", [])
            
            if not exercise_id:
                # Try to find the most recent exercise for this session
                recent_exercise = await mongo_service.db.exercises.find_one(
                    {"session_id": session_id},
                    sort=[("created_at", -1)]
                )
                
                if recent_exercise:
                    exercise_id = str(recent_exercise["_id"])
                else:
                    response_text = "I need to know which exercise you're referring to. Please include the exercise ID."
                    await mongo_service.save_message(session_id, "assistant", response_text)
                    return ChatResponse(response=response_text)
            
            # Evaluate the answers
            try:
                evaluation_result = await evaluate_exercise(exercise_id=exercise_id, user_answers=user_answers, session_id=session_id)
                
                # Format the result as a friendly message
                response_text = f"Evaluation results:\n\n"
                response_text += f"Score: {int(float(evaluation_result['score']) * 100)}%\n\n"
                response_text += f"{evaluation_result['feedback']}\n\n"
                
                if evaluation_result.get('question_feedback'):
                    response_text += "Question feedback:\n"
                    for qf in evaluation_result['question_feedback']:
                        status = "✅" if qf['is_correct'] else "❌"
                        response_text += f"Q{qf['question_number']}: {status} {qf['feedback']}\n"
                
                if evaluation_result.get('explanation'):
                    response_text += f"\nDetailed explanation:\n{evaluation_result['explanation']}"
                
                # Save to conversation with metadata reference
                evaluation_id = evaluation_result.get("_id", "")
                await mongo_service.save_message(session_id, "assistant", response_text,
                                               metadata={"type": "evaluation", "exercise_id": exercise_id, 
                                                        "evaluation_id": str(evaluation_id) if evaluation_id else None})
                return ChatResponse(response=response_text)
            except HTTPException as e:
                error_message = f"Error evaluating answers: {e.detail}"
                await mongo_service.save_message(session_id, "assistant", error_message)
                return ChatResponse(response=error_message)
            
        elif intent == "get_hint":
            # Extract parameters
            params = intent_result.get("parameters", {})
            exercise_id = params.get("exercise_id")
            question_number = params.get("question_number")
            
            if not exercise_id:
                response_text = "To give you a hint, I need to know which exercise you're referring to. Please include the exercise ID."
                await mongo_service.save_message(session_id, "assistant", response_text)
                return ChatResponse(response=response_text)
            
            try:
                # Retrieve the exercise
                exercise = await mongo_service.exercises.find_one({"_id": ObjectId(exercise_id)})
                
                if not exercise:
                    response_text = "I couldn't find that exercise. Please check the exercise ID and try again."
                    await mongo_service.save_message(session_id, "assistant", response_text)
                    return ChatResponse(response=response_text)
                
                # Get the exercise content and solutions
                exercise_content = exercise.get("exercise", {})
                solutions = exercise.get("solutions", {})
                
                # Generate a hint using the LLM
                system_prompt = """Vous êtes un assistant éducatif bienveillant.
                
                TÂCHE : Générez un indice utile pour une question d'exercice sans révéler la réponse complète.
                                
                Règles :
                - Fournissez des conseils qui aident l'élève à réfléchir au problème
                - Ne révélez pas la solution entière
                - Soyez encourageant et bienveillant
                - Concentrez-vous uniquement sur la ou les questions demandées
                """
                user_prompt = f"""Exercise question: 
                {json.dumps(exercise_content.get('questions', [])[question_number-1] if question_number else exercise_content)}
                
                Information sur la solution (utilise cela que pour créer ton indice, PAS pour donner la solution):
                {json.dumps(solutions)}
                
                S'il te plait partage un indice utile pour la question  {question_number if question_number else 'this exercise'}.
                """
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = await llm_service.llm.agenerate([messages])
                hint = response.generations[0][0].text
                
                # Save to conversation with metadata
                await mongo_service.save_message(session_id, "assistant", hint,
                                               metadata={"type": "hint", "exercise_id": exercise_id, 
                                                        "question_number": question_number})
                return ChatResponse(response=hint)
            except Exception as e:
                logger.error(f"Error generating hint: {str(e)}")
                error_message = "I'm having trouble generating a hint right now. Please try again."
                await mongo_service.save_message(session_id, "assistant", error_message)
                return ChatResponse(response=error_message)
                
        elif intent == "get_solution":
            # Extract parameters
            params = intent_result.get("parameters", {})
            exercise_id = params.get("exercise_id")
            question_number = params.get("question_number")
            
            if not exercise_id:
                response_text = "To show you the solution, I need to know which exercise you're referring to. Please include the exercise ID."
                await mongo_service.save_message(session_id, "assistant", response_text)
                return ChatResponse(response=response_text)
            
            try:
                # Get the solutions
                solutions = await get_solutions(exercise_id)
                
                # Format the solution as a friendly message
                if question_number is not None:
                    # Return solution for specific question
                    if 0 <= question_number - 1 < len(solutions.answers):
                        answer = solutions.answers[question_number - 1]
                        
                        response_text = f"Solution for Question {question_number}:\n\n"
                        if isinstance(answer, dict):
                            if "correct_option" in answer:
                                response_text += f"Correct option: {answer['correct_option']}\n"
                            if "explanation" in answer:
                                response_text += f"Explanation: {answer['explanation']}\n"
                            if "answer" in answer:
                                response_text += f"Answer: {answer['answer']}\n"
                        else:
                            response_text += f"{answer}\n"
                        # Save to conversation
                        await mongo_service.save_message(session_id, "assistant", response_text,
                                                      metadata={"type": "solution", "exercise_id": exercise_id, 
                                                               "question_number": question_number})
                        return ChatResponse(response=response_text)
                    else:
                        response_text = f"Question {question_number} doesn't exist in this exercise."
                        await mongo_service.save_message(session_id, "assistant", response_text)
                        return ChatResponse(response=f"Question {question_number} doesn't exist in this exercise.")
                else:
                    # Return all solutions
                    response_text = "Solutions for all questions:\n\n"
                    
                    for i, answer in enumerate(solutions.answers):
                        response_text += f"Question {i+1}:\n"
                        if isinstance(answer, dict):
                            if "correct_option" in answer:
                                response_text += f"Correct option: {answer['correct_option']}\n"
                            if "explanation" in answer:
                                response_text += f"Explanation: {answer['explanation']}\n"
                            if "answer" in answer:
                                response_text += f"Answer: {answer['answer']}\n"
                        else:
                            response_text += f"{answer}\n"
                        response_text += "\n"
                    
                    # Save to conversation
                    await mongo_service.save_message(session_id, "assistant", response_text,
                                                   metadata={"type": "solution", "exercise_id": exercise_id})
                    
                    return ChatResponse(response=response_text)
            except HTTPException as e:
                error_message = f"Error retrieving solutions: {e.detail}"
                await mongo_service.save_message(session_id, "assistant", error_message)
                return ChatResponse(response=error_message)
        
        else:  # Default to chat
            # Handle as regular chat
            response = await llm_service.generate_response(
                message=message,
                session_id=session_id,
                teacher_id=teacher_id
            )
            
            return ChatResponse(response=response)
    
    except Exception as e:
        logger.error(f"Smart chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_exercise(
    exercise_id: str = Body(...),
    user_answers: List[Dict[str, Any]] = Body(...),
    session_id: Optional[str] = None
):
    """
    Evaluate user answers for a previously generated exercise.
    
    Body:
    - exercise_id: The unique ID of the exercise to evaluate
    - user_answers: List of answers provided by the user
    - session_id: Optional session ID for conversation tracking
    """
    try:
        # Retrieve the exercise with solutions from MongoDB
        exercise = await mongo_service.exercises.find_one({"_id": ObjectId(exercise_id)})
        
        if not exercise:
            raise HTTPException(status_code=404, detail="Exercise not found")
        
        if not exercise.get("solutions"):
            raise HTTPException(status_code=404, detail="No solutions available for this exercise")
        
        # Prepare for evaluation
        system_prompt = """Vous êtes un assistant d'évaluation pédagogique.
                
        TÂCHE : Évaluez les réponses de l'élève par rapport aux solutions correctes d'un exercice.
                
        Retournez UNIQUEMENT un JSON valide avec la structure suivante :
        {
        "is_correct": true/false,
        "feedback": "Retour global sur la performance",
        "score": décimal entre 0.0 et 1.0,
        "explanation": "Explication détaillée des réponses correctes/incorrectes",
        "question_feedback": [
            {
            "question_number": 1,
            "is_correct": true/false,
            "feedback": "Retour pour cette question spécifique"
            }
        ]
        }
                
        Règles :
        - Comparez chaque réponse de l'élève à la solution correspondante
        - Calculez un score global comme (nombre de réponses correctes / total des questions)
        - Fournissez un retour utile et constructif
        - Soyez indulgent avec les différences mineures d'orthographe ou les variations de formatage
        """
        # Convert exercise and user answers to JSON
        exercise_json = json.dumps({
            "exercise": exercise["exercise"],
            "solutions": exercise["solutions"]
        })
        user_answers_json = json.dumps(user_answers)
        
        user_prompt = f"""Exercise with solutions:
        {exercise_json}
        
        Student answers:
        {user_answers_json}
        
        Veuillez évaluer les réponses de l’étudiant et fournir des commentaires.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await llm_service.llm.agenerate([messages])
        response_text = response.generations[0][0].text
        
        # Extract JSON from response
        json_match = re.search(r'({[\s\S]*})', response_text)
        if json_match:
            json_str = json_match.group(1)
            try:
                result = json.loads(json_str)
                
                # Store the evaluation result in MongoDB for reference
                await mongo_service.db.exercise_evaluations.insert_one({
                    "exercise_id": exercise_id,
                    "user_answers": user_answers,
                    "evaluation": result,
                    "session_id": session_id,
                    "created_at": datetime.utcnow()
                })
                
                return result
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="Failed to parse evaluation data")
        else:
            raise HTTPException(status_code=500, detail="No valid evaluation data found in response")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/solutions/{exercise_id}", response_model=Solution)
async def get_solutions(exercise_id: str):
    """
    Get solutions for a previously generated exercise.
    This endpoint can be used after submission for review purposes.
    """
    try:
        # Retrieve the exercise with solutions from MongoDB
        exercise = await mongo_service.exercises.find_one({"_id": ObjectId(exercise_id)})
        
        if not exercise:
            raise HTTPException(status_code=404, detail="Exercise not found")
        
        if not exercise.get("solutions"):
            raise HTTPException(status_code=404, detail="No solutions available for this exercise")
        
        # Return the solutions
        return Solution(**exercise["solutions"])
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get solutions error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def analyze_intent(message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Use the LLM to determine the user's intent and extract relevant parameters.
    Identifies if the user wants to:
    - Chat about educational topics
    - Generate an exercise
    - Evaluate their answers
    - Get a hint for an exercise
    - See the solution for an exercise
    """
    # Get conversation history if available
    history_context = ""
    if session_id:
        try:
            history = await llm_service.get_conversation_history(session_id)
            if history and len(history) > 0:
                # Format the last few messages for context
                recent_history = history[-5:] if len(history) > 5 else history
                history_context = "\nConversation history:\n" + "\n".join([
                    f"{'User' if msg.get('type') == 'human' else 'Assistant'}: {msg.get('content', '')}"
                    for msg in recent_history
                ])
        except Exception:
            # If we can't get the history, continue without it
            pass
    
    system_prompt = """Vous êtes un classificateur d'intentions et un assistant d'extraction de paramètres pour un chatbot éducatif.
    
    TÂCHE : Déterminez l'intention de l'utilisateur et extrayez les paramètres pertinents.
        
    Retournez UNIQUEMENT un JSON valide avec l'une des structures suivantes selon l'intention de l'utilisateur :
        
    1. Pour une conversation générale :
    {
    "intent": "chat"
    }
        
    2. Pour les demandes de génération d'exercices :
    {
    "intent": "generate_exercise",
    "parameters": {
        "subject": "la matière concernée",
        "topic": "sujet spécifique dans la matière",
        "exercise_type": "qcm",
        "difficulty": "facile/moyen/difficile/expert",
        "number_of_questions": entier entre 1-10
    }
    }
        
    3. Pour l'évaluation des réponses :
    {
    "intent": "evaluate_answers",
    "parameters": {
        "exercise_id": "ID de l'exercice (si fourni)",
        "user_answers": [liste des réponses fournies]
    }
    }
        
    4. Pour demander un indice :
    {
    "intent": "get_hint",
    "parameters": {
        "exercise_id": "ID de l'exercice",
        "question_number": entier (si une question spécifique est mentionnée, ou null)
    }
    }
        
    5. Pour demander les solutions :
    {
    "intent": "get_solution",
    "parameters": {
        "exercise_id": "ID de l'exercice",
        "question_number": entier (si une question spécifique est mentionnée, ou null)
    }
    }
        
    Les formats d'ID d'exercice ressemblent à : 65f123abc456def789abcdef
        
    ATTENTION : Classifiez comme evaluate_answers uniquement si l'utilisateur soumet clairement des réponses pour évaluation.
        
    EXEMPLES :
    - "Donne-moi des problèmes de mathématiques" → generate_exercise
    - "Voici mes réponses : 1. X=5, 2. Y=10..." → evaluate_answers
    - "Je suis bloqué sur la question 3, peux-tu m'aider ?" → get_hint
    - "Montre-moi la réponse à la question 2" → get_solution
    - "Quelle est la capitale de la France ?" → chat
    """
    
    user_prompt = f"Analyse ce message:{message}{history_context}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = await llm_service.llm.agenerate([messages])
    response_text = response.generations[0][0].text
    
    # Extract JSON from response
    json_match = re.search(r'({[\s\S]*})', response_text)
    if json_match:
        json_str = json_match.group(1)
        try:
            result = json.loads(json_str)
            
            # Handle different intents and normalize parameters
            intent = result.get("intent", "chat")
            
            if intent == "generate_exercise":
                params = result.get("parameters", {})
                result["parameters"] = {
                    "subject": params.get("subject", "general"),
                    "topic": params.get("topic", message),
                    "exercise_type": params.get("exercise_type", "multiple_choice"),
                    "difficulty": params.get("difficulty", "medium"),
                    "number_of_questions": int(params.get("number_of_questions", 3))
                }
                # For backward compatibility
                result["is_exercise_request"] = True
            elif intent == "evaluate_answers":
                # Ensure we have the minimum needed parameters
                if not result.get("parameters", {}).get("user_answers"):
                    # If no answers detected, fallback to chat
                    return {"intent": "chat"}
            elif intent in ["get_hint", "get_solution"]:
                # Ensure we have an exercise_id
                if not result.get("parameters", {}).get("exercise_id"):
                    # Generate a response explaining we need the exercise ID
                    return {
                        "intent": "chat",
                        "error": "missing_exercise_id",
                        "message": "I need the exercise ID to provide hints or solutions. Please include the exercise ID in your request."
                    }
            
            return result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM response: {response_text}")
    
    # Default response if extraction fails completely
    return {"intent": "chat"}