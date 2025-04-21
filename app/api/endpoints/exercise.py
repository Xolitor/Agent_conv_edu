"""
Endpoints for exercise generation and evaluation
"""
from fastapi import APIRouter, HTTPException, Body, Query, Depends
from models.chat import ChatRequest, ChatResponse
from models.exercise import ExerciseRequest, ExerciseResponse, ExerciseType, Solution
from services.llm_service import LLMService
from typing import Dict, List, Optional, Any, Union
from asyncio.log import logger
from bson import ObjectId
import json
import re
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage

router = APIRouter()
llm_service = LLMService()
mongo_service = llm_service.mongo_services

# Helper function to get or create a session
async def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get an existing session or create a new one if none exists"""
    if not session_id:
        session_id = await mongo_service.create_conversation()
    return session_id

@router.post("/generate", response_model=ExerciseResponse)
async def generate_exercise(
    request: ExerciseRequest,
    difficulty: str = Query("medium", enum=["easy", "medium", "hard", "expert"]),
    number_of_questions: int = Query(3, ge=1, le=10)
) -> ExerciseResponse:
    """
    Generate exercises based on subject, topic and difficulty level
    """
    try:
        # Ensure we have a valid session
        session_id = await get_or_create_session(request.session_id)
        
        response = await llm_service.generate_exercise(
            subject=request.subject,
            topic=request.topic,
            exercise_type=request.exercise_type,
            difficulty=difficulty,
            number_of_questions=number_of_questions,
            session_id=session_id,
            teacher_id=request.teacher_id
        )
        
        # Save the exercise with solutions to MongoDB for smart router access
        exercise_data = {
            "exercise": response.exercise.model_dump(),
            "solutions": response.solutions.model_dump() if response.solutions else None,
            "subject": request.subject,
            "topic": request.topic,
            "exercise_type": request.exercise_type.value,
            "difficulty": difficulty,
            "number_of_questions": number_of_questions,
            "session_id": session_id,
            "teacher_id": request.teacher_id,
            "created_at": datetime.utcnow()
        }
        
        # Store in exercises collection
        exercise_id = await mongo_service.save_exercise(exercise_data)
        exercise_id_str = str(exercise_id)
        
        # Add exercise_id to exercise data for reference
        if not request.include_solutions:
            response.solutions = None
            
        # Add exercise ID to the questions for reference
        response.exercise.questions = [
            {**question, "exercise_id": exercise_id_str}
            for question in response.exercise.questions
        ]
        
        # Add exercise ID to instructions for easy reference
        response.exercise.instructions += f"\n\nExercise ID: {exercise_id_str}"
        
        # Save a reference to the exercise in the conversation
        assistant_message = f"J'ai créé un exercice pour toi sur {request.topic}. Exercise ID: {exercise_id_str}"
        await mongo_service.save_message(session_id, "assistant", assistant_message, 
                                       metadata={"type": "exercise", "exercise_id": exercise_id_str})
        
        return response
    except Exception as e:
        logger.error(f"Exercise generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_exercise(
    exercise_id: str = Body(...),
    user_answers: List[Dict[str, Any]] = Body(...),
    session_id: Optional[str] = Body(None)
):
    """
    Evaluate user answers for a previously generated exercise
    """
    try:
        # Ensure we have a valid session
        session_id = await get_or_create_session(session_id)
        
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
        
        Veuillez évaluer les réponses de l'étudiant et fournir des commentaires.
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
                evaluation_id = await mongo_service.db.exercise_evaluations.insert_one({
                    "exercise_id": exercise_id,
                    "user_answers": user_answers,
                    "evaluation": result,
                    "session_id": session_id,
                    "created_at": datetime.utcnow()
                })
                
                # Add the evaluation ID to the result
                result["_id"] = str(evaluation_id.inserted_id)
                
                # Format the result as a friendly message for conversation history
                response_text = f"Evaluation results:\n\n"
                response_text += f"Score: {int(float(result['score']) * 100)}%\n\n"
                response_text += f"{result['feedback']}\n\n"
                
                if result.get('question_feedback'):
                    response_text += "Question feedback:\n"
                    for qf in result['question_feedback']:
                        status = "✅" if qf['is_correct'] else "❌"
                        response_text += f"Q{qf['question_number']}: {status} {qf['feedback']}\n"
                
                # Save to conversation with metadata reference
                await mongo_service.save_message(session_id, "assistant", response_text,
                                             metadata={"type": "evaluation", "exercise_id": exercise_id, 
                                                     "evaluation_id": str(evaluation_id.inserted_id)})
                
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
async def get_solutions(exercise_id: str, session_id: Optional[str] = None):
    """
    Get solutions for a previously generated exercise
    This endpoint can be used after submission for review purposes
    """
    try:
        # Ensure we have a valid session if provided
        if session_id:
            session_id = await get_or_create_session(session_id)
        
        # Retrieve the exercise with solutions from MongoDB
        exercise = await mongo_service.exercises.find_one({"_id": ObjectId(exercise_id)})
        
        if not exercise:
            raise HTTPException(status_code=404, detail="Exercise not found")
        
        if not exercise.get("solutions"):
            raise HTTPException(status_code=404, detail="No solutions available for this exercise")
        
        # If session_id is provided, save to conversation history
        if session_id:
            # Format the solution as a friendly message
            response_text = "Solutions for all questions:\n\n"
            
            for i, answer in enumerate(exercise["solutions"]["answers"]):
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
        
        # Return the solutions
        return Solution(**exercise["solutions"])
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get solutions error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hint", response_model=ChatResponse)
async def get_hint(
    exercise_id: str = Body(...),
    question_number: Optional[int] = Body(None),
    session_id: Optional[str] = Body(None)
):
    """
    Get a hint for a specific exercise question
    """
    try:
        # Ensure we have a valid session
        session_id = await get_or_create_session(session_id)
        
        # Retrieve the exercise
        exercise = await mongo_service.exercises.find_one({"_id": ObjectId(exercise_id)})
        
        if not exercise:
            raise HTTPException(status_code=404, detail="Exercise not found")
        
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
        
        S'il te plait partage un indice utile pour la question {question_number if question_number else 'this exercise'}.
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
        logger.error(f"Get hint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
