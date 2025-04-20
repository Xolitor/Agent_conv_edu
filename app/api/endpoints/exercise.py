"""
Endpoints for exercise generation and evaluation
"""
from fastapi import APIRouter, HTTPException, Body, Query
from models.chat import ChatRequest, ChatResponse
from models.exercise import ExerciseRequest, ExerciseResponse, ExerciseType, Solution
from services.llm_service import LLMService
from typing import Dict, List, Optional, Any
from asyncio.log import logger
from bson import ObjectId
import json
import re
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage

router = APIRouter()
llm_service = LLMService()
mongo_service = llm_service.mongo_services

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
        response = await llm_service.generate_exercise(
            subject=request.subject,
            topic=request.topic,
            exercise_type=request.exercise_type,
            difficulty=difficulty,
            number_of_questions=number_of_questions,
            session_id=request.session_id,
            teacher_id=request.teacher_id
        )
        
        return ExerciseResponse(
            exercise=response.exercise,
            solutions=response.solutions if request.include_solutions else None
        )
    except Exception as e:
        logger.error(f"Exercise generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_exercise(
    exercise_id: str = Body(...),
    user_answers: List[Dict[str, Any]] = Body(...),
    session_id: Optional[str] = None
):
    """
    Evaluate user answers for a previously generated exercise
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
    Get solutions for a previously generated exercise
    This endpoint can be used after submission for review purposes
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

@router.post("/evaluate-answer", response_model=Dict)
async def evaluate_single_answer(
    exercise_id: str,
    student_answer: str = Body(...),
    session_id: Optional[str] = None
):
    """
    Evaluate a single student answer for an exercise
    """
    try:
        evaluation = await llm_service.evaluate_answer(
            exercise_id=exercise_id,
            student_answer=student_answer,
            session_id=session_id
        )
        
        return {
            "is_correct": evaluation.is_correct,
            "feedback": evaluation.feedback,
            "score": evaluation.score,
            "explanation": evaluation.explanation
        }
    except Exception as e:
        logger.error(f"Answer evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
