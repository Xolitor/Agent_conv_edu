
from asyncio.log import logger
from fastapi import APIRouter, HTTPException, Body, Query
from models.chat import ChatRequest, ChatResponse
from models.exercise import ExerciseRequest, ExerciseResponse, ExerciseType
from services.llm_claude import LLMService
from typing import Dict, List, Optional

router = APIRouter()
llm_service = LLMService()

@router.post("/generate-exercise", response_model=ExerciseResponse)
async def generate_exercise(
    request: ExerciseRequest,
    difficulty: str = Query("medium", enum=["easy", "medium", "hard", "expert"]),
    number_of_questions: int = Query(3, ge=1, le=10)
) -> ExerciseResponse:
    """Generate exercises based on subject, topic and difficulty level"""
    try:
        # Use the existing LLM service with a special prompt for exercise generation
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

@router.post("/evaluate-answer", response_model=Dict)
async def evaluate_answer(
    exercise_id: str,
    student_answer: str = Body(...),
    session_id: Optional[str] = None
):
    """Evaluate a student's answer to an exercise"""
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
