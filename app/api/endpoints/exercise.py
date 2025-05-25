"""
Endpoints for exercise generation and evaluation
"""
from fastapi import APIRouter, HTTPException, Body, Query, Depends
from models.chat import ChatRequest, ChatResponse
from typing import Dict, List, Optional, Any, Union
from asyncio.log import logger
from bson import ObjectId
import json
import re
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from services.llm_service import LLMService
from services.mongo_service import MongoDBService
from models.exercise import Solution

router = APIRouter()
llm_service = LLMService()
mongo_service = MongoDBService()

# Helper function to get or create a session
async def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get an existing session or create a new one if none exists"""
    if not session_id:
        session_id = await mongo_service.create_conversation()
    return session_id

@router.post("/generate")
async def generate_exercise(
    subject: str = Body(...),
    topic: str = Body(...),
    exercise_type: str = Body(...),
    difficulty: str = Query("medium", enum=["easy", "medium", "hard", "expert"]),
    number_of_questions: int = Query(3, ge=1, le=10),
    session_id: Optional[str] = Body(None),
    include_solutions: bool = Body(False)
) -> Dict[str, Any]:
    """
    Generate exercises based on subject, topic and difficulty level
    """
    try:
        # Ensure we have a valid session
        session_id = await get_or_create_session(session_id)
        
        # Generate exercise using raw JSON approach
        response = await llm_service.generate_exercise(
            subject=subject,
            topic=topic,
            exercise_type=exercise_type,
            difficulty=difficulty,
            number_of_questions=number_of_questions,
            session_id=session_id
        )
        
        # Save the exercise with solutions to MongoDB for smart router access
        exercise_data = {
            "exercise": response["exercise"],
            "solutions": response["solutions"] if "solutions" in response else None,
            "subject": subject,
            "topic": topic,
            "exercise_type": exercise_type,
            "difficulty": difficulty,
            "number_of_questions": number_of_questions,
            "session_id": session_id,
            "created_at": datetime.utcnow()
        }
        
        # Store in exercises collection
        exercise_id = await mongo_service.save_exercise(exercise_data)
        exercise_id_str = str(exercise_id)
        
        # Remove solutions if not requested
        if not include_solutions and "solutions" in response:
            del response["solutions"]
            
        # Add exercise ID to the questions for reference
        for question in response["exercise"]["questions"]:
            question["exercise_id"] = exercise_id_str
        
        # Add exercise ID to instructions for easy reference
        response["exercise"]["instructions"] += f"\n\nExercise ID: {exercise_id_str}"
        
        # Save a reference to the exercise in the conversation
        assistant_message = f"J'ai créé un exercice pour toi sur {topic}. Exercise ID: {exercise_id_str}"
        await mongo_service.save_message(session_id, "assistant", assistant_message, 
                                       metadata={"type": "exercise", "exercise_id": exercise_id_str})
        
        # Add the exercise ID to the response
        response["exercise_id"] = exercise_id_str
        
        return response
    except Exception as e:
        logger.error(f"Exercise generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_exercise(
    exercise_id: str = Body(...),
    user_answers: str = Body(...),
    session_id: Optional[str] = Body(None)
):
    """
    Evaluate user answers for a previously generated exercise
    """
    try:
        # Ensure we have a valid session
        session_id = await get_or_create_session(session_id)
        
        # Use the exercise manager to evaluate the exercise
        result = await llm_service.evaluate_answer(
            exercise_id=exercise_id,
            user_answers=user_answers,
            session_id=session_id
        )
        
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
                                             "evaluation_id": result.get("_id")})
        
        return result
    
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
