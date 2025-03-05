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
    Smart endpoint that handles both regular chat and exercise requests.
    It uses the LLM to detect intent and extract parameters.
    """
    try:
        # Extract common fields
        message = request.message
        session_id = request.session_id
        
        # First step: Use LLM to determine if this is an exercise request
        intent_result = await analyze_intent(message, session_id)
        
        if intent_result["is_exercise_request"]:
            # Extract exercise parameters from the result
            exercise_params = intent_result["parameters"]
            
            # Generate exercise
            response = await llm_service.generate_exercise(
                subject=exercise_params["subject"],
                topic=exercise_params["topic"],
                exercise_type=ExerciseType(exercise_params["exercise_type"]),
                difficulty=exercise_params["difficulty"],
                number_of_questions=exercise_params["number_of_questions"],
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
                "subject": exercise_params["subject"],
                "topic": exercise_params["topic"],
                "exercise_type": exercise_params["exercise_type"],
                "difficulty": exercise_params["difficulty"],
                "number_of_questions": exercise_params["number_of_questions"],
                "session_id": session_id,
                "teacher_id": teacher_id,
                "created_at": datetime.utcnow()
            }
            
            # Store in exercises collection
            exercise_id = await mongo_service.db.exercises.insert_one(exercise_data)
            exercise_id_str = str(exercise_id.inserted_id)
            
            # Create a copy without solutions to return to the user
            user_response = ExerciseResponse(
                exercise=response.exercise,
                solutions=None  # Hide solutions
            )
            
            # Add the exercise ID to the exercise content for reference
            # We'll add it directly to the questions field to avoid model changes
            user_response.exercise.questions = [
                {**question, "exercise_id": exercise_id_str}
                for question in user_response.exercise.questions
            ]
            
            # Also add the exercise ID to the instructions for easy reference
            user_response.exercise.instructions += f"\n\nExercise ID: {exercise_id_str}"
            
            return user_response
        else:
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
        system_prompt = """You are an education assessment assistant.
        
        TASK: Evaluate student answers against the correct solutions for an exercise.
        
        Return ONLY valid JSON with the following structure:
        {
          "is_correct": true/false,
          "feedback": "Overall feedback on performance",
          "score": decimal between 0.0 and 1.0,
          "explanation": "Detailed explanation of correct/incorrect answers",
          "question_feedback": [
            {
              "question_number": 1,
              "is_correct": true/false,
              "feedback": "Feedback for this specific question"
            }
          ]
        }
        
        Rules:
        - Compare each student answer against the corresponding solution
        - Calculate an overall score as (number of correct answers / total questions)
        - Provide helpful, constructive feedback
        - Be lenient with minor spelling differences or formatting variations
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
        
        Please evaluate the student's answers and provide feedback.
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
        exercise = await mongo_service.db.exercises.find_one({"_id": ObjectId(exercise_id)})
        
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
    Use the LLM to determine if the message is requesting an exercise and extract parameters.
    Returns both the intent classification and parameters in one call to minimize LLM requests.
    """
    # Get conversation history if available
    history_context = ""
    if session_id:
        try:
            history = await llm_service.get_conversation_history(session_id)
            if history and len(history) > 0:
                # Format the last few messages for context
                recent_history = history[-3:] if len(history) > 3 else history
                history_context = "\nConversation history:\n" + "\n".join([
                    f"{'User' if msg.get('type') == 'human' else 'Assistant'}: {msg.get('content', '')}"
                    for msg in recent_history
                ])
        except Exception:
            # If we can't get the history, continue without it
            pass
    
    system_prompt = """You are an intent classifier and parameter extraction assistant.
    
    TASK 1: Determine if the user is requesting an exercise/quiz/test/practice or just having a conversation.
    
    TASK 2: If it is an exercise request, extract the parameters for generating the exercise.
    
    Return ONLY valid JSON with the following structure:
    {
      "is_exercise_request": true/false,
      "parameters": {
        "subject": "the subject area",
        "topic": "specific topic within the subject",
        "exercise_type": "multiple_choice",
        "difficulty": "easy/medium/hard/expert",
        "number_of_questions": integer between 1-10
      }
    }
    
    If it's not an exercise request, return:
    {
      "is_exercise_request": false
    }
    
    For exercise requests, infer the parameters even if not explicitly mentioned:
    - subject: The general field (math, science, history, etc.)
    - topic: The specific topic within the subject
    - exercise_type: "multiple_choice", "true_false", "code_challenge", "math", "fill_in_blank", "short_answer"
    - difficulty: "easy", "medium", "hard", "expert"
    - number_of_questions: A number from 1 to 10
    
    Examples of exercise requests:
    - "Give me some math problems"
    - "I need to practice calculus"
    - "Create a quiz about World War II"
    - "Test me on Python programming"
    - "I want to work on physics exercises"
    - "Help me practice French vocabulary"
    - "Can I get some questions about biology?"
    """
    
    user_prompt = f"Analyze this message:{message}{history_context}"
    
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
            
            # Validate and set defaults for parameters if this is an exercise request
            if result.get("is_exercise_request", False):
                params = result.get("parameters", {})
                result["parameters"] = {
                    "subject": params.get("subject", "general"),
                    "topic": params.get("topic", message),
                    "exercise_type": params.get("exercise_type", "multiple_choice"),
                    "difficulty": params.get("difficulty", "medium"),
                    "number_of_questions": int(params.get("number_of_questions", 3))
                }
            
            return result
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logger.error(f"Failed to parse JSON from LLM response: {json_str}")
    
    # Default response if extraction fails completely
    return {
        "is_exercise_request": False
    }