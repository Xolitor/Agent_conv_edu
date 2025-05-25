"""
Exercise management component for LLM service
Handles exercise generation, evaluation, and solution retrieval
"""
from asyncio.log import logger
from fastapi import HTTPException
from langchain_core.messages import SystemMessage, HumanMessage
import json
import re
import traceback
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from bson import ObjectId

class ExerciseManager:
    """
    Manages exercise generation, evaluation, and solution retrieval using direct JSON
    """
    def __init__(self, llm, mongo_service):
        """Initialize with LLM and MongoDB service"""
        self.llm = llm
        self.mongo_service = mongo_service
        logger.info("Exercise Manager initialized")
    
    async def _ensure_session(self, session_id: Optional[str] = None) -> Dict:
        """Creates or retrieves a session context"""
        if not session_id:
            session_id = await self.mongo_service.create_conversation()
            return {"session_id": session_id, "history": []}
            
        history = await self.mongo_service.get_conversation_history(session_id)
        return {"session_id": session_id, "history": history}
    
    async def generate_exercise(self,
                              subject: str,
                              topic: str,
                              exercise_type: str,
                              difficulty: str,
                              number_of_questions: int,
                              session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate exercises based on subject and parameters - returns raw JSON"""
        try:
            session = await self._ensure_session(session_id)
            
            # Craft a specialized system prompt for exercise generation
            exercise_system_prompt = f"""You are an expert educational exercise creator specialized in {subject}.
            Create {number_of_questions} {difficulty}-level {exercise_type} questions about {topic}.
            
            Follow these guidelines:
            1. Questions should be clear, precise, and appropriate for the {difficulty} difficulty level
            2. For multiple-choice questions, include 4 options with exactly one correct answer
            3. For math questions, use proper LaTeX formatting
            4. Include detailed explanations for the solution
            5. Return your response as structured data suitable for parsing
            
            Format your response in the following structure in JSON (this example is for multiple choice exercise):
            {{
              "exercise": {{
                "instructions": "Brief instructions for the exercise",
                "questions": [
                  {{ 
                    "question": "Question text",
                    "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                    "type": "{exercise_type}"
                  }}
                  // Additional questions...
                ]
              }},
              "solutions": {{
                "answers": [
                  {{ 
                    "correct_answer": "The correct answer or index", 
                    "correct_option": "2"  // For multiple-choice questions, use string format
                  }}
                  // Additional answers...
                ],
                "explanations": [
                  "Detailed explanation for question 1",
                  // Additional explanations...
                ]
              }}
            }}
            
            Note: Ensure that the JSON is valid and well-structured.
            """
            
            messages = []
            messages.append(SystemMessage(content=exercise_system_prompt))
            
            # Add the exercise request as a message
            messages.append(HumanMessage(
                content=f"Please create {number_of_questions} {difficulty} level exercises about {topic} in {subject} using {exercise_type} format."
            ))
            
            # Generate response
            response = await self.llm.agenerate([messages])
            response_text = response.generations[0][0].text
            
            # Extract JSON from the response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                try:
                    exercise_data = json.loads(json_str)
                    
                    # Process the JSON to ensure consistent types
                    # Convert any integer correct_option to string for consistency
                    if "solutions" in exercise_data and "answers" in exercise_data["solutions"]:
                        for answer in exercise_data["solutions"]["answers"]:
                            if isinstance(answer, dict) and "correct_option" in answer:
                                if isinstance(answer["correct_option"], int):
                                    answer["correct_option"] = str(answer["correct_option"])
                    
                    # Return the processed JSON directly
                    return exercise_data
                    
                except json.JSONDecodeError:
                    raise ValueError("Failed to parse exercise data from LLM response")
            else:
                raise ValueError("No valid JSON structure found in LLM response")
            
        except Exception as e:
            logger.error(f"Exercise generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def evaluate_exercise(self,
                              exercise_id: str,
                              user_answers: List[Dict[str, Any]],
                              session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate user answers for a previously generated exercise
        """
        try:
            # Retrieve the exercise with solutions from MongoDB
            exercise = await self.mongo_service.exercises.find_one({"_id": ObjectId(exercise_id)})
            
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
            
            response = await self.llm.agenerate([messages])
            response_text = response.generations[0][0].text
            
            # Extract JSON from response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                try:
                    result = json.loads(json_str)
                    
                    # Store the evaluation result in MongoDB for reference
                    evaluation_id = await self.mongo_service.db.exercise_evaluations.insert_one({
                        "exercise_id": exercise_id,
                        "user_answers": user_answers,
                        "evaluation": result,
                        "session_id": session_id,
                        "created_at": datetime.utcnow()
                    })
                    
                    # Add the evaluation ID to the result
                    result["_id"] = str(evaluation_id.inserted_id)
                    
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
    
    async def generate_hint(self,
                           exercise_id: str,
                           question_number: Optional[int] = None,
                           session_id: Optional[str] = None) -> str:
        """Generate a helpful hint for a specific exercise question"""
        try:
            # Retrieve the exercise and its solutions from database
            exercise_data = await self.mongo_service.get_exercise(exercise_id)
            if not exercise_data:
                raise ValueError(f"Exercise with ID {exercise_id} not found")
            
            session = await self._ensure_session(session_id)
            
            # Extract the specific question or the entire exercise
            exercise_content = exercise_data.get("exercise", {})
            solutions = exercise_data.get("solutions", {})
            
            question_content = None
            solution_content = None
            
            if question_number is not None and "questions" in exercise_content:
                # Get specific question
                if 0 <= question_number - 1 < len(exercise_content["questions"]):
                    question_content = exercise_content["questions"][question_number - 1]
                    
                    # Get corresponding solution if available
                    if "answers" in solutions and 0 <= question_number - 1 < len(solutions["answers"]):
                        solution_content = solutions["answers"][question_number - 1]
                        
                        # Add explanation if available
                        if "explanations" in solutions and 0 <= question_number - 1 < len(solutions["explanations"]):
                            solution_content["explanation"] = solutions["explanations"][question_number - 1]
            else:
                # Use entire exercise
                question_content = exercise_content
                solution_content = solutions
            
            # Create the prompt for hint generation
            hint_system_prompt = """You are a supportive educational assistant.
            
            TASK: Generate a helpful hint for the student without revealing the complete solution.
            
            Guidelines:
            1. Provide guidance that helps the student think about the problem
            2. Do NOT reveal the full solution
            3. Be encouraging and supportive
            4. Focus only on the specific question(s) asked
            5. Offer a step or concept that leads toward the solution
            6. For math problems, consider suggesting formulas or approaches
            """
            
            hint_user_prompt = f"""Exercise question: 
            {json.dumps(question_content)}
            
            Solution information (use this to create your hint, NOT to give away the answer):
            {json.dumps(solution_content)}
            
            Please provide a helpful hint for question {question_number if question_number else "this exercise"}.
            """
            
            messages = [
                SystemMessage(content=hint_system_prompt),
                HumanMessage(content=hint_user_prompt)
            ]
            
            # Generate hint
            response = await self.llm.agenerate([messages])
            hint = response.generations[0][0].text
            
            # Store the hint in the database for future reference
            hint_data = {
                "exercise_id": exercise_id,
                "question_number": question_number,
                "hint": hint,
                "session_id": session["session_id"],
                "created_at": datetime.utcnow()
            }
            await self.mongo_service.save_hint(hint_data)
            
            return hint
            
        except Exception as e:
            # Enhanced error logging
            stack_trace = traceback.format_exc()
            logger.error(f"Hint generation failed: {str(e)}\nStack trace: {stack_trace}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_solutions(self,
                          exercise_id: str,
                          session_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve and format solutions for an exercise"""
        try:
            # Retrieve the exercise and its solutions from database
            exercise_data = await self.mongo_service.get_exercise(exercise_id)
            if not exercise_data:
                raise ValueError(f"Exercise with ID {exercise_id} not found")
            
            # Extract solutions section
            solutions = exercise_data.get("solutions", {})
            if not solutions:
                raise ValueError("No solutions available for this exercise")
            
            # If session is provided, save a record of solutions being viewed
            if session_id:
                session = await self._ensure_session(session_id)
                
                # Format the solution as a friendly message for the chat history
                response_text = "Solutions for all questions:\n\n"
                
                for i, answer in enumerate(solutions.get("answers", [])):
                    response_text += f"Question {i+1}:\n"
                    if isinstance(answer, dict):
                        if "correct_option" in answer:
                            response_text += f"Correct option: {answer['correct_option']}\n"
                        if "correct_answer" in answer:
                            response_text += f"Correct answer: {answer['correct_answer']}\n"
                    else:
                        response_text += f"{answer}\n"
                    
                    # Add explanation if available
                    if "explanations" in solutions and i < len(solutions["explanations"]):
                        response_text += f"Explanation: {solutions['explanations'][i]}\n"
                    
                    response_text += "\n"
                
                # Save to conversation history
                await self.mongo_service.save_message(
                    session["session_id"],
                    "assistant",
                    response_text,
                    metadata={"type": "solution", "exercise_id": exercise_id}
                )
            
            return solutions
            
        except Exception as e:
            logger.error(f"Solutions retrieval failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
