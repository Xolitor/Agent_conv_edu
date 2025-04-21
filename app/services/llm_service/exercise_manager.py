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
from datetime import datetime  # Ensure this import is present
from models.exercise import ExerciseResponse, ExerciseType, ExerciseContent, Solution, EvaluationResult

class ExerciseManager:
    """
    Manages exercise generation, evaluation, and solution retrieval
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
                               exercise_type: ExerciseType,
                               difficulty: str,
                               number_of_questions: int,
                               session_id: Optional[str] = None,
                               teacher_id: Optional[str] = None) -> ExerciseResponse:
        """Generate exercises based on subject and parameters"""
        try:
            session = await self._ensure_session(session_id)
            
            # Craft a specialized system prompt for exercise generation
            exercise_system_prompt = f"""You are an expert educational exercise creator specialized in {subject}.
            Create {number_of_questions} {difficulty}-level {exercise_type.value} questions about {topic}.
            
            Follow these guidelines:
            1. Questions should be clear, precise, and appropriate for the {difficulty} difficulty level
            2. For multiple-choice questions, include 4 options with exactly one correct answer
            3. For math questions, use proper LaTeX formatting
            4. Include detailed explanations for the solution
            5. Return your response as structured data suitable for parsing
            6. If a {teacher_id} is given make sure that the subject of the teacher matches the {subject} of the exercise.
            
            Format your response in the following structure in JSON (this example is for multiple choice exercise):
            {{
              "exercise": {{
                "instructions": "Brief instructions for the exercise",
                "questions": [
                  {{ 
                    "question": "Question text",
                    "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                    "type": "{exercise_type.value}"
                  }}
                  // Additional questions...
                ]
              }},
              "solutions": {{
                "answers": [
                  {{ 
                    "correct_answer": "The correct answer or index", 
                    "correct_option": 2  // For multiple-choice questions
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
            
            Another example for a math exercise:
            {{
              "exercise": {{
                "instructions": "Solve the following math problems",
                "questions": [
                  {{ 
                    "question": "What is 2 + 2?",
                    "options": [],
                    "type": "{exercise_type.value}"
                  }}
                  // Additional questions...
                ]
              }},
              "solutions": {{
                "answers": [
                  {{ 
                    "correct_answer": 4, 
                    "correct_option": null  // No options for math questions
                  }}
                  // Additional answers...
                ],
                "explanations": [
                  "2 + 2 = 4",
                  // Additional explanations...
                ]
              }}
            }}
            """
            
            messages = []
            
            # Use the teacher's style if available
            if teacher_id:
                teacher_data = await self.mongo_service.get_teacher(teacher_id)
                if teacher_data:
                    # Combine teacher prompt with exercise creation instructions
                    combined_prompt = f"{teacher_data['prompt_instructions']}\n\n{exercise_system_prompt}"
                    messages.append(SystemMessage(content=combined_prompt))
                else:
                    messages.append(SystemMessage(content=exercise_system_prompt))
            else:
                messages.append(SystemMessage(content=exercise_system_prompt))
            
            # Add the exercise request as a message
            messages.append(HumanMessage(
                content=f"Please create {number_of_questions} {difficulty} level exercises about {topic} in {subject} using {exercise_type.value} format."
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
                    
                    # Create structured response
                    exercise_content = ExerciseContent(
                        questions=exercise_data["exercise"]["questions"],
                        instructions=exercise_data["exercise"]["instructions"]
                    )
                    
                    solutions = None
                    if "solutions" in exercise_data:
                        # Process answers to ensure correct types
                        answers = exercise_data["solutions"]["answers"]
                        for answer in answers:
                            # Convert any integer correct_option to string
                            if "correct_option" in answer and isinstance(answer["correct_option"], int):
                                answer["correct_option"] = str(answer["correct_option"])
                            
                            # Convert any integers in lists to strings if needed
                            for key, value in answer.items():
                                if isinstance(value, list):
                                    answer[key] = [str(item) if isinstance(item, int) else item for item in value]
                        
                        solutions = Solution(
                            answers=answers,
                            explanations=exercise_data["solutions"]["explanations"]
                        )

                    return ExerciseResponse(
                        exercise=exercise_content,
                        solutions=solutions
                    )
                except json.JSONDecodeError:
                    raise ValueError("Failed to parse exercise data from LLM response")
            else:
                raise ValueError("No valid JSON structure found in LLM response")
            
        except Exception as e:
            logger.error(f"Exercise generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def evaluate_answer(self,
                            exercise_id: str,
                            student_answer: str,
                            session_id: Optional[str] = None) -> EvaluationResult:
        """Evaluate a student's answer to an exercise"""
        try:
            # Retrieve the exercise and its solution from database
            exercise_data = await self.mongo_service.get_exercise(exercise_id)
            if not exercise_data:
                raise ValueError(f"Exercise with ID {exercise_id} not found")
            
            session = await self._ensure_session(session_id)
            
            evaluation_prompt = f"""You are an expert educational evaluator. 
            Evaluate the student's answer to the following question:
            
            Question: {exercise_data['question']}
            
            Correct answer: {exercise_data['correct_answer']}
            
            Student's answer: {student_answer}
            
            Provide an evaluation with:
            1. Whether the answer is correct (true/false)
            2. A score from 0.0 to 1.0
            3. Constructive feedback
            4. A detailed explanation of the correct answer
            
            Format your response as JSON:
            {{
              "is_correct": true/false,
              "score": 0.0-1.0,
              "feedback": "Your feedback here",
              "explanation": "Detailed explanation here"
            }}
            """
            
            messages = [
                SystemMessage(content=evaluation_prompt),
                HumanMessage(content=f"Please evaluate this answer: {student_answer}")
            ]
            
            # Generate evaluation
            response = await self.llm.agenerate([messages])
            response_text = response.generations[0][0].text
            
            # Parse JSON response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                try:
                    evaluation_data = json.loads(json_str)
                    return EvaluationResult(
                        is_correct=evaluation_data["is_correct"],
                        score=evaluation_data["score"],
                        feedback=evaluation_data["feedback"],
                        explanation=evaluation_data["explanation"]
                    )
                except json.JSONDecodeError:
                    raise ValueError("Failed to parse evaluation data")
            else:
                raise ValueError("No valid evaluation data found in response")
            
        except Exception as e:
            logger.error(f"Answer evaluation failed: {str(e)}")
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
                "created_at": datetime.utcnow()  # Make sure datetime is properly imported
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
                          session_id: Optional[str] = None) -> Solution:
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
            
            # Format solutions according to the model
            formatted_solutions = Solution(**solutions)
            
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
            
            return formatted_solutions
            
        except Exception as e:
            logger.error(f"Solutions retrieval failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
