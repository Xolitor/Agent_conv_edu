"""
Exercise management component for LLM service
Handles exercise generation, evaluation, and solution retrieval
"""
from asyncio.log import logger
from fastapi import HTTPException
from langchain_core.messages import SystemMessage, HumanMessage
import json
import re
from typing import Dict, List, Optional, Any, Union
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
