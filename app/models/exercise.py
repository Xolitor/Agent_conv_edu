from enum import Enum
from typing import List, Optional, Dict, Union, Any
from pydantic import BaseModel, Field

class ExerciseType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    FILL_IN_BLANK = "fill_in_blank"
    SHORT_ANSWER = "short_answer"
    CODE_CHALLENGE = "code_challenge"
    TRUE_OR_FALSE = "true_false"

class ExerciseRequest(BaseModel):
    subject: str
    topic: str
    exercise_type: ExerciseType = ExerciseType.MULTIPLE_CHOICE
    include_solutions: bool = False
    session_id: Optional[str] = None
    teacher_id: Optional[str] = None

class ExerciseContent(BaseModel):
    questions: List[Dict[str, Any]]
    instructions: str

class Answer(BaseModel):
    # Make all fields optional to support different exercise types
    correct_option: Optional[Union[str, List[str]]] = Field(default="")
    answer: Optional[Union[str, List[str]]] = Field(default="")
    solution: Optional[str] = Field(default="")
    code: Optional[str] = Field(default="")
    
    # Add a model_config to allow extra fields for flexibility
    model_config = {
        "extra": "allow"
    }

class Solution(BaseModel):
    answers: List[Union[Dict[str, Any], Answer]]
    explanations: List[str]

class ExerciseResponse(BaseModel):
    exercise: ExerciseContent
    solutions: Optional[Solution] = None

class EvaluationResult(BaseModel):
    is_correct: bool
    feedback: str
    score: float
    explanation: str