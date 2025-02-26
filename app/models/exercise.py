
from enum import Enum
from typing import List, Optional, Dict, Union
from pydantic import BaseModel

class ExerciseType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    FILL_IN_BLANK = "fill_in_blank"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    CODE = "code"
    MATH = "math"

class ExerciseRequest(BaseModel):
    subject: str
    topic: str
    exercise_type: ExerciseType = ExerciseType.MULTIPLE_CHOICE
    include_solutions: bool = False
    session_id: Optional[str] = None
    teacher_id: Optional[str] = None

class ExerciseContent(BaseModel):
    questions: List[Dict[str, Union[str, List[str]]]]
    instructions: str

class Solution(BaseModel):
    answers: List[Dict[str, Union[str, List[str]]]]
    explanations: List[str]

class ExerciseResponse(BaseModel):
    exercise: ExerciseContent
    solutions: Optional[Solution] = None

class EvaluationResult(BaseModel):
    is_correct: bool
    feedback: str
    score: float
    explanation: str
