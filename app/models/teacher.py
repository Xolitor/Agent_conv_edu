from typing import Dict, List, Optional
from pydantic import BaseModel

class Teacher(BaseModel) :
    teacher_id: str
    name: str
    subject: str
    description : str
    prompt_instructions: Optional[str] = None