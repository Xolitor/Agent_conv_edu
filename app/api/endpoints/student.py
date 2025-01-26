from fastapi import APIRouter, HTTPException, Body
<<<<<<< HEAD
from models.chat import ChatRequest , ChatResponse, ChatRequestWithCourseData
=======
from models.chat import ChatRequest, ChatResponse, ChatRequestWithCourseData
>>>>>>> 649c245770c686a162ed8a7240ae3834783a3bfe
from services.llm_service import LLMService
from typing import Dict, List

router = APIRouter()