from fastapi import APIRouter, HTTPException, Body
from models.chat import ChatRequest , ChatResponse, ChatRequestWithCourseData
from services.llm_service import LLMService
from typing import Dict, List

router = APIRouter()