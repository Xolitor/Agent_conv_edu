from fastapi import APIRouter, HTTPException, Body
from models.chat import ChatRequest , ChatResponse
from services.llm_service import LLMService
from typing import Dict, List

router = APIRouter()