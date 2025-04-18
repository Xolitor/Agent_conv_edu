from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router as api_router
from services.llm_serv import LLMService
import uvicorn
from services.mongo_services import MongoDBService
from models.teacher import initial_teachers

load_dotenv()

app = FastAPI(
    title="Agent conversationnel",
    description="API pour un agent conversationnel donné lors du TP1",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instance du service LLM
llm_service = LLMService()

# Inclure les routes
app.include_router(api_router)
# app.include_router(chat.router, prefix="/api")
# app.include_router(chat_claude.router, prefix="/api/v2")
# app.include_router(exercises.router, prefix="/api/exercises")

mongo_service = MongoDBService()
@app.on_event("startup")
async def startup_event():
    # Seed the teachers collection with initial data
    await mongo_service.seed_teachers(initial_teachers)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)