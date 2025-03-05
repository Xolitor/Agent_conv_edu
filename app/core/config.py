from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    mongodb_uri: str
    database_name: str = "chatbot"
    collection_name: str = "conversations"
    rag_database_name: str = "courses"
    teachers_database: str = "teachers"
    exercises_database: str = "exercises"
    
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
class Config:
    env_file = ".env"
    settings = Settings()

settings = Settings()
