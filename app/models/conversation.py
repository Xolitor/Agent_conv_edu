from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Union[Dict[str, Any], None]] = None
    
class Conversation(BaseModel):
    session_id: str
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Add this model to handle history responses
class MessageHistoryResponse(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Union[Dict[str, Any], None]] = None