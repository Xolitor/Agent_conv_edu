import pytest
from fastapi.testclient import TestClient
from services.mongo_service import MongoService 
from main import app

client = TestClient(app)

@pytest.mark.asyncio 
async def test_get_conversation_history():
    mongo_service = MongoService()

    test_session_id = "string"

    history = await mongo_service.get_conversation_history(test_session_id)

    assert isinstance(history, list), "History should be a list"
    if history:
        assert all("role" in msg and "content" in msg for msg in history), "Each message should have 'role' and 'content' keys"

