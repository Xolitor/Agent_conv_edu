
import os
import sys
import pytest
import httpx
import asyncio
import uuid
from fastapi.testclient import TestClient
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from  main import app
from app.services.llm_serv import LLMService

client = TestClient(app)

# Helper functions
def generate_test_session_id():
    """Generate a unique session ID for testing"""
    return f"test_session_{uuid.uuid4().hex[:8]}"

# Test data
TEST_PDF_PATH = Path(__file__).parent / "data" / "sample.pdf"
TEST_QUESTION = "Explain what the document is about"
TEST_TEACHER_ID = "math_teacher"

@pytest.fixture(scope="module", autouse=True)
def create_test_pdf():
    """Create a sample PDF file for testing if it doesn't exist"""
    test_dir = Path(__file__).parent / "data"
    test_dir.mkdir(exist_ok=True)
    
    if not TEST_PDF_PATH.exists():
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        c = canvas.Canvas(str(TEST_PDF_PATH), pagesize=letter)
        c.drawString(100, 750, "This is a test PDF document for RAG testing")
        c.drawString(100, 730, "It contains information about machine learning and Python")
        c.drawString(100, 710, "Python is a popular programming language for AI applications")
        c.drawString(100, 690, "Machine learning models can be trained using various algorithms")
        c.save()

# Basic endpoint tests
def test_health_check():
    """Test that the API is running"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"

# Main functionality tests
def test_chat_endpoint():
    """Test the basic chat endpoint"""
    session_id = generate_test_session_id()
    
    response = client.post("/chat", json={
        "message": "Hello, how are you?",
        "session_id": session_id
    })
    
    assert response.status_code == 200
    assert "response" in response.json()
    assert isinstance(response.json()["response"], str)
    assert len(response.json()["response"]) > 0

def test_chat_history_flow():
    """Test that session history works correctly"""
    session_id = generate_test_session_id()
    
    # First message
    response1 = client.post("/chat", json={
        "message": "My name is Test User",
        "session_id": session_id
    })
    assert response1.status_code == 200
    
    # Second message referencing the first
    response2 = client.post("/chat", json={
        "message": "What's my name?",
        "session_id": session_id
    })
    assert response2.status_code == 200
    
    # The response should contain "Test User"
    assert "Test User" in response2.json()["response"]
    
    # Test history retrieval
    history_response = client.get(f"/history/{session_id}")
    assert history_response.status_code == 200
    assert len(history_response.json()) == 4  # 2 user messages + 2 assistant responses

def test_teacher_chat_endpoint():
    """Test the teacher-specific chat endpoint"""
    session_id = generate_test_session_id()
    
    response = client.post(f"/teacher-chat?teacher_id={TEST_TEACHER_ID}", json={
        "message": "Explain what a quadratic equation is",
        "session_id": session_id
    })
    
    assert response.status_code == 200
    assert "response" in response.json()
    assert isinstance(response.json()["response"], str)
    assert len(response.json()["response"]) > 0

@pytest.mark.asyncio
async def test_rag_flow():
    """Test the complete RAG flow: upload file, query, then ask questions about it"""
    session_id = generate_test_session_id()
    
    # Step 1: Upload PDF
    with open(TEST_PDF_PATH, "rb") as pdf_file:
        files = {"files": (TEST_PDF_PATH.name, pdf_file, "application/pdf")}
        upload_response = client.post("/uploadv2", files=files)
    
    assert upload_response.status_code == 200
    assert "processed_files" in upload_response.json()
    assert upload_response.json()["processed_files"][0]["status"] == "success"
    
    # Step 2: Query the uploaded content
    query_response = client.post("/query", params={"query": TEST_QUESTION, "session_id": session_id})
    
    assert query_response.status_code == 200
    assert "answer" in query_response.json()
    assert len(query_response.json()["answer"]) > 0
    
    # Check if the response mentions content from our test PDF
    answer = query_response.json()["answer"].lower()
    assert any(word in answer for word in ["python", "machine learning", "test", "pdf"])
    
    # Step 3: Test RAG chat to ensure context is maintained
    rag_chat_response = client.post("/rag", json={
        "message": "Tell me more about the machine learning part",
        "session_id": session_id
    })
    
    assert rag_chat_response.status_code == 200
    assert "response" in rag_chat_response.json()

# Error handling tests
def test_invalid_teacher_id():
    """Test error handling for invalid teacher IDs"""
    response = client.post(f"/teacher-chat?teacher_id=nonexistent_teacher", json={
        "message": "Hello",
        "session_id": generate_test_session_id()
    })
    
    assert response.status_code == 500  # Should return error

def test_query_without_uploads():
    """Test query behavior when no documents are uploaded"""
    # Clear any existing documents
    client.delete("/delete/all/documents")
    
    response = client.post("/query", params={"query": "Tell me about something"})
    
    assert response.status_code == 200
    assert "Je ne trouve pas d'informations pertinentes" in response.json()["answer"]

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
