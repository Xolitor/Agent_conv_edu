import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app
from services.rag_mongo_services import RAGServiceMongo
from motor.motor_asyncio import AsyncIOMotorClient
import json

client = TestClient(app)

@pytest.fixture
async def mock_rag_service():
    """Fixture to create a mock RAG service"""
    with patch('services.llm_service.RAGServiceMongo') as mock:
        yield mock

@pytest.fixture
def test_documents():
    """Fixture providing test documents"""
    return [
        "This is the first test document about Python programming.",
        "This is the second test document about data structures.",
        "This is the third test document about algorithms."
    ]

@pytest.mark.asyncio
async def test_index_documents_success(mock_rag_service, test_documents):
    """Test successful document indexing"""
    # Setup
    mock_instance = mock_rag_service.return_value
    mock_instance.load_and_index_texts.return_value = None
    
    # Test data
    request_data = {
        "texts": test_documents,
        "clear_existing": False
    }
    
    # Make request
    response = client.post("/documents", json=request_data)
    
    # Assertions
    assert response.status_code == 200
    assert response.json() == {"message": "Documents indexed successfully"}
    
    # Verify the RAG service was called correctly
    mock_instance.load_and_index_texts.assert_called_once_with(
        test_documents,
        False
    )

@pytest.mark.asyncio
async def test_index_documents_with_clear(mock_rag_service, test_documents):
    """Test document indexing with clearing existing documents"""
    # Setup
    mock_instance = mock_rag_service.return_value
    mock_instance.load_and_index_texts.return_value = None
    
    # Test data
    request_data = {
        "texts": test_documents,
        "clear_existing": True
    }
    
    # Make request
    response = client.post("/documents", json=request_data)
    
    # Assertions
    assert response.status_code == 200
    assert response.json() == {"message": "Documents indexed successfully"}
    
    # Verify the RAG service was called with clear_existing=True
    mock_instance.load_and_index_texts.assert_called_once_with(
        test_documents,
        True
    )

@pytest.mark.asyncio
async def test_index_documents_empty_list(mock_rag_service):
    """Test indexing with empty document list"""
    # Test data
    request_data = {
        "texts": [],
        "clear_existing": False
    }
    
    # Make request
    response = client.post("/documents", json=request_data)
    
    # Assertions
    assert response.status_code == 200
    assert response.json() == {"message": "Documents indexed successfully"}

@pytest.mark.asyncio
async def test_index_documents_error(mock_rag_service, test_documents):
    """Test error handling during document indexing"""
    # Setup mock to raise an exception
    mock_instance = mock_rag_service.return_value
    mock_instance.load_and_index_texts.side_effect = Exception("Test error")
    
    # Test data
    request_data = {
        "texts": test_documents,
        "clear_existing": False
    }
    
    # Make request
    response = client.post("/documents", json=request_data)
    
    # Assertions
    assert response.status_code == 500
    assert "detail" in response.json()
    assert "Test error" in response.json()["detail"]

@pytest.mark.asyncio
async def test_index_documents_invalid_input():
    """Test indexing with invalid input"""
    # Test data with invalid type
    request_data = {
        "texts": "not a list",  # Should be a list
        "clear_existing": False
    }
    
    # Make request
    response = client.post("/documents", json=request_data)
    
    # Assertions
    assert response.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_index_documents_large_content(mock_rag_service):
    """Test indexing with large document content"""
    # Create a large document
    large_document = ["A" * 1000000]  # 1MB of text
    
    # Test data
    request_data = {
        "texts": large_document,
        "clear_existing": False
    }
    
    # Make request
    response = client.post("/documents", json=request_data)
    
    # Assertions
    assert response.status_code == 200
    assert response.json() == {"message": "Documents indexed successfully"}