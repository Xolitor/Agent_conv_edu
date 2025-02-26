
import os
import sys
import asyncio
import uuid
import httpx
from pathlib import Path

def log_success(message):
    print(f"\033[92m✓ {message}\033[0m")  # Green text

def log_error(message):
    print(f"\033[91m✗ {message}\033[0m")  # Red text

def log_info(message):
    print(f"\033[94m→ {message}\033[0m")  # Blue text

async def test_endpoints():
    """Quick test of critical endpoints"""
    BASE_URL = "http://127.0.0.1:8000"
    session_id = f"quicktest_{uuid.uuid4().hex[:8]}"
    
    log_info("Starting quick test of critical endpoints...")
    
    # Test 1: Basic chat endpoint
    log_info("Testing basic chat endpoint...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{BASE_URL}/chat", json={
                "message": "Hello, how are you?",
                "session_id": session_id
            })
            response.raise_for_status()
            log_success(f"Chat endpoint working: Got response of length {len(response.json().get('response', ''))}")
        except Exception as e:
            log_error(f"Chat endpoint failed: {str(e)}")
    
    # Test 2: Teacher chat endpoint
    log_info("Testing teacher chat endpoint...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{BASE_URL}/teacher-chat?teacher_id=math_teacher", json={
                "message": "Explain what a quadratic equation is",
                "session_id": session_id
            })
            response.raise_for_status()
            log_success(f"Teacher chat endpoint working: Got response of length {len(response.json().get('response', ''))}")
        except Exception as e:
            log_error(f"Teacher chat endpoint failed: {str(e)}")
    
    # Test 3: Upload a test PDF
    log_info("Testing file upload endpoint...")
    test_pdf_path = create_test_pdf()
    
    async with httpx.AsyncClient() as client:
        try:
            with open(test_pdf_path, "rb") as pdf_file:
                files = {"files": (os.path.basename(test_pdf_path), pdf_file, "application/pdf")}
                response = await client.post(f"{BASE_URL}/uploadv2", files=files)
            response.raise_for_status()
            log_success("File upload endpoint working")
        except Exception as e:
            log_error(f"File upload endpoint failed: {str(e)}")
    
    # Test 4: Query documents
    log_info("Testing query documents endpoint...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{BASE_URL}/query", params={
                "query": "What is this document about?",
                "session_id": session_id
            })
            response.raise_for_status()
            log_success(f"Query endpoint working: Got response of length {len(response.json().get('answer', ''))}")
        except Exception as e:
            log_error(f"Query endpoint failed: {str(e)}")
    
    log_info("Quick test completed!")

def create_test_pdf():
    """Create a sample PDF file for testing"""
    test_dir = Path("tests/data")
    test_dir.mkdir(exist_ok=True, parents=True)
    
    pdf_path = test_dir / "sample.pdf"
    
    if not pdf_path.exists():
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.drawString(100, 750, "This is a test PDF document for RAG testing")
            c.drawString(100, 730, "It contains information about machine learning and Python")
            c.drawString(100, 710, "Python is a popular programming language for AI applications")
            c.drawString(100, 690, "Machine learning models can be trained using various algorithms")
            c.save()
            log_success(f"Created test PDF at {pdf_path}")
        except Exception as e:
            log_error(f"Failed to create test PDF: {str(e)}")
            return None
    
    return pdf_path

if __name__ == "__main__":
    asyncio.run(test_endpoints())
