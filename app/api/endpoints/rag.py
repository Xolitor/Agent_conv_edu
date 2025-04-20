"""
Endpoints for RAG (Retrieval Augmented Generation) functionality
"""
from datetime import datetime
from fastapi import APIRouter, HTTPException, Body, UploadFile, File
from models.chat import ChatRequest, ChatResponse
from services.llm_service import LLMService
from services.mongo_service import MongoDBService
from typing import Dict, List, Optional
import hashlib
from asyncio.log import logger

router = APIRouter()
llm_service = LLMService()
mongo_service = MongoDBService()

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload and process files for RAG
    """
    processed_files = []
    
    for file in files:
        try:
            # Process file content
            chunks = await mongo_service.process_file(file)
            
            # Create metadata
            metadata = {
                "filename": file.filename,
                "file_id": hashlib.md5(file.filename.encode()).hexdigest(),
                "upload_timestamp": datetime.now().isoformat()
            }
            
            # Add to vector store
            await mongo_service.add_texts_to_vectorstore(chunks, metadata)
            
            processed_files.append({
                "filename": file.filename,
                "status": "success",
                "chunks": len(chunks)
            })
            
        except Exception as e:
            processed_files.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"processed_files": processed_files}   

# @router.post("/index")
# async def index_documents(
#     texts: List[str] = Body(...),
#     clear_existing: bool = Body(False)
# ) -> dict:
#     """
#     Index documents for RAG
#     """
#     try:
#         await llm_service.rag.load_and_index_texts(texts, clear_existing)
#         return {"message": "Documents indexed successfully"}
#     except Exception as e:
#         logger.error(f"Document indexing error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents")
async def clear_documents() -> dict:
    """
    Clear all indexed documents
    """
    try:
        mongo_service.close()
        mongo_service.clear()
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        logger.error(f"Clear documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_documents(
    query: str,
    session_id: Optional[str] = None,
    include_chunks: bool = False
):
    """
    Query documents and get contextual answers
    """
    try:
        # Get similar chunks
        chunks = await mongo_service.similarity_search(query)
        
        if not chunks:
            return {
                "answer": "Je ne trouve pas d'informations pertinentes pour répondre à votre question.",
                "chunks": []
            }
        
        # Generate answer using the unified response generator
        answer = await llm_service.generate_response(
            message=query,
            session_id=session_id,
            use_rag=True
        )
        
        response = {
            "answer": answer,
            "metadata": {
                "query": query,
                "num_chunks_used": len(chunks),
                "session_id": session_id
            }
        }
        
        if include_chunks:
            response["chunks"] = chunks
        
        return response
        
    except Exception as e:
        logger.error(f"Query documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug")
async def debug_collection():
    """
    Debug endpoint to inspect RAG collection
    """
    try:
        sample_doc = await mongo_service.collection.find_one(
            {}, 
            {'_id': 0}  # Exclude _id field from the result
        )
        doc_count = await mongo_service.get_document_count()
        
        return {
            "sample_document": sample_doc,
            "document_count": doc_count
        }
    except Exception as e:
        logger.error(f"Debug collection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
