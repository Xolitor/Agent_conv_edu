from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentMetadata(BaseModel):
    """Metadata for a document stored in the vector database"""
    filename: Optional[str] = None
    file_id: Optional[str] = None
    upload_timestamp: Optional[datetime] = None
    source: Optional[str] = None
    chunk_id: Optional[int] = None
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    custom_metadata: Optional[Dict[str, Any]] = None

class DocumentChunk(BaseModel):
    """A chunk of text from a document with its metadata and embedding"""
    text: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    embedding: Optional[List[float]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class VectorSearchResult(BaseModel):
    """Result from a vector search operation"""
    text: str
    metadata: Optional[DocumentMetadata] = None
    score: float

class RAGQueryRequest(BaseModel):
    """Request model for querying the RAG system"""
    query: str
    session_id: Optional[str] = None
    include_chunks: bool = False
    top_k: int = 4
    threshold: Optional[float] = None

class RAGUploadRequest(BaseModel):
    """Request model for uploading documents to the RAG system"""
    texts: List[str]
    clear_existing: bool = False
    metadata: Optional[DocumentMetadata] = None

class RAGStatus(BaseModel):
    """Status information about the RAG system"""
    document_count: int
    has_valid_index: bool
    last_update: Optional[datetime] = None
    collection_name: str
