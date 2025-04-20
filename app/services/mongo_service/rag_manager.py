"""
Manages RAG (Retrieval Augmented Generation) operations with MongoDB
"""
import asyncio
from asyncio.log import logger
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
from fastapi import UploadFile, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from io import BytesIO
from bs4 import BeautifulSoup
# from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_mongodb import MongoDBAtlasVectorSearch

class RAGManager:
    """Handles RAG operations including document processing and vector search"""
    
    def __init__(self, db, collection, embeddings, lock):
        """Initialize with database, collection, embeddings, and lock"""
        self.db = db
        self.collection = collection
        self.embeddings = embeddings
        self.lock = lock
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # Initialize vector store
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name="default",
            text_key="text",
            embedding_key="embedding",
            relevance_score_fn="cosine",
        )
        
        logger.info("RAG Manager initialized")
    
    async def verify_index(self) -> bool:
        """Verify that the vector search index exists"""
        try:
            indexes = await self.collection.list_indexes()
            index_names = [index['name'] for index in await indexes.to_list(length=None)]
            return 'default' in index_names
        except Exception as e:
            logger.error(f"Error verifying index: {str(e)}")
            return False
    
    def clear_collection(self) -> None:
        """Clear the RAG collection"""
        with self.lock:
            logging.debug("Clearing RAG collection...")
            self.collection.delete_many({})
            logging.debug("Collection cleared.")
    
    async def process_file(self, file: UploadFile) -> List[str]:
        """Process uploaded file and return chunks of text"""
        content = await file.read()
        
        if file.filename.endswith('.pdf'):
            text = self._process_pdf(content)
        elif file.filename.endswith('.html'):
            text = self._process_html(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def _process_pdf(self, content: bytes) -> str:
        """Extract text from PDF file"""
        pdf = PdfReader(BytesIO(content))
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    
    def _process_html(self, content: bytes) -> str:
        """Extract text from HTML file"""
        soup = BeautifulSoup(content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    
    async def get_document_count(self) -> int:
        """Get the total number of documents in the RAG collection"""
        try:
            return await self.collection.count_documents({})
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0
    
    async def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Perform similarity search using MongoDB Atlas Vector Search"""
        try:
            # Generate query embedding
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.embeddings.embed_query, query
            )
            
            # Vector search pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "default",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,
                        "limit": k
                    }
                },
                {
                    "$project": {
                        "text": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"},
                        "_id": 0
                    }
                }
            ]
            
            cursor = self.collection.aggregate(pipeline)
            results = await cursor.to_list(length=k)
            
            if not results:
                logger.debug("No results found")
                count = await self.collection.count_documents({})
                logger.debug(f"Total documents in collection: {count}")
                sample = await self.collection.find_one({"embedding": {"$exists": True}})
                logger.debug(f"Sample document exists: {sample is not None}")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed with error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    async def add_texts_to_vectorstore(self, texts: List[str], metadata: Optional[dict] = None):
        """Add text chunks to vector store with verification"""
        try:
            logger.debug(f"Adding {len(texts)} texts to vector store")
            
            # Generate embeddings
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self.embeddings.embed_documents, texts
            )
            logger.debug(f"Generated {len(embeddings)} embeddings")

            # Prepare documents
            documents = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                doc = {
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata or {},
                    "chunk_id": i,
                    "timestamp": datetime.utcnow()
                }
                documents.append(doc)

            # Insert documents
            result = await self.collection.insert_many(documents)
            logger.debug(f"Inserted {len(result.inserted_ids)} documents")

            # Verify insertion
            inserted_count = await self.collection.count_documents(
                {"_id": {"$in": result.inserted_ids}}
            )
            logger.debug(f"Verified {inserted_count} documents inserted")

            return result.inserted_ids

        except Exception as e:
            logger.error(f"Failed to add texts: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to add texts to vector store: {str(e)}")
