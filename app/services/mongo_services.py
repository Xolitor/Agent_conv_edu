import os
import asyncio
from asyncio.log import logger
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from core.config import settings
import threading
import logging
from typing import Any, Dict, List, Optional
from fastapi import UploadFile, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from io import BytesIO
from bs4 import BeautifulSoup
from models.conversation import Conversation, Message
from models.teacher import Teacher
from pymongo import UpdateOne

logging.basicConfig(level=logging.DEBUG)

class MongoDBService:
    """
    Unified MongoDB service handling both conversation management and RAG functionality
    """
    def __init__(self):
        """Initialize the MongoDB service"""
        self.client = AsyncIOMotorClient(settings.mongodb_uri)
        self.db = self.client[settings.database_name]
        
        # Collection references
        self.conversations = self.db[settings.collection_name]
        self.teachers = self.db[settings.teachers_database]
        self.rag_collection = self.db[settings.rag_database_name]
        
        # RAG-specific setup
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.lock = threading.Lock()  
        
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.rag_collection,
            embedding=self.embeddings,
            index_name="default",
            text_key="text",
            embedding_key="embedding",
            relevance_score_fn="cosine",
        )
    
    #############################################
    # Connection and shared database operations #
    #############################################
    
    async def close(self):
        """Close the MongoDB connection"""
        self.client.close()
        logging.debug("MongoDB connection closed.")
    
    #######################################
    # Conversation management operations  #
    #######################################
    
    async def seed_teachers(self, teachers_data: list[Teacher]):
        """Populates the teachers collection with initial data if it is empty."""
        operations = []
        for teacher in teachers_data:
            operations.append(
                UpdateOne(
                    {"teacher_id": teacher.teacher_id},  
                    {"$set": teacher.model_dump()},      
                    upsert=True
                )
            )
        if operations:
            await self.teachers.bulk_write(operations)
    
    async def get_teacher(self, teacher_id: str) -> Optional[Dict]:
        """Get teacher by ID"""
        return await self.teachers.find_one({"teacher_id": teacher_id})
    
    async def save_message(self, session_id: str, role: str, content: str) -> bool:
        """Save a new message in a conversation"""
        message = Message(role=role, content=content)
        
        result = await self.conversations.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": message.model_dump()},
                "$set": {"updated_at": datetime.utcnow()},
                "$setOnInsert": {"created_at": datetime.utcnow()}
            },
            upsert=True
        )
        
        return result.modified_count > 0 or result.upserted_id is not None
    
    async def create_conversation(self, session_id: str) -> bool:
        """Create a new conversation"""
        conversation = {
            "session_id": session_id,
            "messages": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await self.conversations.insert_one(conversation)
        return result.inserted_id is not None
    
    async def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history"""
        conversation = await self.conversations.find_one({"session_id": session_id})
        if conversation:
            messages = conversation.get("messages", [])
            for message in messages:
                if "timestamp" in message and isinstance(message["timestamp"], datetime):
                    message["timestamp"] = message["timestamp"].isoformat()
            return messages
        return []
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation"""
        result = await self.conversations.delete_one({"session_id": session_id})
        return result.deleted_count > 0
    
    async def get_all_sessions(self) -> List[str]:
        """Get all session IDs sorted from newest to oldest"""
        cursor = self.conversations.find({}, {"session_id": 1}).sort("updated_at", -1)
        sessions = await cursor.to_list(length=None)
        return [session["session_id"] for session in sessions]
    
    ###################################
    # RAG Vector operations           #
    ###################################
    
    async def verify_index(self) -> bool:
        """Verify that the vector search index exists"""
        try:
            indexes = await self.rag_collection.list_indexes()
            index_names = [index['name'] for index in await indexes.to_list(length=None)]
            return 'default' in index_names
        except Exception as e:
            logger.error(f"Error verifying index: {str(e)}")
            return False
    
    def clear_rag_collection(self) -> None:
        """Clear the RAG collection"""
        with self.lock:
            logging.debug("Clearing RAG collection...")
            self.rag_collection.delete_many({})
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
            return await self.rag_collection.count_documents({})
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
            
            cursor = self.rag_collection.aggregate(pipeline)
            results = await cursor.to_list(length=k)
            
            if not results:
                logger.debug("No results found")
                count = await self.rag_collection.count_documents({})
                logger.debug(f"Total documents in collection: {count}")
                sample = await self.rag_collection.find_one({"embedding": {"$exists": True}})
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
            result = await self.rag_collection.insert_many(documents)
            logger.debug(f"Inserted {len(result.inserted_ids)} documents")

            # Verify insertion
            inserted_count = await self.rag_collection.count_documents(
                {"_id": {"$in": result.inserted_ids}}
            )
            logger.debug(f"Verified {inserted_count} documents inserted")

            return result.inserted_ids

        except Exception as e:
            logger.error(f"Failed to add texts: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to add texts to vector store: {str(e)}")
    
    async def save_exercise(self, 
                            subject: str,
                            topic: str,
                            exercise_data: dict,
                            teacher_id: Optional[str] = None) -> str:
        """Save an exercise to the database"""
        exercise = {
            "subject": subject,
            "topic": topic,
            "exercise_data": exercise_data,
            "teacher_id": teacher_id,
            "created_at": datetime.utcnow()
        }
        
        result = await self.db.exercises.insert_one(exercise)
        return str(result.inserted_id)

    async def get_exercise(self, exercise_id: str) -> Optional[Dict]:
        """Retrieve an exercise by ID"""
        from bson import ObjectId
        try:
            result = await self.db.exercises.find_one({"_id": ObjectId(exercise_id)})
            return result
        except Exception as e:
            logger.error(f"Failed to retrieve exercise: {str(e)}")
            return None

    async def get_exercises_by_subject(self, subject: str, limit: int = 10) -> List[Dict]:
        """Get exercises for a specific subject"""
        cursor = self.db.exercises.find({"subject": subject}).sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)
