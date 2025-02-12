import asyncio
from asyncio.log import logger
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader
from core.config import settings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    BSHTMLLoader
)
from typing import Any, Dict, List
import os
import threading
import logging
from typing import List, Union, Optional
from pathlib import Path


from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from motor.motor_asyncio import AsyncIOMotorClient
from PyPDF2 import PdfReader
from io import BytesIO
from bs4 import BeautifulSoup
import threading
from typing import Optional
import hashlib

logging.basicConfig(level=logging.DEBUG)

class RAGServiceMongo:
    def __init__(self):
        """
        Initialize the RAG service with MongoDB as the backend.
        
        Args:
            mongo_uri: MongoDB connection URI.
            db_name: Name of the database. 
            collection_name: Name of the collection for storing vectors.
        """
        self.client = AsyncIOMotorClient(settings.mongodb_uri)
        self.db = self.client[settings.database_name]
        self.collection = self.db[settings.rag_database_name]
        
        # Ensure the collection has the required index for vector search

        # self.collection.create_index(
        #     [("vector", "2dsphere")],  # Use MongoDB's $vector search indexing
        #     name="vector_index"
        # )
        
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.lock = threading.Lock()  
        
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name="default",
            text_key="text",
            embedding_key="embedding",
            relevance_score_fn="cosine",
        )
        
    def clear(self) -> None:
        """
        Clears the MongoDB collection.
        """
        with self.lock:
            logging.debug("Clearing MongoDB collection...")
            self.collection.delete_many({})  # Deletes all documents
            logging.debug("Collection cleared.")
            
    # async def load_file(self, file_path: Union[str, Path], file_type: Optional[str] = None) -> List[str]:
    #     """
    #     Load and extract text from various file types
        
    #     Args:
    #         file_path: Path to the file
    #         file_type: Optional file type override ('pdf' or 'html')
            
    #     Returns:
    #         List of extracted text chunks
    #     """
    #     file_path = Path(file_path)
        
    #     if not file_type:
    #         file_type = file_path.suffix.lower()[1:]  # Remove the dot
            
    #     if file_type == 'pdf':
    #         loader = PyPDFLoader(str(file_path))
    #         documents = loader.load()
    #         return [doc.page_content for doc in documents]
            
    #     elif file_type in ['html', 'htm']:
    #         try:
    #             # Try BeautifulSoup loader first
    #             loader = BSHTMLLoader(str(file_path))
    #             documents = loader.load()
    #         except Exception as e:
    #             logging.warning(f"BSHTMLLoader failed, falling back to UnstructuredHTMLLoader: {e}")
    #             # Fallback to UnstructuredHTMLLoader
    #             loader = UnstructuredHTMLLoader(str(file_path))
    #             documents = loader.load()
    #         return [doc.page_content for doc in documents]
            
    #     else:
    #         raise ValueError(f"Unsupported file type: {file_type}")

    # async def process_files(self, 
    #                       file_paths: List[Union[str, Path]], 
    #                       clear_existing: bool = False) -> None:
    #     """
    #     Process multiple files and add them to the vector store
        
    #     Args:
    #         file_paths: List of paths to files
    #         clear_existing: If True, clear existing vector store before processing
    #     """
    #     all_texts = []
        
    #     for file_path in file_paths:
    #         try:
    #             texts = await self.load_file(file_path)
    #             all_texts.extend(texts)
    #         except Exception as e:
    #             logging.error(f"Error processing file {file_path}: {e}")
    #             continue
                
    #     await self.load_and_index_texts(all_texts, clear_existing)
    
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
    
    async def verify_index(self) -> bool:
        """Verify that the vector search index exists"""
        try:
            indexes = await self.collection.list_indexes()
            index_names = [index['name'] for index in await indexes.to_list(length=None)]
            return 'default' in index_names
        except Exception as e:
            logger.error(f"Error verifying index: {str(e)}")
            return False

    async def get_document_count(self) -> int:
        """Get the total number of documents in the collection"""
        return await self.collection.count_documents({})

    async def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Perform similarity search using MongoDB Atlas Vector Search"""
        try:
            # Generate query embedding
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.embeddings.embed_query, query
            )
            
            # Vector search pipeline using the correct syntax
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "default",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,  # Internal limit for consideration
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
                # Check if we have documents
                count = await self.collection.count_documents({})
                logger.debug(f"Total documents in collection: {count}")
                
                # Check if we can retrieve any document
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
    
    async def load_and_index_texts(self, texts: List[str], clear_existing: bool = False) -> None:
        """
        Loads and indexes a list of texts in MongoDB.
        
        Args:
            texts: List of texts to index.
            clear_existing: If True, clears the existing index before indexing.
        """
        with self.lock:
            if clear_existing:
                self.clear()
                
        # Split texts into chunks
        splits = self.text_splitter.split_text("\n\n".join(texts))

        # # Generate embeddings and insert them into MongoDB
        # documents = []
        # for text in splits:
        #     vector = self.embeddings.embed_query(text)
        #     documents.append({
        #         "content": text,
        #         "vector": vector  # Store embedding vector for similarity search
        #     })
        await self.vector_store.add_documents([{ "content": text } for text in splits])
        logging.debug(f"Inserted {len(splits)} documents into MongoDB.")
        # if documents:
        #     self.collection.insert_many(documents)
        #     logging.debug(f"Inserted {len(documents)} documents into MongoDB.")
        
    async def similarity_searchv1(self, query: str, k: int = 4) -> List[str]:
        """
        Performs similarity search using MongoDB.
        
        Args:
            query: Query string.
            k: Number of top results to return.
        
        Returns:
            List of the most relevant documents.
        """
        
        query_vector = self.embeddings.embed_query(query)
        
        # Perform a similarity search with MongoDB's $vector operator
        pipeline = [
            {
                "$search": {
                    "index": "vector_index",
                    "knnBeta": {
                        "vector": query_vector,
                        "path": "vector",
                        "k": k
                    }
                }
            },
            {
                "$project": {
                    "content": 1,
                    "_id": 0  # Exclude the MongoDB `_id` field from results
                }
            }
        ]           
        results = self.collection.aggregate(pipeline)
        return [doc["content"] for doc in results]
    
    async def close(self):
        """
        Close the MongoDB connection.
        """
        self.client.close()
        logging.debug("MongoDB connection closed.")