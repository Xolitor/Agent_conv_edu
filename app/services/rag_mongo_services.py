from motor.motor_asyncio import AsyncIOMotorClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from core.config import settings
from typing import List
import os
import threading
import logging

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
        self.collection.create_index(
            [("vector", "2dsphere")],  # Use MongoDB's $vector search indexing
            name="vector_index"
        )
        
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.lock = threading.Lock()

    def clear(self) -> None:
        """
        Clears the MongoDB collection.
        """
        with self.lock:
            logging.debug("Clearing MongoDB collection...")
            self.collection.delete_many({})  # Deletes all documents
            logging.debug("Collection cleared.")
    
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
        
        # Generate embeddings and insert them into MongoDB
        documents = []
        for text in splits:
            vector = self.embeddings.embed_query(text)
            documents.append({
                "content": text,
                "vector": vector  # Store embedding vector for similarity search
            })
        
        if documents:
            self.collection.insert_many(documents)
            logging.debug(f"Inserted {len(documents)} documents into MongoDB.")

    async def similarity_search(self, query: str, k: int = 4) -> List[str]:
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
    
    def close(self):
        """
        Close the MongoDB connection.
        """
        self.mongo_client.close()
        logging.debug("MongoDB connection closed.")
