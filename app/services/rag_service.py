from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    BSHTMLLoader
)
from typing import List, Union, Optional
import os
import shutil
import threading
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

class RAGService:
    def __init__(self, persist_dir: str = "./data/vectorstore"):
        """
        Initialise le service RAG avec un vector store persistant
        
        Args:
            persist_dir: Chemin où persister le vector store
        """
        self.persist_dir = persist_dir
        
        # Création du dossier de persistance s'il n'existe pas
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
             chunk_size=500,  # Reduced chunk size for better processing
            chunk_overlap=50,
            length_function=len,  # added for pdf 
            is_separator_regex=False # added for pdf    
        )
        
        self.lock = threading.Lock()
        # Chargement d'un vector store existant ou création d'un nouveau
        if os.path.exists(os.path.join(self.persist_dir, "chroma.sqlite3")):
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            self.vector_store = None
            
    async def load_file(self, file_path: Union[str, Path], file_type: Optional[str] = None) -> List[str]:
        """
        Load and extract text from various file types
        
        Args:
            file_path: Path to the file
            file_type: Optional file type override ('pdf' or 'html')
            
        Returns:
            List of extracted text chunks
        """
        file_path = Path(file_path)
        
        if not file_type:
            file_type = file_path.suffix.lower()[1:]  # Remove the dot
            
        if file_type == 'pdf':
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            return [doc.page_content for doc in documents]
            
        elif file_type in ['html', 'htm']:
            try:
                # Try BeautifulSoup loader first
                loader = BSHTMLLoader(str(file_path))
                documents = loader.load()
            except Exception as e:
                logging.warning(f"BSHTMLLoader failed, falling back to UnstructuredHTMLLoader: {e}")
                # Fallback to UnstructuredHTMLLoader
                loader = UnstructuredHTMLLoader(str(file_path))
                documents = loader.load()
            return [doc.page_content for doc in documents]
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    async def process_files(self, 
                          file_paths: List[Union[str, Path]], 
                          clear_existing: bool = False) -> None:
        """
        Process multiple files and add them to the vector store
        
        Args:
            file_paths: List of paths to files
            clear_existing: If True, clear existing vector store before processing
        """
        all_texts = []
        
        for file_path in file_paths:
            try:
                texts = await self.load_file(file_path)
                all_texts.extend(texts)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                continue
                
        await self.load_and_index_texts(all_texts, clear_existing)
            
    async def load_and_index_texts(self, texts: List[str], clear_existing: bool = False) -> None:
        """
        Charge et indexe une liste de textes
        
        Args:
            texts: Liste de textes à indexer
            clear_existing: Si True, supprime l'index existant avant d'indexer
        """
        # Nettoyage optionnel de l'existant
        if clear_existing and os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            os.makedirs(self.persist_dir)
            self.vector_store = None
            
        # Découpage des textes
        splits = self.text_splitter.split_text("\n\n".join(texts))
        
        if self.vector_store is None:
            # Création initiale
            self.vector_store = Chroma.from_texts(
                splits,
                self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            # Ajout à l'existant
            self.vector_store.add_texts(splits)
            
        # Persistance explicite
        self.vector_store.persist()


    async def load_and_index_textsv2(self, texts: List[str], clear_existing: bool = False) -> None:
        """
        Charge et indexe une liste de textes
        
        Args:
            texts: Liste de textes à indexer
            clear_existing: Si True, supprime l'index existant avant d'indexer
        """
        # Nettoyage optionnel de l'existant
        with self.lock:
            if clear_existing and os.path.exists(self.persist_dir):
                self.clear()
            # if self.vector_store:
            #     self.vector_store._client = None  # Release the SQLite client
            #     self.vector_store = None
            # shutil.rmtree(self.persist_dir)
            # os.makedirs(self.persist_dir)
            
            # self.vector_store._client = None
            # self.vector_store = None
            
        # Découpage des textes
            splits = self.text_splitter.split_text("\n\n".join(texts))
            
            if self.vector_store is None:
                # Création initiale
                self.vector_store = Chroma.from_texts(
                    splits,
                    self.embeddings,
                    persist_directory=self.persist_dir
                )
            else:
                # Ajout à l'existant
                self.vector_store.add_texts(splits)
                
            # Persistance explicite
            self.vector_store.persist()
    
    async def similarity_search(self, query: str, k: int = 4) -> List[str]:
        """
        Effectue une recherche par similarité
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats à retourner
            
        Returns:
            Liste des documents les plus pertinents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please add documents first.")
            
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    
    def close(self):
        """Release resources and close the vector store."""
        if self.vector_store:
            logging.debug("Closing vector store...")
            self.vector_store._client = None  # Close SQLite client
            self.vector_store = None
            logging.debug("Vector store closed.")
            
    def clear(self) -> None:
        """
        Supprime toutes les données du vector store
        """
        with self.lock:
            self.close()
            # if self.vector_store:
            #     self.vector_store._client.close()
            #     self.vector_store._client = None  # Release the SQLite client
            #     self.vector_store = None
            
            if os.path.exists(self.persist_dir):
                time.sleep(0.1)
                shutil.rmtree(self.persist_dir)
                os.makedirs(self.persist_dir)
        # self.vector_store = None