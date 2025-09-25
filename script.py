import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import queue
import time

class RAGPipeline:
    def __init__(self, qdrant_url: str, qdrant_port: int, collection_name: str, 
                 chunk_size: int = 512, max_workers: int = 8, 
                 model_name: str = "all-MiniLM-L6-v2"):
        self.qdrant_client = QdrantClient(host=qdrant_url, port=qdrant_port)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.processed_count = 0
        self.lock = threading.Lock()
        
        self._setup_collection()
    
    def _setup_collection(self):
        try:
            self.qdrant_client.get_collection(self.collection_name)
            self.qdrant_client.delete_collection(self.collection_name)
        except:
            pass
        
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
        )
    
    def _read_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def _generate_embeddings(self, chunks: List[str]) -> List[np.ndarray]:
        return self.embedding_model.encode(chunks)
    
    def _create_point_id(self, file_path: str, chunk_idx: int) -> str:
        content = f"{file_path}_{chunk_idx}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _store_in_qdrant(self, points: List[PointStruct]):
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def _process_file(self, file_path: str) -> int:
        try:
            content = self._read_file(file_path)
            chunks = self._chunk_text(content)
            
            if not chunks:
                return 0
            
            embeddings = self._generate_embeddings(chunks)
            
            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = self._create_point_id(file_path, idx)
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "file_path": file_path,
                        "chunk_index": idx,
                        "text": chunk,
                        "file_name": os.path.basename(file_path)
                    }
                ))
            
            self._store_in_qdrant(points)
            
            with self.lock:
                self.processed_count += 1
            
            return len(chunks)
            
        except Exception as e:
            return 0
    
    def process_directory(self, directory_path: str, file_extensions: List[str] = ['.txt', '.md', '.py', '.json']):
        file_paths = []
        for ext in file_extensions:
            file_paths.extend(Path(directory_path).rglob(f'*{ext}'))
        
        file_paths = [str(p) for p in file_paths]
        
        total_chunks = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self._process_file, file_path): file_path 
                            for file_path in file_paths}
            
            for future in as_completed(future_to_file):
                chunks_processed = future.result()
                total_chunks += chunks_processed
        
        end_time = time.time()
        return {
            "total_files": len(file_paths),
            "processed_files": self.processed_count,
            "total_chunks": total_chunks,
            "processing_time": end_time - start_time
        }
    
    def process_file_list(self, file_paths: List[str]):
        total_chunks = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self._process_file, file_path): file_path 
                            for file_path in file_paths}
            
            for future in as_completed(future_to_file):
                chunks_processed = future.result()
                total_chunks += chunks_processed
        
        end_time = time.time()
        return {
            "total_files": len(file_paths),
            "processed_files": self.processed_count,
            "total_chunks": total_chunks,
            "processing_time": end_time - start_time
        }

def main():
    pipeline = RAGPipeline(
        qdrant_url="localhost",
        qdrant_port=6333,
        collection_name="documents",
        chunk_size=512,
        max_workers=16
    )
    
    result = pipeline.process_directory("C:\\Users\\22bcscs055\\Downloads\\mock_data")
    
    return result

if __name__ == "__main__":
    result = main()