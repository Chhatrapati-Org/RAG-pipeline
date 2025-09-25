import os
import re
import torch
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any
from threading import Lock
from pathlib import Path

from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from rag.qdrant import client

COLLECTION_NAME = "ps04"

# Initialize collection if it doesn't exist
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )


class MergedRAGWorker:
    """A single worker that handles the complete pipeline for a batch of files."""
    
    def __init__(self, worker_id: int, chunk_size_kb: int = 4, model_name: str = "jinaai/jina-embeddings-v2-small-en"):
        self.worker_id = worker_id
        self.chunk_size_bytes = chunk_size_kb * 1024
        self.model_name = model_name
        self.model = None
        self.point_id_counter = worker_id * 10000  # Unique ID range per worker
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize embedding model for this worker."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}
        
        self.model = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    
    def _read_file_chunks(self, file_path: str) -> List[Tuple[str, str, int]]:
        """Read a file and return chunks with metadata."""
        filename = os.path.basename(file_path)
        chunks = []
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                _, ext = os.path.splitext(filename)
                chunk_id = 0
                
                while True:
                    content = f.read(self.chunk_size_bytes)
                    if not content:
                        break
                    
                    # Clean JSON content
                    if ext.lower() == ".json":
                        content = re.sub(
                            r"[\[\]\{\}\"\n]",
                            "",
                            re.sub(r"\s[\s]+", " ", re.sub(r"\\[nrt]", "", content)),
                        )
                    
                    # Ensure chunk doesn't exceed size limit
                    content_bytes = content.encode("utf-8")
                    if len(content_bytes) > self.chunk_size_bytes:
                        # Split content to fit within limit
                        while len(content_bytes) > self.chunk_size_bytes:
                            split_point = self.chunk_size_bytes
                            # Try to find a word boundary
                            while split_point > 0 and content[split_point] != ' ':
                                split_point -= 1
                            if split_point == 0:  # No word boundary found
                                split_point = self.chunk_size_bytes
                            
                            chunk_part = content[:split_point]
                            chunks.append((chunk_part, filename, chunk_id))
                            chunk_id += 1
                            content = content[split_point:].lstrip()
                            content_bytes = content.encode("utf-8")
                    
                    if content:  # Add remaining content if any
                        chunks.append((content, filename, chunk_id))
                        chunk_id += 1
                        
        except Exception as e:
            print(f"Worker {self.worker_id}: Error reading {filename}: {e}")
            
        return chunks
    
    def _embed_chunks(self, chunks: List[Tuple[str, str, int]]) -> List[Tuple[List[float], Dict[str, Any]]]:
        """Embed chunks and create payloads."""
        if not chunks:
            return []
        
        try:
            texts = [chunk[0] for chunk in chunks]
            embeddings = self.model.embed_documents(texts)
            
            results = []
            for i, (chunk_text, filename, chunk_id) in enumerate(chunks):
                payload = {
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "chunk_size": len(chunk_text.encode("utf-8")),
                    "unique_id": str(uuid.uuid4()),
                    "worker_id": self.worker_id
                }
                results.append((embeddings[i], payload))
            
            return results
            
        except Exception as e:
            print(f"Worker {self.worker_id}: Error embedding chunks: {e}")
            return []
    
    def _store_embeddings(self, embeddings_with_payloads: List[Tuple[List[float], Dict[str, Any]]]) -> int:
        """Store embeddings in Qdrant."""
        if not embeddings_with_payloads:
            return 0
        
        try:
            points = []
            for embedding, payload in embeddings_with_payloads:
                self.point_id_counter += 1
                points.append(
                    PointStruct(
                        id=self.point_id_counter,
                        vector=embedding,
                        payload=payload,
                    )
                )
            
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            
            return len(points)
            
        except Exception as e:
            print(f"Worker {self.worker_id}: Error storing embeddings: {e}")
            return 0
    
    def process_file_batch(self, file_paths: List[str]) -> Dict[str, int]:
        """Process a batch of files through the complete pipeline."""
        stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0
        }
        
        for file_path in file_paths:
            try:
                # Step 1: Read file and create chunks
                chunks = self._read_file_chunks(file_path)
                if not chunks:
                    continue
                
                stats["chunks_created"] += len(chunks)
                
                # Step 2: Embed chunks
                embeddings_with_payloads = self._embed_chunks(chunks)
                if not embeddings_with_payloads:
                    stats["errors"] += 1
                    continue
                
                stats["embeddings_generated"] += len(embeddings_with_payloads)
                
                # Step 3: Store embeddings
                stored_count = self._store_embeddings(embeddings_with_payloads)
                stats["embeddings_stored"] += stored_count
                
                if stored_count > 0:
                    stats["files_processed"] += 1
                else:
                    stats["errors"] += 1
                    
            except Exception as e:
                print(f"Worker {self.worker_id}: Error processing {file_path}: {e}")
                stats["errors"] += 1
        
        return stats


class MergedMultiThreadedRAGPipeline:
    """Merged pipeline where each thread processes files through the complete pipeline."""
    
    def __init__(self, max_workers: int = 4, chunk_size_kb: int = 4, files_per_batch: int = 5):
        self.max_workers = max_workers
        self.chunk_size_kb = chunk_size_kb
        self.files_per_batch = files_per_batch
        self.lock = Lock()
        self.total_stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0
        }
    
    def _create_file_batches(self, directory_path: str) -> List[List[str]]:
        """Create batches of files for processing."""
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist")
        
        file_paths = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                file_paths.append(file_path)
        
        if not file_paths:
            print("No files found to process")
            return []
        
        # Create batches
        batches = [
            file_paths[i:i + self.files_per_batch] 
            for i in range(0, len(file_paths), self.files_per_batch)
        ]
        
        return batches
    
    def _update_stats(self, worker_stats: Dict[str, int]):
        """Thread-safe stats update."""
        with self.lock:
            for key, value in worker_stats.items():
                self.total_stats[key] += value
    
    def _process_batch_with_worker(self, batch_info: Tuple[int, List[str]]) -> Dict[str, int]:
        """Process a batch of files with a dedicated worker."""
        worker_id, file_paths = batch_info
        worker = MergedRAGWorker(
            worker_id=worker_id,
            chunk_size_kb=self.chunk_size_kb
        )
        
        stats = worker.process_file_batch(file_paths)
        return stats
    
    def process_directory(self, directory_path: str) -> Dict[str, int]:
        """Process directory with merged multithreaded pipeline."""
        print(f"Starting merged multithreaded processing of directory: {directory_path}")
        print(f"Configuration: {self.max_workers} workers, {self.files_per_batch} files per batch, {self.chunk_size_kb}KB max chunk size")
        
        # Create file batches
        file_batches = self._create_file_batches(directory_path)
        if not file_batches:
            return self.total_stats
        
        print(f"Created {len(file_batches)} batches from directory")
        
        # Process batches with thread pool
        batch_info = [(i, batch) for i, batch in enumerate(file_batches)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches for processing
            future_to_batch = {
                executor.submit(self._process_batch_with_worker, info): info
                for info in batch_info
            }
            
            # Process completed batches with progress tracking
            with tqdm(total=len(file_batches), desc="Processing file batches") as pbar:
                for future in as_completed(future_to_batch):
                    batch_info = future_to_batch[future]
                    try:
                        worker_stats = future.result()
                        self._update_stats(worker_stats)
                        
                        # Update progress description
                        pbar.set_postfix({
                            'files': self.total_stats['files_processed'],
                            'chunks': self.total_stats['chunks_created'],
                            'stored': self.total_stats['embeddings_stored']
                        })
                        
                    except Exception as e:
                        print(f"Error processing batch {batch_info[0]}: {e}")
                        self.total_stats["errors"] += len(batch_info[1])
                    finally:
                        pbar.update(1)
        
        # Print final statistics
        print("\n" + "="*50)
        print("PROCESSING COMPLETED")
        print("="*50)
        for key, value in self.total_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        return self.total_stats


# Convenience function
def run_merged_rag_pipeline(directory_path: str, max_workers: int = 4, chunk_size_kb: int = 4, files_per_batch: int = 5) -> Dict[str, int]:
    """
    Run the merged multithreaded RAG pipeline.
    
    Args:
        directory_path: Directory containing files to process
        max_workers: Number of worker threads
        chunk_size_kb: Maximum chunk size in KB
        files_per_batch: Number of files each worker processes per batch
    
    Returns:
        Dictionary with processing statistics
    """
    pipeline = MergedMultiThreadedRAGPipeline(
        max_workers=max_workers,
        chunk_size_kb=chunk_size_kb,
        files_per_batch=files_per_batch
    )
    
    return pipeline.process_directory(directory_path)