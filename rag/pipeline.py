from typing import List, Tuple, Dict
from rag.ingestor import MultiThreadedFileReader
from rag.chunker import MultiThreadedChunker  
from rag.store_embedding import MultiThreadedEmbeddingStore
from rag.merged_pipeline import MergedMultiThreadedRAGPipeline, run_merged_rag_pipeline


class MultiThreadedRAGPipeline:
    """
    Legacy pipeline - processes in separate stages (read all → embed all → store all).
    For better memory efficiency, use MergedMultiThreadedRAGPipeline instead.
    """
    def __init__(self, max_workers=4, chunk_size_kb=4, embedding_batch_size=32, storage_batch_size=100):
        """
        Initialize the multithreaded RAG pipeline.
        
        Args:
            max_workers: Number of worker threads for parallel processing
            chunk_size_kb: Maximum chunk size in KB (4KB default)
            embedding_batch_size: Batch size for embedding processing
            storage_batch_size: Batch size for storage operations
        """
        self.file_reader = MultiThreadedFileReader(max_workers=max_workers, chunk_size_kb=chunk_size_kb)
        self.chunker = MultiThreadedChunker(max_workers=max_workers)
        self.embedding_store = MultiThreadedEmbeddingStore(max_workers=max_workers, batch_size=storage_batch_size)
        self.embedding_batch_size = embedding_batch_size

    def process_directory(self, directory_path: str) -> int:
        """
        Process a directory of files with multithreading:
        1. Read files multithreaded and create 4KB chunks
        2. Embed chunks multithreaded  
        3. Store embeddings with file names in payload
        
        Returns:
            Number of successfully processed chunks
        """
        print(f"Starting legacy multithreaded processing of directory: {directory_path}")
        print("NOTE: Consider using MergedMultiThreadedRAGPipeline for better memory efficiency")
        
        # Step 1: Read files and create chunks (multithreaded)
        print("Step 1: Reading files and creating chunks...")
        chunks = []
        for chunk_content, filename, chunk_id in self.file_reader.read_files_multithreaded(directory_path):
            chunks.append((chunk_content, filename, chunk_id))
        
        if not chunks:
            print("No chunks created from directory")
            return 0
        
        print(f"Created {len(chunks)} chunks from directory")
        
        # Step 2: Embed chunks (multithreaded)
        print("Step 2: Embedding chunks...")
        embeddings_with_payloads = self.chunker.process_chunks_multithreaded(chunks, self.embedding_batch_size)
        
        if not embeddings_with_payloads:
            print("No embeddings created")
            return 0
        
        print(f"Created {len(embeddings_with_payloads)} embeddings")
        
        # Step 3: Store embeddings with payloads (multithreaded)
        print("Step 3: Storing embeddings...")
        stored_count = self.embedding_store.store_embeddings_multithreaded(embeddings_with_payloads)
        
        print(f"Pipeline completed. Processed {stored_count} chunks successfully.")
        return stored_count


def process_directory(directory_path):
    """
    Legacy function for backward compatibility.
    Now uses the new merged pipeline for better efficiency.
    """
    print("Using merged multithreaded RAG pipeline...")
    stats = run_merged_rag_pipeline(directory_path)
    return stats.get('embeddings_stored', 0)


def run_rag_pipeline(directory_path: str, max_workers: int = 4, chunk_size_kb: int = 4, use_merged: bool = True) -> int:
    """
    Convenience function to run the complete RAG pipeline.
    
    Args:
        directory_path: Path to directory containing files to process
        max_workers: Number of worker threads (default: 4)
        chunk_size_kb: Maximum chunk size in KB (default: 4)
        use_merged: Whether to use merged pipeline (default: True, recommended)
    
    Returns:
        Number of successfully processed chunks
    """
    if use_merged:
        # Use the new merged pipeline (recommended)
        stats = run_merged_rag_pipeline(
            directory_path=directory_path,
            max_workers=max_workers,
            chunk_size_kb=chunk_size_kb,
            files_per_batch=5  # Optimal batch size
        )
        return stats.get('embeddings_stored', 0)
    else:
        # Use legacy pipeline
        pipeline = MultiThreadedRAGPipeline(
            max_workers=max_workers,
            chunk_size_kb=chunk_size_kb,
            embedding_batch_size=32,
            storage_batch_size=100
        )
        return pipeline.process_directory(directory_path)
