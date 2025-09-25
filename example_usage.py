#!/usr/bin/env python3
"""
Example usage of the multithreaded RAG pipeline.

This script demonstrates how to use the new multithreaded RAG system that:
1. Reads files multithreaded from a directory
2. Creates chunks of 4KB or less
3. Embeds chunks using multiple threads
4. Stores embeddings with file names in the payload
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.pipeline import MultiThreadedRAGPipeline, run_rag_pipeline


def main():
    # Example directory containing text files to process
    # Replace this with your actual data directory
    data_directory = "data"  # Change this to your data directory path
    
    if not os.path.exists(data_directory):
        print(f"Creating example data directory: {data_directory}")
        os.makedirs(data_directory, exist_ok=True)
        
        # Create some example files for testing
        example_files = {
            "document1.txt": "This is the first document. " * 100,  # Create some content
            "document2.txt": "This is the second document with different content. " * 150,
            "document3.json": '{"title": "JSON Document", "content": "' + "Sample JSON content. " * 200 + '"}',
        }
        
        for filename, content in example_files.items():
            with open(os.path.join(data_directory, filename), "w", encoding="utf-8") as f:
                f.write(content)
        
        print(f"Created example files in {data_directory}")
    
    print("\n" + "="*50)
    print("MULTITHREADED RAG PIPELINE EXAMPLE")
    print("="*50)
    
    # Method 1: Using the convenience function
    print("\nMethod 1: Using convenience function")
    print("-" * 30)
    processed_count = run_rag_pipeline(
        directory_path=data_directory,
        max_workers=4,           # Number of worker threads
        chunk_size_kb=4          # 4KB chunk size limit
    )
    
    print(f"\nProcessed {processed_count} chunks using convenience function")
    
    # Method 2: Using the pipeline class directly (more control)
    print("\n\nMethod 2: Using pipeline class directly")
    print("-" * 40)
    
    pipeline = MultiThreadedRAGPipeline(
        max_workers=6,              # More threads for faster processing
        chunk_size_kb=3,            # Smaller chunks (3KB)
        embedding_batch_size=16,    # Smaller embedding batches
        storage_batch_size=50       # Smaller storage batches
    )
    
    processed_count_2 = pipeline.process_directory(data_directory)
    print(f"\nProcessed {processed_count_2} chunks using pipeline class")
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nFeatures demonstrated:")
    print("✓ Multithreaded file reading")
    print("✓ 4KB (or less) chunk size enforcement")
    print("✓ Multithreaded embedding generation")
    print("✓ File names stored in payload metadata")
    print("✓ Multithreaded embedding storage")
    print("✓ Progress bars for monitoring")


if __name__ == "__main__":
    main()