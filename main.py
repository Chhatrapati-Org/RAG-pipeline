#!/usr/bin/env python3
"""
Main entry point for the multithreaded RAG pipeline.

This script processes a directory of files using the new multithreaded approach:
- Reads files multithreaded and creates chunks of 4KB or less
- Embeds chunks using multiple threads  
- Stores embeddings with file names in payload metadata
"""

import sys
import os
from pathlib import Path

from rag.pipeline import MultiThreadedRAGPipeline, run_rag_pipeline


def main():
    """Main function to run the multithreaded RAG pipeline."""
    # Directory to process - update this path as needed
    directory_path = r"C:\Users\22bcscs055\Downloads\mock_data"  # Change this to your data directory path
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        print("Please update the directory_path in main.py to point to your data directory.")
        sys.exit(1)
    
    print("="*60)
    print("MULTITHREADED RAG PIPELINE - PROCESSING DIRECTORY")
    print("="*60)
    print(f"Processing directory: {directory_path}")
    print(f"Configuration:")
    print(f"  • Max workers: 4 threads")
    print(f"  • Max chunk size: 4KB") 
    print(f"  • Embedding batch size: 32")
    print(f"  • Storage batch size: 100")
    print("-"*60)
    
    try:
        # Method 1: Using convenience function (recommended)
        processed_count = run_rag_pipeline(
            directory_path=directory_path,
            max_workers=4,           # Adjust based on your CPU cores
            chunk_size_kb=4          # 4KB max chunk size
        )
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total chunks processed: {processed_count}")
        print("\nResults stored in Qdrant collection 'ps04' with metadata:")
        print("  • filename: Original file name")
        print("  • chunk_id: Sequential chunk number within file") 
        print("  • text: The actual chunk content")
        print("  • chunk_size: Size of chunk in bytes")
        print("  • unique_id: UUID for each chunk")
        
    except Exception as e:
        print(f"\nError processing directory: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




if __name__ == "__main__":
    main()
