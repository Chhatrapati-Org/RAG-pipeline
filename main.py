#!/usr/bin/env python3
"""
Main entry point for the merged multithreaded RAG pipeline.

This script processes a directory of files using the new merged multithreaded approach:
- Each thread processes a batch of files through the complete pipeline consecutively
- More memory-efficient and better resource utilization
- Reads files → chunks them → embeds → stores (per thread)
"""

import sys
import os
from pathlib import Path

from rag.pipeline import run_rag_pipeline
from rag.merged_pipeline import run_merged_rag_pipeline


def main():
    """Main function to run the merged multithreaded RAG pipeline."""
    # Directory to process - update this path as needed
    directory_path = r"C:\Users\22bcscs055\Downloads\mock_data"  # Change this to your data directory path
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        print("Please update the directory_path in main.py to point to your data directory.")
        sys.exit(1)
    
    print("="*60)
    print("MERGED MULTITHREADED RAG PIPELINE")
    print("="*60)
    print(f"Processing directory: {directory_path}")
    print(f"Configuration:")
    print(f"  • Max workers: 4 threads")
    print(f"  • Max chunk size: 4KB") 
    print(f"  • Files per batch: 5")
    print(f"  • Pipeline: Read → Chunk → Embed → Store (per thread)")
    print("-"*60)
    
    try:
        # Use the new merged pipeline for optimal performance
        stats = run_merged_rag_pipeline(
            directory_path=directory_path,
            max_workers=24,           # Adjust based on your CPU cores
            chunk_size_kb=4,         # 4KB max chunk size
            files_per_batch=5        # Files processed per thread batch
        )
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("FINAL STATISTICS:")
        print(f"Files processed: {stats.get('files_processed', 0)}")
        print(f"Total chunks created: {stats.get('chunks_created', 0)}")
        print(f"Embeddings generated: {stats.get('embeddings_generated', 0)}")
        print(f"Embeddings stored: {stats.get('embeddings_stored', 0)}")
        print(f"Errors encountered: {stats.get('errors', 0)}")
        
        print("\nResults stored in Qdrant collection 'ps04' with metadata:")
        print("  • filename: Original file name")
        print("  • chunk_id: Sequential chunk number within file") 
        print("  • text: The actual chunk content")
        print("  • chunk_size: Size of chunk in bytes")
        print("  • unique_id: UUID for each chunk")
        print("  • worker_id: ID of the worker thread that processed it")
        
    except Exception as e:
        print(f"\nError processing directory: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def legacy_main():
    """Legacy function using the old pipeline approach.""" 
    directory_path = r"C:\Users\22bcscs055\Downloads\final_train"
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        sys.exit(1)
        
    print("Using legacy pipeline (separate stages)...")
    processed_count = run_rag_pipeline(
        directory_path=directory_path,
        max_workers=4,
        chunk_size_kb=4,
        use_merged=False  # Use legacy approach
    )
    
    print(f"Legacy pipeline processed {processed_count} chunks")


def process_directory(directory_path):
    """Legacy function for backward compatibility - now uses merged pipeline."""
    stats = run_merged_rag_pipeline(directory_path)
    return stats.get('embeddings_stored', 0)


if __name__ == "__main__":
    main()
