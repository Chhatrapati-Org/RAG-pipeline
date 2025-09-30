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
import datetime
from pathlib import Path

from rag.merged_pipeline import run_merged_rag_pipeline
from rag.retrieval import run_multithreaded_retrieval



def main():
    """Main function to run the merged multithreaded RAG pipeline."""
    # Directory to process - update this path as needed
    directory_path = r"C:\Users\22bcscs055\Downloads\mock_data"  # Change this to your data directory path
    
    try:
        # Use the new merged pipeline for optimal performance
        stats = run_merged_rag_pipeline(
            directory_path=directory_path,
            max_workers=16,           # Adjust based on your CPU cores
            chunk_size_kb=4,         # 4KB max chunk size
            files_per_batch=50        # Files processed per thread batch
        )
        queries_file = r"C:\Users\22bcscs055\Downloads\Queries.json"
        output_file = r"C:\Users\22bcscs055\Documents\ps04-rag-v2\retrieval_results.json"
        
        results = run_multithreaded_retrieval(
            queries_file_path=queries_file,
            output_file_path=output_file,
            max_workers=16,
            top_k=5,
            queries_per_batch=20
        )
        
        print(f"Processing complete! Results saved with {len(results)} queries processed.")
        
        
    except Exception as e:
        print(f"\nError processing directory: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
