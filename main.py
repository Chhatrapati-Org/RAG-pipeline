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

from rag.merged_pipeline import run_merged_rag_pipeline


def main():
    """Main function to run the merged multithreaded RAG pipeline."""
    # Directory to process - update this path as needed
    directory_path = r"C:\Users\22bcscs055\Downloads\mock_data"  # Change this to your data directory path
    
    try:
        # Use the new merged pipeline for optimal performance
        stats = run_merged_rag_pipeline(
            directory_path=directory_path,
            max_workers=4,           # Adjust based on your CPU cores
            chunk_size_kb=4,         # 4KB max chunk size
            files_per_batch=50        # Files processed per thread batch
        )
        
        
    except Exception as e:
        print(f"\nError processing directory: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
