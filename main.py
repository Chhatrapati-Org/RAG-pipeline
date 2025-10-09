#!/usr/bin/env python3

import datetime
import os
import sys
from pathlib import Path

from rag.pipeline import run_merged_rag_pipeline
from rag.retrieve import run_multithreaded_retrieval


def main():
    """Main function to run the merged multithreaded RAG pipeline."""
    # Directory to process - update this path as needed
    directory_path_mock = r"C:\Users\22bcscs055\Downloads\mock_data_half_processed"  # Change this to your data directory path
    directory_path_json = r"C:\Users\22bcscs055\Documents\ps04-rag-v2\json_file"
    try:
        # Use the new merged pipeline for optimal performance
        stats = run_merged_rag_pipeline(
            directory_path=directory_path_mock,
            max_workers=20,  # Adjust based on your CPU cores
            chunk_size_kb=4,  # 4KB max chunk size
            files_per_batch=20,  # Files processed per thread batch
        )
        # stats = run_chunk_based_rag_pipeline(
        #     directory_path=directory_path_json,
        #     max_workers=16,           # Adjust based on your CPU cores
        #     chunk_size_kb=1,         # 4KB max chunk size
        #     chunks_per_batch=50        # Files processed per thread batch
        # )
        queries_file = r"C:\Users\22bcscs055\Downloads\Queries.json"
        output_file = (
            r"C:\Users\22bcscs055\Documents\ps04-rag-v2\retrieval_results3.json"
        )

        results = run_multithreaded_retrieval(
            queries_file_path=queries_file,
            output_file_path=output_file,
            max_workers=16,
            top_k=5,
            queries_per_batch=20,
        )

        print(
            f"Processing complete! Results saved with {len(results)} queries processed."
        )

    except Exception as e:
        print(f"\nError processing directory: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
