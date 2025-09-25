#!/usr/bin/env python3
"""
Example usage of both the legacy and merged multithreaded RAG pipelines.

This script demonstrates:
1. Legacy approach: Read all → Embed all → Store all (memory intensive)
2. Merged approach: Each thread processes batch through complete pipeline (memory efficient)
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.pipeline import run_rag_pipeline
from rag.merged_pipeline import run_merged_rag_pipeline


def create_example_data(data_directory="example_data"):
    """Create example files for testing."""
    if not os.path.exists(data_directory):
        print(f"Creating example data directory: {data_directory}")
        os.makedirs(data_directory, exist_ok=True)
        
        # Create diverse example files
        example_files = {
            "document1.txt": "This is the first document about artificial intelligence. " * 100,
            "document2.txt": "This document discusses machine learning algorithms. " * 120,
            "document3.txt": "Natural language processing and deep learning concepts. " * 80,
            "data.json": '{"title": "JSON Document", "content": "' + "Sample JSON data for RAG processing. " * 150 + '"}',
            "research.txt": "Research paper content about neural networks and transformers. " * 200,
            "notes.txt": "Meeting notes and project documentation. " * 90,
        }
        
        for filename, content in example_files.items():
            file_path = os.path.join(data_directory, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        print(f"Created {len(example_files)} example files in {data_directory}")
        
        # Print file sizes
        print("File sizes:")
        for filename in example_files.keys():
            file_path = os.path.join(data_directory, filename)
            size = os.path.getsize(file_path)
            print(f"  {filename}: {size:,} bytes ({size/1024:.1f} KB)")
    
    return data_directory


def demo_legacy_pipeline(data_directory):
    """Demonstrate the legacy pipeline approach."""
    print("\n" + "="*60)
    print("LEGACY PIPELINE DEMO")
    print("="*60)
    print("Architecture: Read All Files → Embed All Chunks → Store All")
    print("Memory Usage: Higher (stores all data in memory)")
    print("-"*60)
    
    processed_count = run_rag_pipeline(
        directory_path=data_directory,
        max_workers=4,
        chunk_size_kb=4,
        use_merged=False  # Use legacy approach
    )
    
    print(f"\n✓ Legacy pipeline processed {processed_count} chunks")
    return processed_count


def demo_merged_pipeline(data_directory):
    """Demonstrate the new merged pipeline approach."""
    print("\n" + "="*60)
    print("MERGED PIPELINE DEMO")
    print("="*60)
    print("Architecture: Each Thread → Read Batch → Chunk → Embed → Store")
    print("Memory Usage: Lower (processes batches consecutively)")
    print("Resource Utilization: Better (all components working in parallel)")
    print("-"*60)
    
    stats = run_merged_rag_pipeline(
        directory_path=data_directory,
        max_workers=4,           # 4 worker threads
        chunk_size_kb=4,         # 4KB max chunks
        files_per_batch=2        # 2 files per batch per thread
    )
    
    print(f"\n✓ Merged pipeline completed with detailed stats:")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    return stats


def compare_approaches():
    """Compare both pipeline approaches."""
    print("\n" + "="*60)
    print("APPROACH COMPARISON")
    print("="*60)
    
    comparison = """
┌─────────────────────┬──────────────────┬────────────────────┐
│ Aspect              │ Legacy Pipeline  │ Merged Pipeline    │
├─────────────────────┼──────────────────┼────────────────────┤
│ Memory Usage        │ High             │ Low                │
│ Resource Efficiency │ Sequential       │ Parallel           │
│ Scalability         │ Limited          │ Excellent          │
│ Error Isolation     │ Global           │ Per-batch          │
│ Progress Tracking   │ Stage-based      │ Overall            │
│ Throughput          │ Good             │ Better             │
│ Recommended For     │ Small datasets   │ Large datasets     │
└─────────────────────┴──────────────────┴────────────────────┘
    """
    
    print(comparison)
    
    print("\nKey Differences:")
    print("📊 Legacy: Read all files → Process all chunks → Store all embeddings")
    print("🚀 Merged: Thread 1 processes batch A, Thread 2 processes batch B, etc.")
    print("💾 Merged uses less memory by processing batches consecutively")
    print("⚡ Merged provides better CPU/GPU utilization")


def main():
    """Run the complete demo."""
    print("="*60)
    print("RAG PIPELINE COMPARISON DEMO")
    print("="*60)
    print("Demonstrating both legacy and merged multithreaded approaches")
    
    try:
        # Create example data
        data_directory = create_example_data()
        
        # Show comparison table
        compare_approaches()
        
        # Demo both approaches
        print(f"\nProcessing {len(os.listdir(data_directory))} files from {data_directory}")
        
        # Note: Uncomment the approach you want to test
        # Both require Qdrant server and model downloads
        
        try:
            # Demo merged pipeline (recommended)
            merged_stats = demo_merged_pipeline(data_directory)
            print("\n🎉 Merged pipeline demo completed successfully!")
            
            # Optionally demo legacy pipeline for comparison
            # legacy_count = demo_legacy_pipeline(data_directory)
            # print("\n🎉 Legacy pipeline demo completed successfully!")
            
        except Exception as e:
            print(f"\n⚠️  Demo requires:")
            print("  1. Qdrant server: docker run -p 6333:6333 qdrant/qdrant")
            print("  2. Dependencies: pip install -e .")
            print(f"\nError: {e}")
        
        print("\n" + "="*60)
        print("DEMO COMPLETED")
        print("="*60)
        print("✅ Example data created successfully")
        print("✅ Pipeline architectures explained")
        print("📚 Use merged pipeline for production workloads")
        print("🔧 Use legacy pipeline only for small datasets")
        
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()