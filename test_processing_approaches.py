"""
Test script to compare file-based vs chunk-based processing approaches.
"""

import os
import time
from rag.merged_pipeline import (
    run_merged_rag_pipeline,
    run_chunk_based_rag_pipeline,
    run_hybrid_rag_pipeline
)

def test_processing_approaches():
    """Test and compare different processing approaches."""
    
    # Test directory - update this path
    test_directory = r"C:\Users\22bcscs055\Downloads\mock_data"
    
    if not os.path.exists(test_directory):
        print(f"❌ Test directory not found: {test_directory}")
        print("Please update the path in the script")
        return
    
    # Count files for context
    file_count = len([f for f in os.listdir(test_directory) 
                     if os.path.isfile(os.path.join(test_directory, f))])
    
    print("🧪 TESTING RAG PROCESSING APPROACHES")
    print("=" * 60)
    print(f"📁 Directory: {test_directory}")
    print(f"📄 Files: {file_count}")
    print("=" * 60)
    
    # Test parameters
    max_workers = 4
    chunk_size_kb = 4
    
    results = {}
    
    # Approach 1: File-based processing
    print("\n🔹 Testing File-Based Processing")
    print("-" * 40)
    start_time = time.time()
    
    try:
        file_stats = run_merged_rag_pipeline(
            directory_path=test_directory,
            max_workers=max_workers,
            chunk_size_kb=chunk_size_kb,
            files_per_batch=5
        )
        
        file_duration = time.time() - start_time
        results['file_based'] = {
            'stats': file_stats,
            'duration': file_duration,
            'status': 'success'
        }
        
        print(f"✅ File-based processing completed in {file_duration:.2f}s")
        
    except Exception as e:
        print(f"❌ File-based processing failed: {e}")
        results['file_based'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # Wait a bit between approaches
    time.sleep(2)
    
    # Approach 2: Chunk-based processing
    print("\n🔸 Testing Chunk-Based Processing")
    print("-" * 40)
    start_time = time.time()
    
    try:
        chunk_stats = run_chunk_based_rag_pipeline(
            directory_path=test_directory,
            max_workers=max_workers,
            chunk_size_kb=chunk_size_kb,
            chunks_per_batch=50
        )
        
        chunk_duration = time.time() - start_time
        results['chunk_based'] = {
            'stats': chunk_stats,
            'duration': chunk_duration,
            'status': 'success'
        }
        
        print(f"✅ Chunk-based processing completed in {chunk_duration:.2f}s")
        
    except Exception as e:
        print(f"❌ Chunk-based processing failed: {e}")
        results['chunk_based'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # Wait a bit between approaches
    time.sleep(2)
    
    # Approach 3: Hybrid (auto-select)
    print("\n🔷 Testing Hybrid Processing (Auto-Select)")
    print("-" * 40)
    start_time = time.time()
    
    try:
        hybrid_stats = run_hybrid_rag_pipeline(
            directory_path=test_directory,
            max_workers=max_workers,
            chunk_size_kb=chunk_size_kb,
            files_per_batch=5,
            chunks_per_batch=50,
            use_chunk_based=None  # Auto-select
        )
        
        hybrid_duration = time.time() - start_time
        results['hybrid'] = {
            'stats': hybrid_stats,
            'duration': hybrid_duration,
            'status': 'success'
        }
        
        print(f"✅ Hybrid processing completed in {hybrid_duration:.2f}s")
        
    except Exception as e:
        print(f"❌ Hybrid processing failed: {e}")
        results['hybrid'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # Compare results
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS")
    print("=" * 60)
    
    for approach, result in results.items():
        print(f"\n🔍 {approach.replace('_', ' ').title()}:")
        
        if result['status'] == 'success':
            stats = result['stats']
            duration = result['duration']
            
            print(f"   ⏱️  Duration: {duration:.2f}s")
            print(f"   📄 Files: {stats.get('files_processed', 0)}")
            print(f"   🧩 Chunks: {stats.get('chunks_created', 0)}")
            print(f"   📦 Stored: {stats.get('embeddings_stored', 0)}")
            print(f"   ❌ Errors: {stats.get('errors', 0)}")
            
            # Calculate throughput
            if duration > 0:
                chunks_per_sec = stats.get('chunks_created', 0) / duration
                print(f"   🚀 Throughput: {chunks_per_sec:.1f} chunks/sec")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • File count: {file_count}")
    if file_count <= 10:
        print(f"   • Recommended: File-based (fewer files, simpler coordination)")
    elif file_count <= 50:
        print(f"   • Recommended: Either approach works well")
    else:
        print(f"   • Recommended: Chunk-based (many files, better load balancing)")
    
    print(f"   • Use hybrid mode for automatic selection based on file count")

def demo_chunk_processing():
    """Demonstrate chunk-based processing features."""
    
    print("\n" + "=" * 60)
    print("🧩 CHUNK-BASED PROCESSING DEMO")
    print("=" * 60)
    
    print("Key Features:")
    print("✅ Better load balancing across workers")
    print("✅ Handles files with varying chunk counts")
    print("✅ More granular progress tracking")
    print("✅ Efficient for many small files")
    print("✅ Separates I/O from compute operations")
    
    print(f"\nProcessing Flow:")
    print(f"1. 📖 Read all files sequentially (extract chunks)")
    print(f"2. 📦 Group chunks into batches (e.g., 50 chunks/batch)")
    print(f"3. 🧵 Distribute batches across worker threads")
    print(f"4. ⚡ Workers process: embed → store in parallel")
    print(f"5. 📊 Aggregate results from all workers")
    
    print(f"\nBest Use Cases:")
    print(f"• Many small JSON files with varying sizes")
    print(f"• Mixed file types (JSON + text)")
    print(f"• When load balancing is important")
    print(f"• Large datasets with uneven file sizes")

if __name__ == "__main__":
    demo_chunk_processing()
    print("\n" + "🔥" * 20)
    test_processing_approaches()