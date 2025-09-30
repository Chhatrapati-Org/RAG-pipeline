"""
Test script for the multithreaded retrieval module.
"""

from rag.retrieval import run_multithreaded_retrieval
import os

def test_retrieval():
    """Test the multithreaded retrieval system."""
    
    # File paths
    queries_file = r"C:\Users\22bcscs055\Downloads\Queries.json"
    output_file = r"C:\Users\22bcscs055\Documents\ps04-rag-v2\retrieval_results.json"
    
    # Check if queries file exists
    if not os.path.exists(queries_file):
        print(f"❌ Queries file not found: {queries_file}")
        return
    
    print("🚀 Starting multithreaded retrieval test...")
    
    try:
        # Run retrieval with custom parameters
        results = run_multithreaded_retrieval(
            queries_file_path=queries_file,
            output_file_path=output_file,
            max_workers=4,        # 4 worker threads
            top_k=5,             # Top 5 similar chunks per query
            queries_per_batch=20  # 20 queries per batch
        )
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Total queries processed: {len(results)}")
        print(f"📁 Results saved to: {output_file}")
        
        # Show sample results
        if results:
            print(f"\n📝 Sample result structure:")
            sample = results[0]
            for key in sample.keys():
                if key.startswith('chunk_1'):
                    print(f"   {key}: {str(sample[key])[:100]}...")
                else:
                    print(f"   {key}: {sample[key]}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retrieval()