"""
Direct retrieval script - processes the queries and saves results.
Run this script to process all queries from Queries.json
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.retrieval import run_multithreaded_retrieval

def main():
    """Main function to run the retrieval."""
    
    # Configuration
    queries_file = r"C:\Users\22bcscs055\Downloads\Queries.json"
    output_file = r"C:\Users\22bcscs055\Documents\ps04-rag-v2\query_retrieval_results.json"
    
    # Retrieval parameters
    MAX_WORKERS = 4           # Number of threads
    TOP_K = 5                # Number of similar chunks per query
    QUERIES_PER_BATCH = 15   # Queries per batch
    
    print("🔍 MULTITHREADED QUERY RETRIEVAL")
    print("=" * 50)
    print(f"📁 Input file: {queries_file}")
    print(f"💾 Output file: {output_file}")
    print(f"🧵 Workers: {MAX_WORKERS}")
    print(f"🔢 Top-K: {TOP_K}")
    print(f"📦 Batch size: {QUERIES_PER_BATCH}")
    print("=" * 50)
    
    # Check if input file exists
    if not os.path.exists(queries_file):
        print(f"❌ Error: Queries file not found at {queries_file}")
        print("Please make sure the file exists and try again.")
        return
    
    try:
        # Run the retrieval
        results = run_multithreaded_retrieval(
            queries_file_path=queries_file,
            output_file_path=output_file,
            max_workers=MAX_WORKERS,
            top_k=TOP_K,
            queries_per_batch=QUERIES_PER_BATCH
        )
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Processed {len(results)} queries")
        print(f"📄 Results saved to: {output_file}")
        
        # Show first few results as examples
        if results:
            print(f"\n📋 Sample results:")
            for i, result in enumerate(results[:3]):  # Show first 3
                print(f"\n  Query {result.get('query_num', 'N/A')}: {result.get('query', 'N/A')[:60]}...")
                if 'chunk_1_text' in result:
                    print(f"    Top chunk: {result['chunk_1_text'][:100]}...")
                    print(f"    From file: {result.get('chunk_1_filename', 'N/A')}")
                    print(f"    Score: {result.get('chunk_1_score', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()