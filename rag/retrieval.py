"""
Multithreaded Retrieval Module for Query Processing
==================================================

This module processes queries from a JSON file, divides them into batches for threading,
encodes each query using the shared embedding model, finds top-k similar chunks from 
the Qdrant collection, and outputs results to a JSON file.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from threading import Lock
from tqdm import tqdm

from rag.pipeline import SharedEmbeddingModel, COLLECTION_NAME
from rag.qdrant import client




class MultiThreadedRetriever:
    """Multithreaded retrieval system for processing query batches."""
    
    def __init__(self, max_workers: int = 4, top_k: int = 5, queries_per_batch: int = 10):
        """
        Initialize the multithreaded retriever.
        
        Args:
            max_workers: Number of worker threads
            top_k: Number of top similar chunks to retrieve per query
            queries_per_batch: Number of queries each worker processes per batch
        """
        self.max_workers = max_workers
        self.top_k = top_k
        self.queries_per_batch = queries_per_batch
        self.lock = Lock()
        
        # Initialize shared embedding model
        print("Initializing shared embedding model for retrieval...")
        self.shared_model = SharedEmbeddingModel()
        self.shared_model.initialize_model()
        print("✅ Shared embedding model ready for retrieval")
        
        # Verify collection exists
        self._verify_collection()
    
    def _verify_collection(self):
        """Verify that the target collection exists."""
        try:
            if not client.collection_exists(COLLECTION_NAME):
                raise ValueError(f"Collection '{COLLECTION_NAME}' does not exist. Please run the ingestion pipeline first.")
            
            # Get collection info
            collection_info = client.get_collection(COLLECTION_NAME)
            vector_size = collection_info.config.params.vectors.size
            point_count = client.count(COLLECTION_NAME).count
            
            print(f"✅ Collection '{COLLECTION_NAME}' found:")
            print(f"   - Vector dimensions: {vector_size}")
            print(f"   - Total points: {point_count}")
            
        except Exception as e:
            raise RuntimeError(f"Error verifying collection: {e}")
    
    def load_queries(self, queries_file_path: str) -> List[Dict[str, str]]:
        """Load queries from JSON file."""
        try:
            with open(queries_file_path, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            
            print(f"✅ Loaded {len(queries)} queries from {queries_file_path}")
            return queries
            
        except Exception as e:
            raise RuntimeError(f"Error loading queries from {queries_file_path}: {e}")
    
    def _create_query_batches(self, queries: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        """Divide queries into batches for threading."""
        batches = [
            queries[i:i + self.queries_per_batch] 
            for i in range(0, len(queries), self.queries_per_batch)
        ]
        
        print(f"✅ Created {len(batches)} query batches ({self.queries_per_batch} queries per batch)")
        return batches
    
    def _process_single_query(self, query_data: Dict[str, str]) -> Dict[str, Any]:
        """Process a single query and return results."""
        query_num = query_data["query_num"]
        query_text = query_data["query"]
        
        try:
            # Encode the query using shared model
            query_embedding = self.shared_model.embed_documents([query_text])[0]
            
            # Search for similar chunks in Qdrant
            search_results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=self.top_k,
                with_payload=True,
                with_vectors=False  # We don't need the vectors in results
            )
            
            # Format results
            chunks = []
            for i, result in enumerate(search_results, 1):
                chunk_data = {
                    f"chunk_{i}_text": result.payload.get("text", ""),
                    f"chunk_{i}_filename": result.payload.get("filename", ""),
                    f"chunk_{i}_score": float(result.score),
                    f"chunk_{i}_chunk_id": result.id
                }
                chunks.append(chunk_data)
            files = [result.payload.get("filename", "") for result in search_results]
            # Create result structure
            result_data = {
                "query_num": query_num,
                "query": query_text,
                "response": files
            }
            
            # # Add chunk data to results
            # for chunk in chunks:
            #     result_data.update(chunk)
            
            return result_data
            
        except Exception as e:
            print(f"❌ Error processing query {query_num}: {e}")
            return {
                "query_num": query_num,
                "query": query_text,
                "error": str(e),
                "retrieval_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def _process_query_batch(self, batch_info: Tuple[int, List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """Process a batch of queries."""
        batch_id, queries = batch_info
        results = []
        
        print(f"Worker {batch_id}: Processing {len(queries)} queries...")
        
        for query_data in queries:
            result = self._process_single_query(query_data)
            results.append(result)
        
        print(f"Worker {batch_id}: Completed {len(results)} queries")
        return results
    
    def process_queries(self, queries_file_path: str, output_file_path: str = None) -> List[Dict[str, Any]]:
        """
        Process all queries using multithreading.
        
        Args:
            queries_file_path: Path to the input JSON file with queries
            output_file_path: Path to save results (optional)
        
        Returns:
            List of results for all queries
        """
        print("="*60)
        print("STARTING MULTITHREADED QUERY RETRIEVAL")
        print("="*60)
        
        # Load queries
        queries = self.load_queries(queries_file_path)
        
        # Create batches
        query_batches = self._create_query_batches(queries)
        
        # Process batches with threading
        all_results = []
        batch_info = [(i, batch) for i, batch in enumerate(query_batches)]
        
        print(f"🚀 Starting retrieval with {self.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_query_batch, info): info
                for info in batch_info
            }
            
            # Process results with progress tracking
            with tqdm(total=len(query_batches), desc="Processing query batches") as pbar:
                for future in as_completed(future_to_batch):
                    batch_info = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                        
                        # Update progress
                        pbar.set_postfix({
                            'completed': len(all_results),
                            'total_queries': len(queries)
                        })
                        
                    except Exception as e:
                        print(f"❌ Error processing batch {batch_info[0]}: {e}")
                    finally:
                        pbar.update(1)
        
        # Sort results by query_num for consistent ordering
        try:
            all_results.sort(key=lambda x: int(x.get("query_num", "0")))
        except (ValueError, TypeError):
            # Fallback to string sorting if numeric conversion fails
            all_results.sort(key=lambda x: x.get("query_num", ""))
        
        # Save results if output path provided
        if output_file_path:
            self._save_results(all_results, output_file_path)
        
        # Print summary
        print("\n" + "="*60)
        print("RETRIEVAL COMPLETED")
        print("="*60)
        successful = len([r for r in all_results if "error" not in r])
        failed = len(all_results) - successful
        
        print(f"Total queries processed: {len(all_results)}")
        print(f"Successful retrievals: {successful}")
        print(f"Failed retrievals: {failed}")
        print(f"Top-k per query: {self.top_k}")
        
        return all_results
    
    def _save_results(self, results: List[Dict[str, Any]], output_file_path: str):
        """Save results to JSON file."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Results saved to: {output_file_path}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def run_multithreaded_retrieval(
    queries_file_path: str,
    output_file_path: str = None,
    max_workers: int = 16,
    top_k: int = 5,
    queries_per_batch: int = 20
) -> List[Dict[str, Any]]:
    """
    Convenience function to run the multithreaded retrieval.
    
    Args:
        queries_file_path: Path to input JSON file with queries
        output_file_path: Path to save results JSON file
        max_workers: Number of worker threads
        top_k: Number of top similar chunks to retrieve per query
        queries_per_batch: Number of queries per batch
    
    Returns:
        List of retrieval results
    """
    retriever = MultiThreadedRetriever(
        max_workers=max_workers,
        top_k=top_k,
        queries_per_batch=queries_per_batch
    )
    
    return retriever.process_queries(queries_file_path, output_file_path)


if __name__ == "__main__":
    # Example usage
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
