import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Tuple

from FlagEmbedding import FlagReranker
from qdrant_client import models
from tqdm import tqdm

from rag.pipeline import SharedEmbeddingModel

class SharedRerankingModel:
    """Singleton class for shared reranking model across all threads."""
    
    _instance = None
    _reranker_model = None
    _lock = Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SharedRerankingModel, cls).__new__(cls)
        return cls._instance

    def initialize_model(
        self,
        reranker_model_name: str = "BAAI/bge-reranker-base",
        use_fp16: bool = True,  # Use FP16 for faster inference on GPU
    ):
        """
        Initialize the BGE reranker model.
        
        Args:
            reranker_model_name: HuggingFace model name for BGE reranker
            use_fp16: Whether to use FP16 precision (faster on GPU)
        """
        with self._lock:
            if self._initialized and self._reranker_model is not None:
                return

            print("Initializing BGE reranking model...")
            print(f"  Reranker model: {reranker_model_name}")

            try:
                # Initialize FlagReranker from FlagEmbedding
                self._reranker_model = FlagReranker(
                    reranker_model_name,
                    use_fp16=use_fp16,  # Enable FP16 for GPU acceleration
                    cache_dir="./model_cache"
                )
                
                # Test the model
                test_scores = self._reranker_model.compute_score(
                    [['test query', 'test document']]
                )
                print(f"✅ Reranker model initialized successfully (test score: {test_scores})")
                
                self._initialized = True

            except Exception as e:
                print(f"❌ Failed to initialize reranker model: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError("Cannot initialize reranker model")

    def get_reranker_model(self):
        """Get the initialized reranker model."""
        if not self._initialized or self._reranker_model is None:
            raise RuntimeError("Reranker model not initialized. Call initialize_model() first.")
        return self._reranker_model

    def compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute reranking scores for query-document pairs.
        
        Args:
            query: The query string
            documents: List of document strings to rerank
            
        Returns:
            List of relevance scores (higher = more relevant)
        """
        with self._lock:
            # Validate inputs
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            if not documents:
                return []
            
            reranker = self.get_reranker_model()
            
            # Create query-document pairs, filter out empty documents
            pairs = [[query, doc if doc else " "] for doc in documents]
            
            # Compute scores
            scores = reranker.compute_score(pairs, normalize=True)
            
            # Ensure scores is a list
            if not isinstance(scores, list):
                scores = [scores]
            
            # Validate output length
            if len(scores) != len(documents):
                raise RuntimeError(f"Reranker returned {len(scores)} scores for {len(documents)} documents")
            
            return scores


class MultiThreadedRetriever:
    def __init__(
        self,
        COLLECTION_NAME,
        qdrant_client,
        max_workers: int = 4,
        top_k: int = 5,
        queries_per_batch: int = 10,
        unique_per_filename: bool = True,  # New parameter to control grouping
        use_reranker: bool = True,  # Enable reranking with BGE reranker
    ):
        self.qdrant_client = qdrant_client
        self.collection_name = COLLECTION_NAME
        self.max_workers = max_workers
        self.top_k = top_k
        self.queries_per_batch = queries_per_batch
        self.unique_per_filename = unique_per_filename  # Enable/disable unique filename filtering
        self.use_reranker = use_reranker  # Enable/disable BGE reranking
        self.lock = Lock()

        print("Initializing shared embedding model for retrieval...")
        self.shared_model = SharedEmbeddingModel()
        self.shared_model.initialize_model(self.qdrant_client)
        print("✅ Shared embedding model ready for retrieval")

        if self.use_reranker:
            print("Initializing shared reranking model...")
            self.reranker_model = SharedRerankingModel()
            self.reranker_model.initialize_model()
            print("✅ Shared reranking model ready")

        self._verify_collection()

    def _verify_collection(self):
        try:
            if not self.qdrant_client.collection_exists(self.collection_name):
                raise ValueError(
                    f"Collection '{self.collection_name}' does not exist. Please run the ingestion pipeline first."
                )

            collection_info = self.qdrant_client.get_collection(self.collection_name)
            point_count = self.qdrant_client.count(self.collection_name).count

            print(f"✅ Collection '{self.collection_name}' found:")
            print(f"   - Collection type: Hybrid search (dense + sparse + ColBERT late interaction)")
            print(f"   - Total points: {point_count}")
            print(f"   - Retrieval Pipeline:")
            print(f"      1. Hybrid search (dense + sparse prefetch)")
            print(f"      2. ColBERT late interaction scoring")
            print(f"      3. BGE Reranking: {'Enabled ✓' if self.use_reranker else 'Disabled ✗'}")
            print(f"      4. Unique filename filtering: {'Enabled ✓' if self.unique_per_filename else 'Disabled ✗'}")

        except Exception as e:
            raise RuntimeError(f"Error verifying collection: {e}")
    
    def _get_unique_by_filename(self, results_list, limit: int):
        """
        Group results by filename and keep only the highest scoring result per filename.
        
        Args:
            results_list: List of (result, score) tuples (already reranked if applicable)
            limit: Maximum number of unique filenames to return
            
        Returns:
            List of top (result, score) tuples with unique filenames
        """
        unique_results = {}
        
        for result, score in results_list:
            filename = result.payload.get("filename", "")
            
            # Skip empty filenames
            if not filename:
                continue
            
            # Keep the highest scoring result for each filename
            if filename not in unique_results or score > unique_results[filename]["score"]:
                unique_results[filename] = {
                    "result": result,
                    "score": score
                }
        
        # Sort by score (descending) and take top limit
        sorted_unique = sorted(
            unique_results.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:limit]
        
        return [(item["result"], item["score"]) for item in sorted_unique]

    def load_queries(self, queries_file_path: str) -> List[Dict[str, str]]:
        try:
            with open(queries_file_path, "r", encoding="utf-8") as f:
                queries = json.load(f)

            print(f"✅ Loaded {len(queries)} queries from {queries_file_path}")
            return queries
        except Exception as e:
            raise RuntimeError(f"Error loading queries from {queries_file_path}: {e}")

    def _create_query_batches(
        self, queries: List[Dict[str, str]]
    ) -> List[List[Dict[str, str]]]:
        batches = [
            queries[i : i + self.queries_per_batch]
            for i in range(0, len(queries), self.queries_per_batch)
        ]

        print(
            f"✅ Created {len(batches)} query batches ({self.queries_per_batch} queries per batch)"
        )
        return batches

    def _process_single_query(self, query_data: Dict[str, str]) -> Dict[str, Any]:
        query_num = query_data["query_num"]
        query_text = query_data["query"]

        try:
            # Generate all three types of embeddings for the query
            dense_embeddings, sparse_embeddings, late_interaction_embeddings = self.shared_model.embed_documents([query_text])
            
            # Extract single query embeddings
            dense_vector = dense_embeddings[0]
            sparse_vector = sparse_embeddings[0]
            late_interaction_vector = late_interaction_embeddings[0]

            # Hybrid search with prefetch (dense + sparse), then rerank with ColBERT
            # Fetch more results than needed to ensure unique filenames after grouping
            # Prefetch must be >= fetch_limit (funnel architecture)

            prefetch_limit = self.top_k * 5   # Get 5x candidates
            fetch_limit = self.top_k * 3      # ColBERT narrows to 3x
            # Then unique filter gets top_k
            
            prefetch = [
                models.Prefetch(
                    query=dense_vector,
                    using="dense_embedding",
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(**sparse_vector.as_object()),
                    using="sparse_embedding",
                    limit=prefetch_limit,
                ),
            ]

            # Perform hybrid search with late interaction
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=late_interaction_vector,
                using="reranker",
                with_payload=True,
                limit=fetch_limit,  # Fetch more results for reranking
            )

            # Check if any results were found
            if not search_results.points:
                print(f"Query {query_num}: No results found in collection")
                return {
                    "query_num": query_num,
                    "query": query_text,
                    "response": [],
                    "message": "No matching documents found"
                }

            # Apply BGE reranking if enabled
            if self.use_reranker:
                # Extract documents and their metadata
                documents = [result.payload.get("text", "") for result in search_results.points]
                
                # Handle case when no documents are found
                if not documents:
                    print(f"Query {query_num}: No documents to rerank")
                    reranked_results = []
                else:
                    # Compute reranking scores using BGE reranker
                    rerank_scores = self.reranker_model.compute_scores(query_text, documents)
                    
                    # Ensure we have the same number of scores as results
                    if len(rerank_scores) != len(search_results.points):
                        print(f"Warning: Score count mismatch for query {query_num}")
                        # Fall back to original scores
                        reranked_results = [
                            (result, float(result.score))
                            for result in search_results.points
                        ]
                    else:
                        # Combine results with reranked scores
                        reranked_results = [
                            (result, float(score))
                            for result, score in zip(search_results.points, rerank_scores)
                        ]
                        
                        # Sort by reranked scores (descending)
                        reranked_results.sort(key=lambda x: x[1], reverse=True)
                        
                        # print(f"Query {query_num}: Reranked {len(reranked_results)} results with BGE")
            else:
                # Use original scores from Qdrant
                reranked_results = [
                    (result, float(result.score))
                    for result in search_results.points
                ]

            # Apply unique filename filtering if enabled
            if self.unique_per_filename:
                final_results = self._get_unique_by_filename(reranked_results, self.top_k)
            else:
                # Just take top_k results without grouping
                final_results = reranked_results[:self.top_k]
            
            # Format results
            chunks = []
            for i, (result, score) in enumerate(final_results, 1):
                chunk_data = {
                    f"chunk_{i}_text": result.payload.get("text", ""),
                    f"chunk_{i}_filename": result.payload.get("filename", ""),
                    f"chunk_{i}_score": float(score),
                    f"chunk_{i}_chunk_id": result.id,
                }
                chunks.append(chunk_data)
            
            files = [result.payload.get("filename", "") for result, _ in final_results]
            
            # Create result structure
            result_data = {
                "query_num": query_num,
                "query": query_text,
                "response": files,
            }

            # Add chunk data to results
            for chunk in chunks:
                result_data.update(chunk)

            return result_data

        except Exception as e:
            print(f"❌ Error processing query {query_num}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "query_num": query_num,
                "query": query_text,
                "error": str(e),
                "retrieval_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

    def _process_query_batch(
        self, batch_info: Tuple[int, List[Dict[str, str]]]
    ) -> List[Dict[str, Any]]:
        batch_id, queries = batch_info
        results = []

        # print(f"Worker {batch_id}: Processing {len(queries)} queries...")

        for query_data in queries:
            result = self._process_single_query(query_data)
            results.append(result)

        # print(f"Worker {batch_id}: Completed {len(results)} queries")
        return results

    def process_queries(
        self, queries_file_path: str, output_file_path: str = None
    ) -> List[Dict[str, Any]]:
        print("=" * 60)
        print("STARTING MULTITHREADED QUERY RETRIEVAL")
        print("=" * 60)

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
            with tqdm(
                total=len(query_batches), desc="Processing query batches"
            ) as pbar:
                for future in as_completed(future_to_batch):
                    batch_info = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)

                        # Update progress
                        pbar.set_postfix(
                            {
                                "completed": len(all_results),
                                "total_queries": len(queries),
                            }
                        )

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
        print("\n" + "=" * 60)
        print("RETRIEVAL COMPLETED")
        print("=" * 60)
        successful = len([r for r in all_results if "error" not in r])
        failed = len(all_results) - successful

        print(f"Total queries processed: {len(all_results)}")
        print(f"Successful retrievals: {successful}")
        print(f"Failed retrievals: {failed}")
        print(f"Top-k per query: {self.top_k}")

        return all_results

    def _save_results(self, results: List[Dict[str, Any]], output_file_path: str):
        try:
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"✅ Results saved to: {output_file_path}")

        except Exception as e:
            print(f"❌ Error saving results: {e}")


def run_multithreaded_retrieval(
    COLLECTION_NAME,
    qdrant_client,
    queries_file_path: str,
    output_file_path: str = None,
    max_workers: int = 16,
    top_k: int = 5,
    queries_per_batch: int = 20,
    unique_per_filename: bool = True,  # Enable unique filename filtering
    use_reranker: bool = True,  # Enable BGE reranker
) -> List[Dict[str, Any]]:
    """
    Run multithreaded retrieval on queries with hybrid search and optional BGE reranking.
    
    Retrieval Pipeline:
    1. Hybrid search with dense + sparse embeddings (prefetch)
    2. ColBERT late interaction scoring (Qdrant native)
    3. BGE reranker scoring (optional, improves relevance)
    4. Unique filename filtering (optional, deduplicates by file)
    
    Args:
        COLLECTION_NAME: Name of the Qdrant collection
        qdrant_client: Qdrant client instance
        queries_file_path: Path to queries JSON file
        output_file_path: Optional path to save results
        max_workers: Number of concurrent workers
        top_k: Number of results to return per query
        queries_per_batch: Number of queries per batch
        unique_per_filename: If True, returns only the highest scoring chunk per unique filename
                            If False, returns top_k chunks regardless of filename
        use_reranker: If True, applies BGE reranker after ColBERT scoring
                     If False, uses only Qdrant's hybrid search scores
    
    Returns:
        List of retrieval results
    """
    retriever = MultiThreadedRetriever(
        COLLECTION_NAME=COLLECTION_NAME,
        qdrant_client=qdrant_client,
        max_workers=max_workers,
        top_k=top_k,
        queries_per_batch=queries_per_batch,
        unique_per_filename=unique_per_filename,
        use_reranker=use_reranker,
    )

    return retriever.process_queries(queries_file_path, output_file_path)
