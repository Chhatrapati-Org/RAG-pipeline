import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from rag.pipeline import COLLECTION_NAME, SharedEmbeddingModel

COLLECTION_NAME = COLLECTION_NAME


class MultiThreadedRetriever:
    def __init__(
        self,
        qdrant_client,
        max_workers: int = 4,
        top_k: int = 5,
        queries_per_batch: int = 10,
    ):
        self.qdrant_client = qdrant_client
        self.max_workers = max_workers
        self.top_k = top_k
        self.queries_per_batch = queries_per_batch
        self.lock = Lock()

        print("Initializing shared embedding model for retrieval...")
        self.shared_model = SharedEmbeddingModel()
        self.shared_model.initialize_model(self.qdrant_client)
        print("✅ Shared embedding model ready for retrieval")

        self._verify_collection()

    def _verify_collection(self):
        try:
            if not self.qdrant_client.collection_exists(COLLECTION_NAME):
                raise ValueError(
                    f"Collection '{COLLECTION_NAME}' does not exist. Please run the ingestion pipeline first."
                )

            collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)
            vector_size = collection_info.config.params.vectors.size
            point_count = self.qdrant_client.count(COLLECTION_NAME).count

            print(f"✅ Collection '{COLLECTION_NAME}' found:")
            print(f"   - Vector dimensions: {vector_size}")
            print(f"   - Total points: {point_count}")

        except Exception as e:
            raise RuntimeError(f"Error verifying collection: {e}")

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
            # Encode the query using shared model
            query_embedding = self.shared_model.embed_documents([query_text])[0]

            # Search for similar chunks in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=self.top_k,
                with_payload=True,
                with_vectors=False,  # We don't need the vectors in results
            )

            # Format results
            chunks = []
            for i, result in enumerate(search_results, 1):
                chunk_data = {
                    f"chunk_{i}_text": result.payload.get("text", ""),
                    f"chunk_{i}_filename": result.payload.get("filename", ""),
                    f"chunk_{i}_score": float(result.score),
                    f"chunk_{i}_chunk_id": result.id,
                }
                chunks.append(chunk_data)
            files = [result.payload.get("filename", "") for result in search_results]
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

        print(f"Worker {batch_id}: Processing {len(queries)} queries...")

        for query_data in queries:
            result = self._process_single_query(query_data)
            results.append(result)

        print(f"Worker {batch_id}: Completed {len(results)} queries")
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
    qdrant_client,
    queries_file_path: str,
    output_file_path: str = None,
    max_workers: int = 16,
    top_k: int = 5,
    queries_per_batch: int = 20,
) -> List[Dict[str, Any]]:
    retriever = MultiThreadedRetriever(
        qdrant_client=qdrant_client,
        max_workers=max_workers,
        top_k=top_k,
        queries_per_batch=queries_per_batch,
    )

    return retriever.process_queries(queries_file_path, output_file_path)
