import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any
from threading import Lock
import uuid
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from rag.qdrant import client

COLLECTION_NAME = "ps04-fragmented"

# Initialize collection if it doesn't exist
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )


class MultiThreadedEmbeddingStore:
    def __init__(self, max_workers=4, batch_size=100):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.lock = Lock()
        self.point_id_counter = 0

    def _get_next_point_id(self) -> int:
        """Get next unique point ID in a thread-safe manner."""
        with self.lock:
            self.point_id_counter += 1
            return self.point_id_counter

    def _store_embedding_batch(self, embeddings_batch: List[Tuple[List[float], Dict[str, Any]]]) -> bool:
        """Store a batch of embeddings to Qdrant."""
        try:
            points = []
            for embedding, payload in embeddings_batch:
                point_id = self._get_next_point_id()
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )
            
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            return True
        except Exception as e:
            print(f"Error storing embedding batch: {e}")
            return False

    def store_embeddings_multithreaded(self, embeddings_with_payloads: List[Tuple[List[float], Dict[str, Any]]]) -> int:
        """Store embeddings in parallel batches."""
        if not embeddings_with_payloads:
            return 0

        # Group embeddings into batches
        batches = [
            embeddings_with_payloads[i:i + self.batch_size] 
            for i in range(0, len(embeddings_with_payloads), self.batch_size)
        ]
        
        successful_batches = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches for storage
            future_to_batch = {
                executor.submit(self._store_embedding_batch, batch): batch 
                for batch in batches
            }
            
            # Process completed batches as they finish
            with tqdm(total=len(batches), desc="Storing embeddings") as pbar:
                for future in as_completed(future_to_batch):
                    try:
                        if future.result():
                            successful_batches += 1
                    except Exception as e:
                        print(f"Error processing storage batch: {e}")
                    finally:
                        pbar.update(1)
        
        total_stored = successful_batches * self.batch_size
        if len(embeddings_with_payloads) % self.batch_size != 0:
            # Adjust for last partial batch
            total_stored = successful_batches * self.batch_size
            if successful_batches == len(batches):  # All batches succeeded
                total_stored = len(embeddings_with_payloads)
        
        print(f"Successfully stored {total_stored} embeddings out of {len(embeddings_with_payloads)}")
        return total_stored

