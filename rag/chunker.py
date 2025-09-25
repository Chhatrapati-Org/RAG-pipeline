import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from threading import Lock
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.utils.math import cosine_similarity
from tqdm import tqdm


class MultiThreadedChunker:
    def __init__(self, max_workers=4, model_name="jinaai/jina-embeddings-v2-small-en"):
        self.max_workers = max_workers
        self.model_name = model_name
        self.model = None
        self.lock = Lock()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}

        self.model = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def _embed_chunk_batch(self, chunks_batch: List[Tuple[str, str, int]]) -> List[Tuple[List[float], dict]]:
        """Embed a batch of chunks and return embeddings with metadata."""
        try:
            texts = [chunk[0] for chunk in chunks_batch]
            embeddings = self.model.embed_documents(texts)
            
            results = []
            for i, (chunk_text, filename, chunk_id) in enumerate(chunks_batch):
                payload = {
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "chunk_size": len(chunk_text.encode("utf-8")),
                }
                results.append((embeddings[i], payload))
            
            return results
        except Exception as e:
            print(f"Error embedding batch: {e}")
            return []

    def process_chunks_multithreaded(self, chunks: List[Tuple[str, str, int]], batch_size=32) -> List[Tuple[List[float], dict]]:
        """Process chunks in parallel batches for embedding."""
        if not chunks:
            return []

        # Group chunks into batches for efficient embedding
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches for processing
            future_to_batch = {
                executor.submit(self._embed_chunk_batch, batch): batch 
                for batch in batches
            }
            
            # Process completed batches as they finish
            with tqdm(total=len(batches), desc="Embedding chunks") as pbar:
                for future in as_completed(future_to_batch):
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                    finally:
                        pbar.update(1)
        
        return all_results

    def chunker(self, chunks: List[Tuple[str, str, int]]) -> List[Tuple[List[float], dict]]:
        """Main chunker function that processes chunks and returns embeddings with payloads."""
        return self.process_chunks_multithreaded(chunks)

