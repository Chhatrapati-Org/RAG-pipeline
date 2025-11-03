import datetime
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, Generator, List, Tuple
import math

import torch
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_community.utils.math import cosine_similarity
from nltk.tokenize import sent_tokenize
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams, Modifier, MultiVectorConfig, MultiVectorComparator, HnswConfigDiff
from tqdm import tqdm
import onnxruntime as ort

from rag.parse_json import parser
from rag.preprocess import preprocess_chunk_text

now = datetime.datetime.now()
formatted_now = now.strftime("%d-%m-%Y %H %M")
COLLECTION_NAME = "ps04_" + formatted_now


class SharedEmbeddingModel:
    _instance = None
    _dense_model = None
    _sparse_model = None
    _late_interaction_model = None
    _lock = Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SharedEmbeddingModel, cls).__new__(cls)
        return cls._instance

    def initialize_model(
        self, 
        qdrant_client,
        dense_model_name: str = "BAAI/bge-base-en-v1.5",
        sparse_model_name: str = "Qdrant/bm25", #"prithivida/Splade_PP_en_v1",
        late_interaction_model_name: str = "colbert-ir/colbertv2.0", #"jinaai/jina-colbert-v2"
    ):
        with self._lock:
            if self._initialized:
                return

            print("Initializing hybrid embedding models...")
            print(f"  Dense: {dense_model_name}")
            print(f"  Sparse: {sparse_model_name}")
            print(f"  late_interaction: {late_interaction_model_name}")

            try:
                # Initialize dense embedding model
                available_providers = ort.get_available_providers()
                print(f"Available ONNX providers: {available_providers}")
                
                # Set providers - prefer CUDA
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    print("Using CUDA for acceleration")
                else:
                    providers = ['CPUExecutionProvider']
                    print("CUDA not available, using CPU")
                    
                self._dense_model = TextEmbedding(
                    model_name=dense_model_name,
                    providers=providers,  # Use GPU if available
                    cache_dir="./model_cache"
                )
                
                self._sparse_model = SparseTextEmbedding(
                    model_name=sparse_model_name,
                    providers=providers,  # Use GPU if available
                    cache_dir="./model_cache"
                )
                
                self._late_interaction_model = LateInteractionTextEmbedding(
                    model_name=late_interaction_model_name,
                    providers=providers,  # Use GPU if available
                    cache_dir="./model_cache"
                )
                # Test embeddings to get dimensions
                test_text = ["Test sentence"]
                test_dense = list(self._dense_model.embed(test_text))[0]
                test_sparse = list(self._sparse_model.embed(test_text))[0]
                test_late_interaction = list(self._late_interaction_model.embed(test_text))[0]

                dense_dim = len(test_dense)
                late_interaction_dim = len(test_late_interaction[0])  # ColBERT produces multi-vectors
                
                print(f"✅ Dense vectors: {dense_dim} dimensions")
                print(f"✅ late_interaction vectors: {late_interaction_dim} dimensions (multi-vector)")

                initialize_collection_if_needed(qdrant_client, dense_dim, late_interaction_dim)

                self._initialized = True
                print("✅ All embedding models ready")

            except Exception as e:
                print(f"❌ Failed to initialize embedding models: {e}")
                raise RuntimeError("Cannot initialize embedding models")

    def get_dense_model(self):
        if not self._initialized or self._dense_model is None:
            raise RuntimeError("Models not initialized. Call initialize_model() first.")
        return self._dense_model

    def get_sparse_model(self):
        if not self._initialized or self._sparse_model is None:
            raise RuntimeError("Models not initialized. Call initialize_model() first.")
        return self._sparse_model

    def get_late_interaction_model(self):
        if not self._initialized or self._late_interaction_model is None:
            raise RuntimeError("Models not initialized. Call initialize_model() first.")
        return self._late_interaction_model

    def embed_documents(self, texts: List[str]) -> Tuple[List[List[float]], List[Any], List[List[List[float]]]]:
        """
        Generate all three types of embeddings for documents.
        Returns: (dense_embeddings, sparse_embeddings, late_interaction_embeddings)
        """
        with self._lock:
            dense_model = self.get_dense_model()
            sparse_model = self.get_sparse_model()
            late_interaction_model = self.get_late_interaction_model()
            
            # Generate embeddings
            dense_embeddings = list(dense_model.embed(texts))
            sparse_embeddings = list(sparse_model.embed(texts))
            late_interaction_embeddings = list(late_interaction_model.embed(texts))
            
            return dense_embeddings, sparse_embeddings, late_interaction_embeddings


    def embed_sentences(self, texts: List[str]) -> List[List[float]]:
        """
        Generate only dense embeddings for sentences (used in semantic chunking).
        Returns: dense_embeddings (List of embedding vectors)
        """
        with self._lock:
            dense_model = self.get_dense_model()
            
            # Generate embeddings
            dense_embeddings = list(dense_model.embed(texts))
            
            return dense_embeddings

def initialize_collection_if_needed(client, dense_dim: int, late_interaction_dim: int):
    """Initialize Qdrant collection with hybrid search configuration."""
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating Qdrant collection '{COLLECTION_NAME}' with hybrid search support")
        print(f"  Dense embedding: {dense_dim}D")
        print(f"  late_interaction embedding: {late_interaction_dim}D (multi-vector)")
        
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense_embedding": VectorParams(
                    size=dense_dim,
                    distance=Distance.COSINE,
                ),
                "late_interaction": VectorParams(
                    size=late_interaction_dim,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=HnswConfigDiff(m=0)  # Disable HNSW for reranking
                ),
            },
            sparse_vectors_config={
                "sparse_embedding": SparseVectorParams(
                    modifier=Modifier.IDF
                )
            }
        )
        print("✅ Hybrid search collection created")
    else:
        try:
            collection_info = client.get_collection(COLLECTION_NAME)
            print(f"✅ Collection '{COLLECTION_NAME}' already exists")
        except Exception as e:
            print(f"Warning: Error checking collection: {e}")
        except Exception as e:
            print(f"Error checking collection: {e}")


def lazy_read(file_handle, chunk_size_kb=4):
    chunk_size_bytes = chunk_size_kb * 1024

    while True:
        chunk = file_handle.read(chunk_size_bytes)
        if not chunk:
            break
        yield chunk


class MergedRAGWorker:
    def __init__(
        self,
        qdrant_client,
        worker_id: int,
        chunk_size_kb: int = 4, # FIXME: currently needs to be integer, change to float
        shared_model: SharedEmbeddingModel = None,
    ):
        self.worker_id = worker_id
        self.chunk_size_bytes = chunk_size_kb * 1024
        self.shared_model = shared_model if shared_model is not None else SharedEmbeddingModel()
        self.qdrant_client = qdrant_client
        self.point_id_counter = worker_id * 10000  # Unique ID range per worker
        # print(f"Worker {self.worker_id}: Initialized with shared model")

    def _read_txt_file_chunks(
        self, file_path: str
    ) -> Generator[Tuple[str, str, int], None, None]:
        filename = os.path.basename(file_path)

        # TODO: Change chunking method here
        # TODO: Remove threshold from payload in subsequent functions if not using semantic chunking or special json chunking
        try:
            yield from self._special_json_chunking(
                file_path
            )  # switch to semantic or fixed_size.
            return

        except Exception as e:
            print(f"Worker {self.worker_id}: Error reading {filename}: {e}")

    def _simple_chunking(
        self, file_path: str
    ) -> Generator[Tuple[str, str, int], None, None]:
        filename = os.path.basename(file_path)
        try:
            chunk_id = 0
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                yield (f.read(), filename, chunk_id)
        except Exception as e:
            print(f"Worker {self.worker_id}: Error reading {filename}: {e}")

    def _special_json_chunking(
        self, file_path: str
    ) -> Generator[Tuple[str, str, int], None, None]:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                chunk_id = 0
                for chunk in lazy_read(f, chunk_size_kb=4):
                    chunk = " " + chunk
                    root = parser(chunk)
                    
                    def chunks_in(node, chunk):
                        if node.children:
                            lst = []
                            for child in node.children:
                                lst.extend(chunks_in(child, chunk))
                            return lst
                        else:
                            part = chunk[node.start+1:node.end]
                            return [part]
                    
                    chunklets = chunks_in(root, chunk)
                    for chunk_to_yield in chunklets:
                        chunk_to_yield = re.sub(r"[\{\}\[\]]", " ", chunk_to_yield) # remove REMAINING brackets
                        chunk_to_yield = re.sub(r"\s+", " ", chunk_to_yield) # remove REMAINING whitespace
                        if len(chunk_to_yield.strip()) > 20:  # Yield only non-empty chunks
                            if len(chunk_to_yield.encode('utf-8')) > self.chunk_size_bytes:
                                yield from self._semantic_chunking_logic(chunk_to_yield, filename, chunk_id)
                                chunk_id += 100
                            else:
                                yield (chunk_to_yield, filename, chunk_id)
                                chunk_id += 1

        except Exception as e:
            print(f"Worker {self.worker_id}: Error reading {filename}: {e}")

    def _fixed_size_chunking(
        self, file_path: str
    ) -> Generator[Tuple[str, str, int], None, None]:
        filename = os.path.basename(file_path)
        num_sent = 5
        overlap = 2
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                chunk_id = 0
                for chunk in lazy_read(f, chunk_size_kb=self.chunk_size_bytes // 1024):
                    sentences = []
                    for sentence in sent_tokenize(chunk):
                        sentences.append(sentence.strip())
                    big_sentences = []
                    for i in range(len(sentences)):
                        if i == len(sentences) - 1:
                            big_sentences.append(sentences[i])
                            break
                        if len(sentences[i]) < 25:
                            sentences[i + 1] = sentences[i] + " " + sentences[i + 1]
                        else:
                            big_sentences.append(sentences[i])

                    sentences = big_sentences
                    i = 0
                    while True:
                        chunk_to_yield = " ".join(sentences[i : i + num_sent])
                        yield (chunk_to_yield, filename, chunk_id)
                        chunk_id += 1
                        i += num_sent - overlap
                        if i >= len(sentences):
                            break

        except Exception as e:
            print(f"Worker {self.worker_id}: Error reading {filename}: {e}")

    def _semantic_chunking(
        self, file_path: str
    ) -> Generator[Tuple[str, str, int], None, None]:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                chunk_id = 0
                # Lazily create chunks for memory and time efficiency
                for chunk in lazy_read(f, chunk_size_kb=self.chunk_size_bytes // 1024):
                    yield from self._semantic_chunking_logic(chunk, filename, chunk_id)
                    chunk_id += 100
        except Exception as e:
            print(f"Error reading {filename}: {e}")


    def _semantic_chunking_logic(self, chunk: str, filename: str, chunk_id: int) -> Generator[Tuple[str, str, int], None, None]:
        sentences = []
        sentences = [x.strip() for x in sent_tokenize(chunk)]
        big_sentences = []
        # Combine short sentences
        for i in range(len(sentences)):
            if i == len(sentences) - 1:
                big_sentences.append(sentences[i])
                break
            if len(sentences[i]) < 30:
                sentences[i + 1] = sentences[i] + " " + sentences[i + 1]
            else:
                big_sentences.append(sentences[i])

        sentences = big_sentences

        embeddings = []
        combined_sentences = []
        combined_sentences.append(sentences[0])
        distances = [0]
        for i in range(1, len(sentences)):
            combined_sentences.append(sentences[i - 1] + sentences[i])
        # We combine two sentences to get better context for similarity
        for i in range(1, len(sentences)):
            # Use only dense embeddings for semantic similarity comparison
            dense_embeddings = self.shared_model.embed_sentences(combined_sentences)
            current = dense_embeddings[i]
            prev = dense_embeddings[i - 1]

            similarity = cosine_similarity([prev],[current])[0][0]
            distances.append(1- similarity)
        breakpoint_distance_threshold = 0.25 # Tuned for BGE embeddings but can be increased a bit
        indices_above_thresh = [i for i,x in enumerate(distances) if x > breakpoint_distance_threshold]
        # No breakpoints found - yield as single chunk or split if too large
        if len(indices_above_thresh) == 0:
            if len(chunk.encode('utf-8')) <= self.chunk_size_bytes:
                yield (chunk, filename, chunk_id)
            else:
                for chunk_to_yield in self._split_large_text(chunk, self.chunk_size_bytes):
                    yield (chunk_to_yield, filename, chunk_id)
                    chunk_id += 1
            return
        # Creating chunks based on detected breakpoints
        o=0
        for i in range(len(indices_above_thresh)):
            chunk_to_yield = " ".join(sentences[o:indices_above_thresh[i]])
            if len(chunk_to_yield.encode('utf-8')) <= self.chunk_size_bytes:
                yield (chunk_to_yield, filename, chunk_id)
                chunk_id += 1
            else:
                for chunk in self._split_large_text(chunk_to_yield, self.chunk_size_bytes):
                    yield (chunk, filename, chunk_id)
                    chunk_id += 1

            o = indices_above_thresh[i]
            
        if o < len(sentences):
            chunk_to_yield = " ".join(sentences[o:len(sentences)])
            if len(chunk_to_yield.encode('utf-8')) <= self.chunk_size_bytes:
                yield (chunk_to_yield, filename, chunk_id)
            else:
                for chunk in self._split_large_text(chunk_to_yield, self.chunk_size_bytes):
                    yield (chunk, filename, chunk_id)
                    chunk_id += 1
        

    def _split_large_text(self, text: str, max_size: int) -> Generator[str, None, None]:
        text = text.strip()
        if not text:
            return

        while text:
            size = len(text.encode("utf-8"))
            if size <= max_size:
                yield text
                break

            split_point = size // math.ceil(size / max_size)
            temp_text = text[:split_point]
            sentence_end = max(
                temp_text.rfind(". "), temp_text.rfind("! "), temp_text.rfind("? ")
            )
            if sentence_end > max_size // 2:
                split_point = sentence_end + 1
            else:
                word_boundary = temp_text.rfind(" ")
                if word_boundary > max_size // 2:
                    split_point = word_boundary

            chunk = text[:split_point].strip()
            if chunk:
                yield chunk

            text = text[split_point:].strip()

    def _embed_chunks(
        self, chunks: List[Tuple[str, str, int]]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Generate hybrid embeddings (dense, sparse, late_interaction) for chunks.
        Returns: List of (embeddings_dict, payload_dict) tuples
        """
        if not chunks:
            return []

        try:
            texts = [chunk[0] for chunk in chunks]

            # Merge small texts
            valid_chunks = []
            prefix = ""
            for i, chunk in enumerate(chunks):
                text = prefix + chunk[0]
                if len(text.strip()) < 75:
                    prefix = text
                else:
                    valid_chunks.append((text, chunk[1], chunk[2]))
            if not valid_chunks:
                print(f"Worker {self.worker_id} for {chunks[0][1]}: No valid texts to embed")
                return []
            texts = [chunk[0] for chunk in valid_chunks]
            chunks = valid_chunks

            # print(f"Worker {self.worker_id}: Embedding {len(texts)} text chunks with hybrid models...")

            try:
                # Generate all three types of embeddings
                dense_embeddings, sparse_embeddings, late_interaction_embeddings = self.shared_model.embed_documents(texts)
                
                if not dense_embeddings or len(dense_embeddings) != len(texts):
                    print(
                        f"Worker {self.worker_id}: Embedding count mismatch - expected {len(texts)}, got {len(dense_embeddings) if dense_embeddings else 0}"
                    )
                    return []

            except RuntimeError as rt_error:
                if "meta tensor" in str(rt_error).lower():
                    print(
                        f"Worker {self.worker_id}: Meta tensor error detected with shared model: {rt_error}"
                    )
                    raise rt_error
                else:
                    raise rt_error

            results = []
            for i, (chunk_text, filename, chunk_id) in enumerate(chunks):
                # Create embeddings dictionary with all three types
                embeddings_dict = {
                    "dense_embedding": dense_embeddings[i],
                    "sparse_embedding": sparse_embeddings[i],
                    "late_interaction": late_interaction_embeddings[i]
                }
                
                payload = {
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "chunk_size": len(chunk_text.encode("utf-8")),
                    "worker_id": self.worker_id,
                }
                results.append((embeddings_dict, payload))

            # print(
            #     f"Worker {self.worker_id}: Successfully embedded {len(results)} chunks with hybrid embeddings"
            # )
            return results

        except Exception as e:
            print(f"Worker {self.worker_id}: Error embedding chunks: {e}")
            print(f"Worker {self.worker_id}: Error type: {type(e).__name__}")

            return []

    def _store_embeddings(
        self, embeddings_with_payloads: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> int:
        """
        Store hybrid embeddings (dense, sparse, late_interaction) in Qdrant.
        """
        if not embeddings_with_payloads:
            return 0

        try:
            points = []
            for embeddings_dict, payload in embeddings_with_payloads:
                self.point_id_counter += 1

                # Convert sparse embedding to proper format
                sparse_embedding = embeddings_dict["sparse_embedding"]
                
                points.append(
                    PointStruct(
                        id=self.point_id_counter,
                        vector={
                            "dense_embedding": embeddings_dict["dense_embedding"],
                            "sparse_embedding": sparse_embedding.as_object(),
                            "late_interaction": embeddings_dict["late_interaction"]
                        },
                        payload=payload,
                    )
                )

            if not points:
                print(f"Worker {self.worker_id}: No valid points to store")
                return 0

            self.qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )

            # print(
            #     f"Worker {self.worker_id}: Successfully stored {len(points)} hybrid embeddings"
            # )
            return len(points)

        except Exception as e:
            print(f"Worker {self.worker_id}: Error storing embeddings: {e}")
            return 0

    def process_file_batch(self, file_paths: List[str]) -> Dict[str, int]:
        stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0,
        }

        for file_path in file_paths:
            try:

                filename = os.path.basename(file_path)
                _, ext = os.path.splitext(filename)
                
                chunks_generator = self._read_txt_file_chunks(file_path)
                chunks = list(chunks_generator)
                if not chunks:
                    continue

                stats["chunks_created"] += len(chunks)

                embeddings_with_payloads = self._embed_chunks(chunks)
                if not embeddings_with_payloads:
                    stats["errors"] += 1
                    continue

                stats["embeddings_generated"] += len(embeddings_with_payloads)

                stored_count = self._store_embeddings(embeddings_with_payloads)
                stats["embeddings_stored"] += stored_count
                
                if stored_count > 0:
                    stats["files_processed"] += 1
                else:
                    stats["errors"] += 1

            except Exception as e:
                print(f"Worker {self.worker_id}: Error processing {file_path}: {e}")
                stats["errors"] += 1

        return stats

    def process_chunk_batch(
        self, chunk_batch: List[Tuple[str, str, int]]
    ) -> Dict[str, int]:
        stats = {
            "chunks_processed": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0,
        }

        if not chunk_batch:
            return stats

        try:
            # print(
            #     f"Worker {self.worker_id}: Processing batch of {len(chunk_batch)} chunks"
            # )

            embeddings_with_payloads = self._embed_chunks(chunk_batch)
            if not embeddings_with_payloads:
                stats["errors"] = len(chunk_batch)
                return stats

            stats["embeddings_generated"] = len(embeddings_with_payloads)

            stored_count = self._store_embeddings(embeddings_with_payloads)
            stats["embeddings_stored"] = stored_count

            if stored_count > 0:
                stats["chunks_processed"] = len(chunk_batch)
            else:
                stats["errors"] = len(chunk_batch)

        except Exception as e:
            print(f"Worker {self.worker_id}: Error processing chunk batch: {e}")
            stats["errors"] = len(chunk_batch)

        return stats

    def read_all_file_chunks(self, file_paths: List[str]) -> List[Tuple[str, str, int]]:
        all_chunks = []

        for file_path in file_paths:
            try:
                filename = os.path.basename(file_path)
                _, ext = os.path.splitext(filename)
                
                chunks_generator = self._read_txt_file_chunks(file_path)
                
                chunks = list(chunks_generator)
                all_chunks.extend(chunks)

                # print(
                #     f"Worker {self.worker_id}: Extracted {len(chunks)} chunks from {filename}"
                # )

            except Exception as e:
                print(f"Worker {self.worker_id}: Error reading {file_path}: {e}")
                continue

        return all_chunks


class ChunkBasedMultiThreadedRAGPipeline:
    def __init__(
        self,
        qdrant_client,
        max_workers: int = 4,
        chunk_size_kb: int = 4,
        chunks_per_batch: int = 50,
    ):
        self.max_workers = max_workers
        self.chunk_size_kb = chunk_size_kb
        self.chunks_per_batch = chunks_per_batch
        self.qdrant_client = qdrant_client
        self.lock = Lock()
        self.total_stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "chunks_processed": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0,
        }

        print("Initializing shared embedding model for chunk-based processing...")
        self.shared_model = SharedEmbeddingModel()
        self.shared_model.initialize_model(self.qdrant_client)
        print("Shared embedding model initialized successfully")

    def _read_all_files_to_chunks(
        self, directory_path: str
    ) -> List[Tuple[str, str, int]]:
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist")

        file_paths = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                file_paths.append(file_path)

        if not file_paths:
            print("No files found to process")
            return []

        print(f"Reading chunks from {len(file_paths)} files...")

        worker = MergedRAGWorker(
            qdrant_client=self.qdrant_client,
            worker_id=0,
            chunk_size_kb=self.chunk_size_kb,
            shared_model=self.shared_model,
        )

        all_chunks = worker.read_all_file_chunks(file_paths)

        self.total_stats["files_processed"] = len(file_paths)
        self.total_stats["chunks_created"] = len(all_chunks)

        print(f"Extracted {len(all_chunks)} total chunks from {len(file_paths)} files")
        return all_chunks

    def _create_chunk_batches(
        self, chunks: List[Tuple[str, str, int]]
    ) -> List[List[Tuple[str, str, int]]]:
        if not chunks:
            return []

        batches = [
            chunks[i : i + self.chunks_per_batch]
            for i in range(0, len(chunks), self.chunks_per_batch)
        ]

        print(
            f"Created {len(batches)} chunk batches ({self.chunks_per_batch} chunks per batch)"
        )
        return batches

    def _update_stats(self, worker_stats: Dict[str, int]):
        with self.lock:
            for key, value in worker_stats.items():
                if key in self.total_stats:
                    self.total_stats[key] += value

    def _process_chunk_batch_with_worker(
        self, batch_info: Tuple[int, List[Tuple[str, str, int]]]
    ) -> Dict[str, int]:
        worker_id, chunk_batch = batch_info
        worker = MergedRAGWorker(
            qdrant_client=self.qdrant_client,
            worker_id=worker_id,
            chunk_size_kb=self.chunk_size_kb,
            shared_model=self.shared_model,
        )

        stats = worker.process_chunk_batch(chunk_batch)
        return stats

    def process_directory(self, directory_path: str) -> Dict[str, int]:
        print(
            f"Starting chunk-based multithreaded processing of directory: {directory_path}"
        )
        print(
            f"Configuration: {self.max_workers} workers, {self.chunks_per_batch} chunks per batch, {self.chunk_size_kb}KB max chunk size"
        )

        all_chunks = self._read_all_files_to_chunks(directory_path)
        if not all_chunks:
            return self.total_stats

        chunk_batches = self._create_chunk_batches(all_chunks)
        if not chunk_batches:
            return self.total_stats

        batch_info = [(i, batch) for i, batch in enumerate(chunk_batches)]

        print(
            f"Processing {len(chunk_batches)} chunk batches with {self.max_workers} workers..."
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_chunk_batch_with_worker, info): info
                for info in batch_info
            }

            with tqdm(
                total=len(chunk_batches), desc="Processing chunk batches"
            ) as pbar:
                for future in as_completed(future_to_batch):
                    batch_info = future_to_batch[future]
                    try:
                        worker_stats = future.result()
                        self._update_stats(worker_stats)

                        pbar.set_postfix(
                            {
                                "chunks_processed": self.total_stats[
                                    "chunks_processed"
                                ],
                                "embedded": self.total_stats["embeddings_generated"],
                                "stored": self.total_stats["embeddings_stored"],
                            }
                        )

                    except Exception as e:
                        print(f"Error processing chunk batch {batch_info[0]}: {e}")

                        self.total_stats["errors"] += len(batch_info[1])
                    finally:
                        pbar.update(1)

        print("\n" + "=" * 60)
        print("CHUNK-BASED PROCESSING COMPLETED")
        print("=" * 60)
        for key, value in self.total_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        return self.total_stats


# Main pipeline with merged reading, chunking, embedding, and storage
class MergedMultiThreadedRAGPipeline:
    def __init__(
        self,
        qdrant_client,
        max_workers: int = 4,
        chunk_size_kb: int = 4,
        files_per_batch: int = 5,
    ):
        self.max_workers = max_workers
        self.chunk_size_kb = chunk_size_kb
        self.files_per_batch = files_per_batch
        self.qdrant_client = qdrant_client
        self.lock = Lock()
        self.total_stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0,
        }

        print("Initializing shared embedding model...")
        self.shared_model = SharedEmbeddingModel()

        self.shared_model.initialize_model(self.qdrant_client)
        print("Shared embedding model initialized successfully")

    def _create_file_batches(self, directory_path: str) -> List[List[str]]:
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist")

        file_paths = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                file_paths.append(file_path)

        if not file_paths:
            print("No files found to process")
            return []

        batches = [
            file_paths[i : i + self.files_per_batch]
            for i in range(0, len(file_paths), self.files_per_batch)
        ]

        return batches

    def _update_stats(self, worker_stats: Dict[str, int]):
        with self.lock:
            for key, value in worker_stats.items():
                self.total_stats[key] += value

    def _process_batch_with_worker(
        self, batch_info: Tuple[int, List[str]]
    ) -> Dict[str, int]:
        worker_id, file_paths = batch_info
        worker = MergedRAGWorker(
            qdrant_client=self.qdrant_client,
            worker_id=worker_id,
            chunk_size_kb=self.chunk_size_kb,
            shared_model=self.shared_model,
        )

        stats = worker.process_file_batch(file_paths)
        return stats

    def process_directory(self, directory_path: str) -> Dict[str, int]:
        print(
            f"Starting merged multithreaded processing of directory: {directory_path}"
        )
        print(
            f"Configuration: {self.max_workers} workers, {self.files_per_batch} files per batch, {self.chunk_size_kb}KB max chunk size"
        )

        file_batches = self._create_file_batches(directory_path)
        if not file_batches:
            return self.total_stats

        print(f"Created {len(file_batches)} batches from directory")

        batch_info = [(i, batch) for i, batch in enumerate(file_batches)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_batch_with_worker, info): info
                for info in batch_info
            }

            with tqdm(total=len(file_batches), desc="Processing file batches") as pbar:
                for future in as_completed(future_to_batch):
                    batch_info = future_to_batch[future]
                    try:
                        worker_stats = future.result()
                        self._update_stats(worker_stats)

                        pbar.set_postfix(
                            {
                                "files": self.total_stats["files_processed"],
                                "chunks": self.total_stats["chunks_created"],
                                "stored": self.total_stats["embeddings_stored"],
                            }
                        )

                    except Exception as e:
                        print(f"Error processing batch {batch_info[0]}: {e}")
                        self.total_stats["errors"] += len(batch_info[1])
                    finally:
                        pbar.update(1)

        print("\n" + "=" * 50)
        print("PROCESSING COMPLETED")
        print("=" * 50)
        for key, value in self.total_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        return self.total_stats


def run_merged_rag_pipeline(
    qdrant_client,
    directory_path: str,
    max_workers: int = 4,
    chunk_size_kb: int = 4,
    files_per_batch: int = 5,
) -> Dict[str, int]:
    pipeline = MergedMultiThreadedRAGPipeline(
        qdrant_client=qdrant_client,
        max_workers=max_workers,
        chunk_size_kb=chunk_size_kb,
        files_per_batch=files_per_batch,
    )

    return pipeline.process_directory(directory_path), COLLECTION_NAME


def run_chunk_based_rag_pipeline(
    qdrant_client,
    directory_path: str,
    max_workers: int = 4,
    chunk_size_kb: int = 4,
    chunks_per_batch: int = 50,
) -> Dict[str, int]:
    pipeline = ChunkBasedMultiThreadedRAGPipeline(
        qdrant_client=qdrant_client,
        max_workers=max_workers,
        chunk_size_kb=chunk_size_kb,
        chunks_per_batch=chunks_per_batch,
    )

    return pipeline.process_directory(directory_path)
