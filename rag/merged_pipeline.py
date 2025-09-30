import os
import re
import torch
import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional, Generator
from threading import Lock

from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from rag.qdrant import client

now = datetime.datetime.now()
formatted_now = now.strftime("%d-%m-%Y %H %M")
COLLECTION_NAME = "ps04_"+formatted_now


class SharedEmbeddingModel:
    """Singleton class for shared embedding model across all threads."""
    
    _instance = None
    _model = None
    _lock = Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SharedEmbeddingModel, cls).__new__(cls)
        return cls._instance
    
    def initialize_model(self, model_name: str = "jinaai/jina-embeddings-v2-small-en"):
        """Initialize the shared model with proper error handling."""
        with self._lock:
            if self._initialized and self._model is not None:
                return self._model
            
            print(f"Initializing shared embedding model: {model_name}")
            
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")
                
                # Try with minimal configuration to avoid meta tensor issues
                model_kwargs = {"device": device}
                encode_kwargs = {"normalize_embeddings": True}
                
                try:
                    self._model = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs=model_kwargs,
                        encode_kwargs=encode_kwargs,
                    )
                    print("✅ Shared model initialized successfully")
                    
                except Exception as primary_error:
                    print(f"Primary initialization failed: {primary_error}")
                    print("Trying CPU-only initialization...")
                    
                    # Fallback to CPU-only
                    model_kwargs = {"device": "cpu"}
                    self._model = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs=model_kwargs,
                        encode_kwargs=encode_kwargs,
                    )
                    print("✅ Shared model initialized on CPU")
                
                # Test the model with a sample text
                test_embedding = self._model.embed_documents(["Test sentence"])
                vector_size = len(test_embedding[0])
                print(f"✅ Model produces {vector_size}-dimensional vectors")
                
                # Initialize Qdrant collection with correct vector size
                initialize_collection_if_needed(vector_size)
                
                self._initialized = True
                return self._model
                
            except Exception as e:
                print(f"❌ Failed to initialize shared model: {e}")
                # Try alternative model as last resort
                try:
                    print("Trying alternative model as fallback...")
                    model_kwargs = {"device": "cpu"}
                    encode_kwargs = {"normalize_embeddings": True}
                    
                    self._model = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs=model_kwargs,
                        encode_kwargs=encode_kwargs,
                    )
                    
                    # Test and setup collection
                    test_embedding = self._model.embed_documents(["Test sentence"])
                    vector_size = len(test_embedding[0])
                    print(f"✅ Fallback model produces {vector_size}-dimensional vectors")
                    initialize_collection_if_needed(vector_size)
                    
                    self._initialized = True
                    return self._model
                    
                except Exception as final_error:
                    print(f"❌ All model initialization attempts failed: {final_error}")
                    raise RuntimeError("Cannot initialize any embedding model")
    
    def get_model(self):
        """Get the initialized shared model."""
        if not self._initialized or self._model is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        return self._model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Thread-safe embedding method."""
        with self._lock:
            model = self.get_model()
            return model.embed_documents(texts)


def initialize_collection_if_needed(vector_size: int):
    """Initialize collection with the correct vector size."""
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating Qdrant collection '{COLLECTION_NAME}' with vector size {vector_size}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        try:
            collection_info = client.get_collection(COLLECTION_NAME)
            existing_size = collection_info.config.params.vectors.size
            if existing_size != vector_size:
                print(f"Warning: Existing collection has vector size {existing_size}, but model produces {vector_size}")
                print(f"Deleting existing collection and recreating with correct size...")
                client.delete_collection(COLLECTION_NAME)
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
        except Exception as e:
            print(f"Error checking collection: {e}")



def preprocess_text(text: str, type: str) -> str:
    """Basic text preprocessing to clean and normalize."""
    text = re.sub(r"\[|\]|\{|\}|\\n|\\", " ", text) # remove garbage json symbols 
    text = re.sub(r"\.\.+", "", text) # remove multiple dots
    # text = re.sub(r"(?<=\bhttps?://[^\s]+)(/[^\s]+)+", "", text) # remove sub domains r"\bhttps?://[^\s]+\b" "(/[^\s]+)+"
    text = re.sub(r"<[^>]+>", "", text) # remove HTML tags
    text = re.sub(r":\s*(null|NULL|Null|None|NONE|none),?", " is empty,", text)  # remove key with null values
    text = re.sub(r": ", " is ", text) # humanize key-value pairs
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    text = text.strip()
    return text

class MergedRAGWorker:
    """A single worker that handles the complete pipeline for a batch of files using shared model."""
    
    def __init__(self, worker_id: int, chunk_size_kb: int = 4, shared_model: SharedEmbeddingModel = None):
        self.worker_id = worker_id
        self.chunk_size_bytes = chunk_size_kb * 1024
        self.shared_model = shared_model or SharedEmbeddingModel()
        self.point_id_counter = worker_id * 10000  # Unique ID range per worker
        print(f"Worker {self.worker_id}: Initialized with shared model")
    
    def get_model(self):
        """Get the shared model instance."""
        return self.shared_model.get_model()
    
    
    def _read_txt_file_chunks(self, file_path: str) -> Generator[Tuple[str, str, int], None, None]:
        """
        Read a file and yield chunks of text, splitting on word boundaries.
        Automatically detects JSON files and uses specialized processing.

        Yields:
            Generator[Tuple[str, str, int], None, None]: A generator that yields
            tuples of (chunk_text, filename, chunk_id).
        """
        filename = os.path.basename(file_path)
        _, ext = os.path.splitext(filename)
        
        # Use specialized JSON processing for JSON files
        if ext.lower() == '.json':
            yield from self._read_json_file_chunks(file_path)
            return
        
        # Default text processing for other files
        chunk_id = 0

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                leftover = ""
                while True:
                    # Read a new chunk from the file
                    chunk = f.read(self.chunk_size_bytes)
                    
                    # If the file is exhausted, yield any remaining text and stop
                    if not chunk:
                        if leftover:
                            yield (leftover, filename, chunk_id)
                        break

                    # Combine the leftover from the previous iteration with the new chunk
                    data = preprocess_text(leftover + chunk)
                    
                    # Find the last space to safely split the text
                    split_point = data.rfind(' ')

                    # If a space is found, we can split
                    if split_point != -1:
                        chunk_to_yield = data[:split_point]
                        leftover = data[split_point:].lstrip() # Keep the rest for the next iteration
                        
                        yield (chunk_to_yield, filename, chunk_id)
                        chunk_id += 1
                    
                    # If no space is found (e.g., a very long word), keep it for the next chunk
                    else:
                        leftover = data

        except Exception as e:
            print(f"Worker {self.worker_id}: Error reading {filename}: {e}")

    def _read_json_file_chunks(self, file_path: str) -> Generator[Tuple[str, str, int], None, None]:
        """
        Read a JSON file and yield humanized chunks by recursively finding highest-level nodes under 1KB.

        Yields:
            Generator[Tuple[str, str, int], None, None]: A generator that yields
            tuples of (humanized_text, filename, chunk_id).
        """
        filename = os.path.basename(file_path)
        chunk_id = 0
        max_chunk_size = self.chunk_size_bytes  # 1KB limit for JSON chunks

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                json_data = json.load(f)
            
            # Process the JSON data recursively to find optimal chunks
            for chunk_text in self._extract_json_chunks(json_data, max_chunk_size):
                if chunk_text.strip():  # Only yield non-empty chunks
                    yield (chunk_text, filename, chunk_id)
                    chunk_id += 1
                    
        except json.JSONDecodeError as e:
            print(f"Worker {self.worker_id}: Invalid JSON in {filename}: {e}")
            # Fallback to treating as regular text file
            yield from self._read_txt_file_chunks(file_path)
        except Exception as e:
            print(f"Worker {self.worker_id}: Error reading JSON {filename}: {e}")

    def _extract_json_chunks(self, data: Any, max_size: int, current_path: str = "") -> Generator[str, None, None]:
        """
        Recursively extract chunks from JSON data, finding the highest level nodes under max_size.
        
        Args:
            data: The JSON data (dict, list, or primitive)
            max_size: Maximum size in bytes for each chunk
            current_path: Current path in the JSON structure (for context)
        
        Yields:
            str: Humanized text chunks
        """
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{current_path}.{key}" if current_path else key
                
                # Try to create a chunk with this key-value pair
                chunk_candidate = self._humanize_json_object({key: value}, new_path)
                chunk_size = len(chunk_candidate.encode('utf-8'))
                
                if chunk_size <= max_size:
                    # This entire key-value pair fits in one chunk
                    yield chunk_candidate
                else:
                    # This key-value pair is too large, recurse into the value
                    if isinstance(value, (dict, list)):
                        # Add context about what we're entering
                        context = f"In {new_path}: "
                        yield from self._extract_json_chunks(value, max_size - len(context.encode('utf-8')), new_path)
                    else:
                        # It's a primitive value that's too large, split it
                        large_text = self._humanize_primitive_value(key, value, new_path)
                        yield from self._split_large_text(large_text, max_size)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{current_path}[{i}]" if current_path else f"item_{i}"
                
                # Try to create a chunk with this array item
                chunk_candidate = self._humanize_json_array_item(item, new_path, i)
                chunk_size = len(chunk_candidate.encode('utf-8'))
                
                if chunk_size <= max_size:
                    # This entire array item fits in one chunk
                    yield chunk_candidate
                else:
                    # This array item is too large, recurse into it
                    yield from self._extract_json_chunks(item, max_size, new_path)
        
        else:
            # It's a primitive value
            text = self._humanize_primitive_value("value", data, current_path)
            if len(text.encode('utf-8')) <= max_size:
                yield text
            else:
                yield from self._split_large_text(text, max_size)

    def _humanize_json_object(self, obj: dict, path: str) -> str:
        """
        Convert a JSON object to human-readable text.
        
        Args:
            obj: Dictionary to humanize
            path: Path context for the object
            
        Returns:
            str: Human-readable text
        """
        if not obj:
            return f"The {path} section is empty."
        
        human_text = []
        
        for key, value in obj.items():
            # Humanize the key
            human_key = key.replace('_', ' ').replace('-', ' ').title()
            
            if isinstance(value, dict):
                if not value:
                    human_text.append(f"{human_key} is empty.")
                else:
                    nested_items = []
                    for nested_key, nested_value in value.items():
                        nested_human_key = nested_key.replace('_', ' ').replace('-', ' ')
                        if isinstance(nested_value, (str, int, float, bool)):
                            if isinstance(nested_value, (str)):
                                nested_items.append(f"{nested_human_key} is {preprocess_text(nested_value)}")
                            else:
                                nested_items.append(f"{nested_human_key} is {nested_value}")
                        elif isinstance(nested_value, list):
                            if nested_value:
                                nested_items.append(f"{nested_human_key} includes {len(nested_value)} items")
                            else:
                                nested_items.append(f"{nested_human_key} is empty")
                        else:
                            nested_items.append(f"{nested_human_key} contains additional details")
                    
                    human_text.append(f"{human_key} contains: {', '.join(nested_items)}.")
            
            elif isinstance(value, list):
                if not value:
                    human_text.append(f"{human_key} is an empty list.")
                else:
                    if len(value) == 1:
                        item_desc = self._describe_list_item(value[0])
                        human_text.append(f"{human_key} contains one item: {item_desc}.")
                    else:
                        item_types = set()
                        for item in value[:3]:  # Sample first 3 items
                            item_types.add(self._get_item_type_description(item))
                        
                        type_desc = ", ".join(item_types)
                        human_text.append(f"{human_key} contains {len(value)} items including {type_desc}.")
            
            elif isinstance(value, (str, int, float)):
                if isinstance(value, str):
                    human_text.append(f"{human_key} is {preprocess_text(value)}")
                else:
                    human_text.append(f"{human_key} is {value}.")
            
            elif isinstance(value, bool):
                human_text.append(f"{human_key} is {'yes' if value else 'no'}.")
            
            elif value is None:
                human_text.append(f"{human_key} is not specified.")
            
            else:
                human_text.append(f"{human_key} contains {type(value).__name__} data.")
        
        return " ".join(human_text)

    def _humanize_json_array_item(self, item: Any, path: str, index: int) -> str:
        """
        Humanize a single array item.
        
        Args:
            item: The array item
            path: Path context
            index: Index in the array
            
        Returns:
            str: Human-readable description
        """
        if isinstance(item, dict):
            return f"Item {index + 1}: {self._humanize_json_object(item, f'{path}[{index}]')}"
        elif isinstance(item, list):
            return f"Item {index + 1} is a list containing {len(item)} elements."
        elif isinstance(item, str):
            if len(item) > 100:
                return f"Item {index + 1} is a text entry: {item}..."
            else:
                return f"Item {index + 1} is: {item}"
        elif isinstance(item, (int, float)):
            return f"Item {index + 1} has value: {item}"
        elif isinstance(item, bool):
            return f"Item {index + 1} is {'true' if item else 'false'}"
        elif item is None:
            return f"Item {index + 1} is empty"
        else:
            return f"Item {index + 1} contains {type(item).__name__} data"

    def _describe_list_item(self, item: Any) -> str:
        """Describe a single list item briefly."""
        if isinstance(item, dict):
            keys = list(item.keys())[:3]
            key_desc = ", ".join(keys)
            return f"an object with keys like {key_desc}"
        elif isinstance(item, list):
            return f"a nested list with {len(item)} items"
        elif isinstance(item, str):
            return f"text: {item}"
        elif isinstance(item, (int, float)):
            return f"number: {item}"
        elif isinstance(item, bool):
            return f"boolean: {item}"
        else:
            return f"{type(item).__name__} data"

    def _get_item_type_description(self, item: Any) -> str:
        """Get a brief type description for an item."""
        if isinstance(item, dict):
            return "objects"
        elif isinstance(item, list):
            return "lists"
        elif isinstance(item, str):
            return "text entries"
        elif isinstance(item, (int, float)):
            return "numbers"
        elif isinstance(item, bool):
            return "boolean values"
        else:
            return f"{type(item).__name__} items"

    def _humanize_primitive_value(self, key: str, value: Any, path: str) -> str:
        """Humanize a primitive value with context."""
        human_key = key.replace('_', ' ').replace('-', ' ').title()
        
        if isinstance(value, str):
            if len(value) > 200:
                return f"{human_key} in {path} is a detailed text: {value}"
            else:
                return f"{human_key} in {path} is: {value}"
        elif isinstance(value, (int, float)):
            return f"{human_key} in {path} has value: {value}"
        elif isinstance(value, bool):
            return f"{human_key} in {path} is {'yes' if value else 'no'}"
        elif value is None:
            return f"{human_key} in {path} is not specified"
        else:
            return f"{human_key} in {path} contains {type(value).__name__} data"

    def _split_large_text(self, text: str, max_size: int) -> Generator[str, None, None]:
        """Split large text into chunks that fit within max_size."""
        text = text.strip()
        if not text:
            return
        
        while text:
            if len(text.encode('utf-8')) <= max_size:
                yield text
                break
            
            # Find a good split point (sentence boundary, then word boundary)
            split_point = max_size
            temp_text = text[:split_point]
            
            # Try to split at sentence boundary
            sentence_end = max(temp_text.rfind('. '), temp_text.rfind('! '), temp_text.rfind('? '))
            if sentence_end > max_size // 2:  # Only if we're not splitting too early
                split_point = sentence_end + 1
            else:
                # Split at word boundary
                word_boundary = temp_text.rfind(' ')
                if word_boundary > max_size // 2:
                    split_point = word_boundary
            
            chunk = text[:split_point].strip()
            if chunk:
                yield chunk
            
            text = text[split_point:].strip()

        
    
    def _embed_chunks(self, chunks: List[Tuple[str, str, int]]) -> List[Tuple[List[float], Dict[str, Any]]]:
        """Embed chunks and create payloads with robust error handling."""
        if not chunks:
            return []
        
        try:
            texts = [chunk[0] for chunk in chunks]
            
            # Validate texts before embedding
            if not texts or any(not text.strip() for text in texts):
                print(f"Worker {self.worker_id}: Warning - some texts are empty or whitespace-only")
                # Filter out empty texts
                valid_chunks = [(text, filename, chunk_id) for text, filename, chunk_id in chunks if text.strip()]
                if not valid_chunks:
                    print(f"Worker {self.worker_id}: No valid texts to embed")
                    return []
                texts = [chunk[0] for chunk in valid_chunks]
                chunks = valid_chunks
            
            print(f"Worker {self.worker_id}: Embedding {len(texts)} text chunks...")
            
            # Try embedding with error handling using shared model
            try:
                embeddings = self.shared_model.embed_documents(texts)
                ##### print(embeddings[0])
                if not embeddings or len(embeddings) != len(texts):
                    print(f"Worker {self.worker_id}: Embedding count mismatch - expected {len(texts)}, got {len(embeddings) if embeddings else 0}")
                    return []
                    
            except RuntimeError as rt_error:
                if "meta tensor" in str(rt_error).lower():
                    print(f"Worker {self.worker_id}: Meta tensor error detected with shared model: {rt_error}")
                    # With shared model, we can't reinitialize per worker - just re-raise
                    raise rt_error
                else:
                    raise rt_error
            
            results = []
            for i, (chunk_text, filename, chunk_id) in enumerate(chunks):
                payload = {
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "text": chunk_text, 
                    "chunk_size": len(chunk_text.encode("utf-8")),
                    "worker_id": self.worker_id,
                    "text_length": len(chunk_text)
                }
                results.append((embeddings[i], payload))
            
            print(f"Worker {self.worker_id}: Successfully embedded {len(results)} chunks")
            return results
            
        except Exception as e:
            print(f"Worker {self.worker_id}: Error embedding chunks: {e}")
            print(f"Worker {self.worker_id}: Error type: {type(e).__name__}")
            
            # Try fallback approach - embed one at a time
            print(f"Worker {self.worker_id}: Attempting fallback individual embedding...")
            try:
                results = []
                for chunk_text, filename, chunk_id in chunks:
                    try:
                        embedding = self.shared_model.embed_documents([chunk_text])
                        if embedding and len(embedding) > 0:
                            payload = {
                                "filename": filename,
                                "chunk_id": chunk_id,
                                "text": chunk_text,
                                "chunk_size": len(chunk_text.encode("utf-8")),
                                "worker_id": self.worker_id,
                                "text_length": len(chunk_text)
                            }
                            results.append((embedding[0], payload))
                    except Exception as single_error:
                        print(f"Worker {self.worker_id}: Failed to embed individual chunk from {filename}: {single_error}")
                        continue
                
                if results:
                    print(f"Worker {self.worker_id}: Fallback embedding succeeded for {len(results)} chunks")
                    return results
                    
            except Exception as fallback_error:
                print(f"Worker {self.worker_id}: Fallback embedding also failed: {fallback_error}")
            
            return []
    
    def _store_embeddings(self, embeddings_with_payloads: List[Tuple[List[float], Dict[str, Any]]]) -> int:
        """Store embeddings in Qdrant with dynamic vector size detection."""
        if not embeddings_with_payloads:
            return 0
        
        try:
            # Detect vector size from first embedding
            first_embedding = embeddings_with_payloads[0][0]
            vector_size = len(first_embedding)
            
            # Initialize collection with correct vector size
            initialize_collection_if_needed(vector_size)
            
            points = []
            for embedding, payload in embeddings_with_payloads:
                self.point_id_counter += 1
                
                # Validate embedding size
                if len(embedding) != vector_size:
                    print(f"Worker {self.worker_id}: Warning - embedding size mismatch: expected {vector_size}, got {len(embedding)}")
                    continue
                
                points.append(
                    PointStruct(
                        id=self.point_id_counter,
                        vector=embedding, ### check this
                        payload=payload,
                    )
                )
            
            if not points:
                print(f"Worker {self.worker_id}: No valid points to store")
                return 0
            
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            
            print(f"Worker {self.worker_id}: Successfully stored {len(points)} embeddings (vector size: {vector_size})")
            return len(points)
            
        except Exception as e:
            print(f"Worker {self.worker_id}: Error storing embeddings: {e}")
            return 0
    
    def process_file_batch(self, file_paths: List[str]) -> Dict[str, int]:
        """Process a batch of files through the complete pipeline."""
        stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0
        }
        
        for file_path in file_paths:
            try:
                # Step 1: Read file and create chunks
                chunks_generator = self._read_json_file_chunks(file_path)
                chunks = list(chunks_generator)  # Convert generator to list for multiple passes
                if not chunks:
                    continue
                
                stats["chunks_created"] += len(chunks)
                
                # Step 2: Embed chunks
                embeddings_with_payloads = self._embed_chunks(chunks)
                if not embeddings_with_payloads:
                    stats["errors"] += 1
                    continue
                
                stats["embeddings_generated"] += len(embeddings_with_payloads)
                
                # Step 3: Store embeddings
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

    def process_chunk_batch(self, chunk_batch: List[Tuple[str, str, int]]) -> Dict[str, int]:
        """
        Process a batch of chunks through embedding and storage pipeline.
        
        Args:
            chunk_batch: List of tuples (chunk_text, filename, chunk_id)
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            "chunks_processed": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0
        }
        
        if not chunk_batch:
            return stats
        
        try:
            print(f"Worker {self.worker_id}: Processing batch of {len(chunk_batch)} chunks")
            
            # Step 1: Embed chunks
            embeddings_with_payloads = self._embed_chunks(chunk_batch)
            if not embeddings_with_payloads:
                stats["errors"] = len(chunk_batch)
                return stats
            
            stats["embeddings_generated"] = len(embeddings_with_payloads)
            
            # Step 2: Store embeddings
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
        """
        Read all files and extract all chunks without processing them.
        
        Args:
            file_paths: List of file paths to read
            
        Returns:
            List of all chunks from all files
        """
        all_chunks = []
        
        for file_path in file_paths:
            try:
                # Determine file type and use appropriate reader
                filename = os.path.basename(file_path)
                _, ext = os.path.splitext(filename)
                
                if ext.lower() == '.json':
                    chunks_generator = self._read_json_file_chunks(file_path)
                else:
                    chunks_generator = self._read_txt_file_chunks(file_path)
                
                chunks = list(chunks_generator)
                all_chunks.extend(chunks)
                
                print(f"Worker {self.worker_id}: Extracted {len(chunks)} chunks from {filename}")
                
            except Exception as e:
                print(f"Worker {self.worker_id}: Error reading {file_path}: {e}")
                continue
        
        return all_chunks


class ChunkBasedMultiThreadedRAGPipeline:
    """Pipeline where chunks are distributed across threads for parallel processing."""
    
    def __init__(self, max_workers: int = 4, chunk_size_kb: int = 4, chunks_per_batch: int = 50):
        self.max_workers = max_workers
        self.chunk_size_kb = chunk_size_kb
        self.chunks_per_batch = chunks_per_batch
        self.lock = Lock()
        self.total_stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "chunks_processed": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0
        }
        
        # Initialize shared model instance
        print("Initializing shared embedding model for chunk-based processing...")
        self.shared_model = SharedEmbeddingModel()
        self.shared_model.initialize_model()
        print("Shared embedding model initialized successfully")
    
    def _read_all_files_to_chunks(self, directory_path: str) -> List[Tuple[str, str, int]]:
        """Read all files in directory and extract all chunks."""
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
        
        # Use a single worker to read all files and extract chunks
        worker = MergedRAGWorker(
            worker_id=0,
            chunk_size_kb=self.chunk_size_kb,
            shared_model=self.shared_model
        )
        
        all_chunks = worker.read_all_file_chunks(file_paths)
        
        self.total_stats["files_processed"] = len(file_paths)
        self.total_stats["chunks_created"] = len(all_chunks)
        
        print(f"Extracted {len(all_chunks)} total chunks from {len(file_paths)} files")
        return all_chunks
    
    def _create_chunk_batches(self, chunks: List[Tuple[str, str, int]]) -> List[List[Tuple[str, str, int]]]:
        """Divide chunks into batches for threading."""
        if not chunks:
            return []
        
        batches = [
            chunks[i:i + self.chunks_per_batch] 
            for i in range(0, len(chunks), self.chunks_per_batch)
        ]
        
        print(f"Created {len(batches)} chunk batches ({self.chunks_per_batch} chunks per batch)")
        return batches
    
    def _update_stats(self, worker_stats: Dict[str, int]):
        """Thread-safe stats update."""
        with self.lock:
            for key, value in worker_stats.items():
                if key in self.total_stats:
                    self.total_stats[key] += value
    
    def _process_chunk_batch_with_worker(self, batch_info: Tuple[int, List[Tuple[str, str, int]]]) -> Dict[str, int]:
        """Process a batch of chunks with a dedicated worker."""
        worker_id, chunk_batch = batch_info
        worker = MergedRAGWorker(
            worker_id=worker_id,
            chunk_size_kb=self.chunk_size_kb,
            shared_model=self.shared_model
        )
        
        stats = worker.process_chunk_batch(chunk_batch)
        return stats
    
    def process_directory(self, directory_path: str) -> Dict[str, int]:
        """Process directory with chunk-based multithreaded pipeline."""
        print(f"Starting chunk-based multithreaded processing of directory: {directory_path}")
        print(f"Configuration: {self.max_workers} workers, {self.chunks_per_batch} chunks per batch, {self.chunk_size_kb}KB max chunk size")
        
        # Step 1: Read all files and extract chunks
        all_chunks = self._read_all_files_to_chunks(directory_path)
        if not all_chunks:
            return self.total_stats
        
        # Step 2: Create chunk batches
        chunk_batches = self._create_chunk_batches(all_chunks)
        if not chunk_batches:
            return self.total_stats
        
        # Step 3: Process chunk batches with thread pool
        batch_info = [(i, batch) for i, batch in enumerate(chunk_batches)]
        
        print(f"Processing {len(chunk_batches)} chunk batches with {self.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunk batches for processing
            future_to_batch = {
                executor.submit(self._process_chunk_batch_with_worker, info): info
                for info in batch_info
            }
            
            # Process completed batches with progress tracking
            with tqdm(total=len(chunk_batches), desc="Processing chunk batches") as pbar:
                for future in as_completed(future_to_batch):
                    batch_info = future_to_batch[future]
                    try:
                        worker_stats = future.result()
                        self._update_stats(worker_stats)
                        
                        # Update progress description
                        pbar.set_postfix({
                            'chunks_processed': self.total_stats['chunks_processed'],
                            'embedded': self.total_stats['embeddings_generated'],
                            'stored': self.total_stats['embeddings_stored']
                        })
                        
                    except Exception as e:
                        print(f"Error processing chunk batch {batch_info[0]}: {e}")
                        # Count all chunks in failed batch as errors
                        self.total_stats["errors"] += len(batch_info[1])
                    finally:
                        pbar.update(1)
        
        # Print final statistics
        print("\n" + "="*60)
        print("CHUNK-BASED PROCESSING COMPLETED")
        print("="*60)
        for key, value in self.total_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        return self.total_stats


class MergedMultiThreadedRAGPipeline:
    """Merged pipeline where each thread processes files through the complete pipeline."""
    
    def __init__(self, max_workers: int = 4, chunk_size_kb: int = 4, files_per_batch: int = 5):
        self.max_workers = max_workers
        self.chunk_size_kb = chunk_size_kb
        self.files_per_batch = files_per_batch
        self.lock = Lock()
        self.total_stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0
        }
        
        # Initialize shared model instance
        print("Initializing shared embedding model...")
        self.shared_model = SharedEmbeddingModel()
        # Actually initialize the model
        self.shared_model.initialize_model()
        print("Shared embedding model initialized successfully")
    
    def _create_file_batches(self, directory_path: str) -> List[List[str]]:
        """Create batches of files for processing."""
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
        
        # Create batches
        batches = [
            file_paths[i:i + self.files_per_batch] 
            for i in range(0, len(file_paths), self.files_per_batch)
        ]
        
        return batches
    
    def _update_stats(self, worker_stats: Dict[str, int]):
        """Thread-safe stats update."""
        with self.lock:
            for key, value in worker_stats.items():
                self.total_stats[key] += value
    
    def _process_batch_with_worker(self, batch_info: Tuple[int, List[str]]) -> Dict[str, int]:
        """Process a batch of files with a dedicated worker."""
        worker_id, file_paths = batch_info
        worker = MergedRAGWorker(
            worker_id=worker_id,
            chunk_size_kb=self.chunk_size_kb,
            shared_model=self.shared_model
        )
        
        stats = worker.process_file_batch(file_paths)
        return stats
    
    def process_directory(self, directory_path: str) -> Dict[str, int]:
        """Process directory with merged multithreaded pipeline."""
        print(f"Starting merged multithreaded processing of directory: {directory_path}")
        print(f"Configuration: {self.max_workers} workers, {self.files_per_batch} files per batch, {self.chunk_size_kb}KB max chunk size")
        
        # Create file batches
        file_batches = self._create_file_batches(directory_path)
        if not file_batches:
            return self.total_stats
        
        print(f"Created {len(file_batches)} batches from directory")
        
        # Process batches with thread pool
        batch_info = [(i, batch) for i, batch in enumerate(file_batches)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches for processing
            future_to_batch = {
                executor.submit(self._process_batch_with_worker, info): info
                for info in batch_info
            }
            
            # Process completed batches with progress tracking
            with tqdm(total=len(file_batches), desc="Processing file batches") as pbar:
                for future in as_completed(future_to_batch):
                    batch_info = future_to_batch[future]
                    try:
                        worker_stats = future.result()
                        self._update_stats(worker_stats)
                        
                        # Update progress description
                        pbar.set_postfix({
                            'files': self.total_stats['files_processed'],
                            'chunks': self.total_stats['chunks_created'],
                            'stored': self.total_stats['embeddings_stored']
                        })
                        
                    except Exception as e:
                        print(f"Error processing batch {batch_info[0]}: {e}")
                        self.total_stats["errors"] += len(batch_info[1])
                    finally:
                        pbar.update(1)
        
        # Print final statistics
        print("\n" + "="*50)
        print("PROCESSING COMPLETED")
        print("="*50)
        for key, value in self.total_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        return self.total_stats


# Convenience functions
def run_merged_rag_pipeline(directory_path: str, max_workers: int = 4, chunk_size_kb: int = 4, files_per_batch: int = 5) -> Dict[str, int]:
    """
    Run the merged multithreaded RAG pipeline (file-based processing).
    
    Args:
        directory_path: Directory containing files to process
        max_workers: Number of worker threads
        chunk_size_kb: Maximum chunk size in KB
        files_per_batch: Number of files each worker processes per batch
    
    Returns:
        Dictionary with processing statistics
    """
    pipeline = MergedMultiThreadedRAGPipeline(
        max_workers=max_workers,
        chunk_size_kb=chunk_size_kb,
        files_per_batch=files_per_batch
    )
    
    return pipeline.process_directory(directory_path)


def run_chunk_based_rag_pipeline(directory_path: str, max_workers: int = 4, chunk_size_kb: int = 4, chunks_per_batch: int = 50) -> Dict[str, int]:
    """
    Run the chunk-based multithreaded RAG pipeline.
    
    This approach:
    1. Reads all files and extracts chunks first (single-threaded)
    2. Distributes chunks across workers for parallel embedding/storage
    3. Better load balancing when files have varying chunk counts
    
    Args:
        directory_path: Directory containing files to process
        max_workers: Number of worker threads
        chunk_size_kb: Maximum chunk size in KB
        chunks_per_batch: Number of chunks each worker processes per batch
    
    Returns:
        Dictionary with processing statistics
    """
    pipeline = ChunkBasedMultiThreadedRAGPipeline(
        max_workers=max_workers,
        chunk_size_kb=chunk_size_kb,
        chunks_per_batch=chunks_per_batch
    )
    
    return pipeline.process_directory(directory_path)


def run_hybrid_rag_pipeline(directory_path: str, max_workers: int = 4, chunk_size_kb: int = 4, 
                           files_per_batch: int = 5, chunks_per_batch: int = 50, 
                           use_chunk_based: bool = None) -> Dict[str, int]:
    """
    Run RAG pipeline with automatic selection between file-based and chunk-based processing.
    
    Args:
        directory_path: Directory containing files to process
        max_workers: Number of worker threads
        chunk_size_kb: Maximum chunk size in KB
        files_per_batch: Number of files per batch (file-based mode)
        chunks_per_batch: Number of chunks per batch (chunk-based mode)
        use_chunk_based: Force chunk-based processing (None = auto-select)
    
    Returns:
        Dictionary with processing statistics
    """
    # Auto-select processing method if not specified
    if use_chunk_based is None:
        # Count files to make decision
        file_count = 0
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path):
                    file_count += 1
        
        # Use chunk-based for many small files, file-based for fewer large files
        use_chunk_based = file_count > 20
        
        print(f"Auto-selected {'chunk-based' if use_chunk_based else 'file-based'} processing for {file_count} files")
    
    if use_chunk_based:
        return run_chunk_based_rag_pipeline(
            directory_path=directory_path,
            max_workers=max_workers,
            chunk_size_kb=chunk_size_kb,
            chunks_per_batch=chunks_per_batch
        )
    else:
        return run_merged_rag_pipeline(
            directory_path=directory_path,
            max_workers=max_workers,
            chunk_size_kb=chunk_size_kb,
            files_per_batch=files_per_batch
        )