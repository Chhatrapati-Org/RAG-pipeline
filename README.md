# PS04 RAG - Multithreaded RAG Pipeline

A high-performance, multithreaded Retrieval-Augmented Generation (RAG) pipeline that processes documents efficiently using parallel processing.

## Features

### 🚀 Multithreaded Processing
- **File Reading**: Parallel processing of multiple files
- **Chunking**: Documents split into 4KB or smaller chunks  
- **Embedding**: Multithreaded embedding generation
- **Storage**: Parallel storage to Qdrant vector database

### 📊 Key Specifications
- **Max Chunk Size**: 4KB per chunk (configurable)
- **Thread Safety**: All operations are thread-safe
- **Metadata Storage**: File names and chunk information stored in payloads
- **Progress Tracking**: Real-time progress bars for all operations

## Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -e .

# Start Qdrant server (required for embedding storage)
docker run -p 6333:6333 qdrant/qdrant
```

### Basic Usage

#### Option 1: Simple Function Call
```python
from rag.pipeline import run_rag_pipeline

# Process a directory with default settings
processed_count = run_rag_pipeline(
    directory_path="path/to/your/documents",
    max_workers=4,
    chunk_size_kb=4
)
print(f"Processed {processed_count} chunks")
```

#### Option 2: Advanced Pipeline Control
```python
from rag.pipeline import MultiThreadedRAGPipeline

# Create pipeline with custom settings
pipeline = MultiThreadedRAGPipeline(
    max_workers=6,              # More threads for faster processing
    chunk_size_kb=3,            # Smaller chunks (3KB)
    embedding_batch_size=16,    # Embedding batch size
    storage_batch_size=50       # Storage batch size
)

# Process directory
result = pipeline.process_directory("path/to/your/documents")
```

#### Option 3: Run Main Script
```bash
# Update directory path in main.py, then run:
python main.py
```

## Architecture

### Components

#### 1. MultiThreadedFileReader (`rag/ingestor.py`)
- Reads files in parallel using ThreadPoolExecutor
- Creates chunks of specified size (4KB default)
- Handles various file formats (text, JSON)
- Thread-safe chunk generation

#### 2. MultiThreadedChunker (`rag/chunker.py`)  
- Processes chunks in parallel batches
- Generates embeddings using HuggingFace BGE model
- Creates metadata payloads with file information
- Thread-safe embedding operations

#### 3. MultiThreadedEmbeddingStore (`rag/store_embedding.py`)
- Stores embeddings in parallel batches
- Uses Qdrant vector database
- Thread-safe point ID generation
- Comprehensive error handling

#### 4. MultiThreadedRAGPipeline (`rag/pipeline.py`)
- Orchestrates the complete pipeline
- Configurable threading and batch sizes
- Progress monitoring and error reporting

## Metadata Structure

Each chunk is stored with the following payload:
```json
{
    "filename": "document.txt",
    "chunk_id": 0,
    "text": "The actual chunk content...",
    "chunk_size": 4096,
    "unique_id": "uuid-string"
}
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_workers` | Number of worker threads | 4 |
| `chunk_size_kb` | Maximum chunk size in KB | 4 |
| `embedding_batch_size` | Batch size for embedding | 32 |
| `storage_batch_size` | Batch size for storage | 100 |

## Performance Tips

1. **CPU Cores**: Set `max_workers` to your CPU core count
2. **Memory**: Larger batch sizes use more memory but may be faster
3. **Chunk Size**: Smaller chunks (2-3KB) may improve retrieval accuracy
4. **GPU**: Embeddings automatically use CUDA if available

## Testing

Run the test suite to verify functionality:
```bash
python test_pipeline.py
```

Run the example to see the pipeline in action:
```bash
python example_usage.py
```

## Dependencies

- `langchain>=0.3.27` - LLM framework and embeddings
- `langchain-community>=0.3.29` - Community extensions
- `qdrant-client>=1.15.1` - Vector database client
- `pandas>=2.3.2` - Data processing
- `tqdm>=4.67.1` - Progress bars

## File Structure

```
ps04-rag-v2/
├── rag/
│   ├── ingestor.py      # Multithreaded file reading and chunking
│   ├── chunker.py       # Multithreaded embedding generation
│   ├── store_embedding.py # Multithreaded storage operations
│   ├── pipeline.py      # Main pipeline orchestration
│   └── qdrant.py        # Qdrant client configuration
├── main.py              # Main execution script
├── example_usage.py     # Usage examples
├── test_pipeline.py     # Test suite
└── pyproject.toml       # Project dependencies
```

## Troubleshooting

### Common Issues

1. **Qdrant Connection Error**
   - Ensure Qdrant server is running on localhost:6333
   - Check firewall settings

2. **Memory Issues**  
   - Reduce `embedding_batch_size` and `storage_batch_size`
   - Reduce `max_workers`

3. **CUDA Issues**
   - Pipeline will automatically fall back to CPU
   - Ensure CUDA-compatible PyTorch is installed for GPU acceleration

### Error Messages

- `"Directory does not exist"`: Check the directory path
- `"No files found to process"`: Ensure directory contains readable files
- `"Error embedding batch"`: Model loading issue, check internet connection

## Advanced Usage

### Custom Embedding Model
```python
chunker = MultiThreadedChunker(
    max_workers=4,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Processing Specific File Types
The pipeline automatically handles:
- Text files (.txt, .md, etc.)
- JSON files (with content cleaning)
- Any UTF-8 encoded files

## Contributing

1. Ensure thread safety in all operations
2. Add progress bars for long-running operations
3. Include comprehensive error handling
4. Update tests for new functionality

## License

This project is part of the PS04 assignment series.