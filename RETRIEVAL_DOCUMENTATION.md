# Multithreaded Query Retrieval System

## Overview

This system processes queries from a JSON file, uses multithreading to divide the workload, encodes queries using a shared embedding model, searches for similar chunks in a Qdrant vector database, and outputs results in a structured JSON format.

## Features

✅ **Multithreaded Processing**: Efficient parallel processing of query batches  
✅ **Shared Embedding Model**: Single model instance prevents memory issues  
✅ **Vector Similarity Search**: Uses Qdrant for fast semantic similarity matching  
✅ **Structured Output**: Organized JSON results with metadata  
✅ **Progress Tracking**: Real-time progress bars and statistics  
✅ **Error Handling**: Robust error handling with detailed logging  

## Input Format

The system expects a JSON file with queries in this format:

```json
[
    {
        "query_num": "1",
        "query": "In what year did the Green Giant make his national ad debut?"
    },
    {
        "query_num": "10", 
        "query": "In what year was 111 Huntington Avenue completed?"
    }
]
```

## Output Format

The system outputs a JSON file with this structure for each query:

```json
{
    "query_num": "1",
    "query": "In what year did the Green Giant make his national ad debut?",
    "retrieval_timestamp": "2025-09-30 14:30:15",
    "top_k": 5,
    "chunk_1_text": "The Green Giant made his national advertising debut in 1928...",
    "chunk_1_filename": "advertising_history.txt",
    "chunk_1_score": 0.8945,
    "chunk_1_chunk_id": 15,
    "chunk_2_text": "Green Giant's early marketing campaigns...",
    "chunk_2_filename": "marketing_data.json", 
    "chunk_2_score": 0.8721,
    "chunk_2_chunk_id": 23,
    // ... up to chunk_5
}
```

## Usage

### Method 1: Direct Script Execution

Run the standalone script:

```bash
python run_retrieval.py
```

### Method 2: Import and Use

```python
from rag.retrieval import run_multithreaded_retrieval

results = run_multithreaded_retrieval(
    queries_file_path=r"C:\Users\22bcscs055\Downloads\Queries.json",
    output_file_path="results.json",
    max_workers=4,     # Number of threads
    top_k=5,          # Number of similar chunks per query
    queries_per_batch=15  # Queries per batch
)
```

### Method 3: Advanced Usage

```python
from rag.retrieval import MultiThreadedRetriever

retriever = MultiThreadedRetriever(
    max_workers=6,
    top_k=10,
    queries_per_batch=20
)

results = retriever.process_queries(
    queries_file_path="queries.json",
    output_file_path="detailed_results.json"
)
```

## Configuration Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `max_workers` | Number of worker threads | 4 | 4-8 |
| `top_k` | Similar chunks per query | 5 | 5-10 |
| `queries_per_batch` | Queries per batch | 10 | 10-20 |

## System Requirements

- **Qdrant Collection**: Must have collection named "ps04-merged"
- **Embedding Model**: Shared model from merged_pipeline
- **Memory**: ~2-4GB for embedding model
- **Dependencies**: torch, sentence-transformers, qdrant-client, tqdm

## Performance

- **Query Processing**: ~100-500 queries/minute (depends on complexity)
- **Batch Size**: Optimal batch size is 10-20 queries
- **Threading**: 4-6 workers optimal for most systems
- **Memory Usage**: Shared model approach minimizes memory footprint

## File Structure

```
ps04-rag-v2/
├── rag/
│   ├── retrieval.py          # Main retrieval module
│   ├── merged_pipeline.py    # Shared embedding model
│   └── qdrant.py            # Database client
├── run_retrieval.py         # Standalone execution script
├── test_retrieval.py        # Test script
└── query_retrieval_results.json  # Output file
```

## Error Handling

The system handles various error scenarios:

- **Missing Collection**: Verifies Qdrant collection exists
- **Invalid Queries**: Handles malformed JSON input
- **Embedding Errors**: Graceful handling of model issues
- **Network Issues**: Retry logic for database connections
- **Memory Issues**: Shared model prevents memory overflow

## Monitoring

The system provides real-time feedback:

```
🚀 Starting retrieval with 4 workers...
Processing query batches: 100%|████████| 50/50 [02:15<00:00,  2.70s/it]

✅ RETRIEVAL COMPLETED
Total queries processed: 1000
Successful retrievals: 998
Failed retrievals: 2
Top-k per query: 5
```

## Troubleshooting

### Common Issues

1. **Collection Not Found**
   ```
   Collection 'ps04-merged' does not exist
   ```
   **Solution**: Run the ingestion pipeline first to create embeddings

2. **Memory Errors**
   ```
   CUDA out of memory
   ```
   **Solution**: Reduce `max_workers` or use CPU-only mode

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'torch'
   ```
   **Solution**: Install dependencies: `pip install torch sentence-transformers`

4. **File Not Found**
   ```
   Queries file not found
   ```
   **Solution**: Check file path and permissions

## Performance Tuning

### For Large Query Sets (>1000 queries):
- Increase `queries_per_batch` to 20-30
- Use 6-8 `max_workers`
- Consider CPU-only mode for stability

### For Small Query Sets (<100 queries):
- Reduce `max_workers` to 2-3
- Use smaller batch sizes (5-10)
- Single-threaded may be sufficient

### For High Precision Requirements:
- Increase `top_k` to 10-15
- Use more sophisticated similarity scoring
- Consider query preprocessing

## Integration with Existing Pipeline

The retrieval module integrates seamlessly with your existing RAG pipeline:

```python
# 1. Ingest documents
from rag.merged_pipeline import run_merged_rag_pipeline
run_merged_rag_pipeline("documents/")

# 2. Retrieve similar chunks  
from rag.retrieval import run_multithreaded_retrieval
results = run_multithreaded_retrieval("queries.json", "results.json")

# 3. Use results for downstream tasks
# (e.g., answer generation, summarization)
```