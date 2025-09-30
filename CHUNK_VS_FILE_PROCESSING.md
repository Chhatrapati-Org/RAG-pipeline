# Chunk-Based vs File-Based Processing Documentation

## Overview

The RAG pipeline now supports two different multithreading approaches:

1. **File-Based Processing** - Each thread processes complete files
2. **Chunk-Based Processing** - Threads process batches of chunks from all files

## Processing Approaches Comparison

### 🗂️ File-Based Processing (`MergedMultiThreadedRAGPipeline`)

**How it works:**
- Divides files into batches (e.g., 5 files per batch)
- Each worker thread processes an entire batch of files
- Per file: read → chunk → embed → store (sequentially)

**Best for:**
- Fewer files (< 20)
- Large files with many chunks
- When file-level atomicity is important
- Simple processing scenarios

**Architecture:**
```
Worker 1: [File1, File2, File3] → Process Complete Pipeline
Worker 2: [File4, File5, File6] → Process Complete Pipeline  
Worker 3: [File7, File8, File9] → Process Complete Pipeline
```

### 🧩 Chunk-Based Processing (`ChunkBasedMultiThreadedRAGPipeline`)

**How it works:**
- Reads ALL files first to extract chunks (single-threaded)
- Divides chunks into batches (e.g., 50 chunks per batch)
- Workers process chunk batches: embed → store (parallel)

**Best for:**
- Many files (> 20)
- Files with varying chunk counts
- Better load balancing needed
- I/O and compute separation

**Architecture:**
```
Phase 1: Read All Files → Extract All Chunks (Single Thread)
Phase 2: Distribute Chunks Across Workers
Worker 1: [Chunk1-50] → Embed & Store
Worker 2: [Chunk51-100] → Embed & Store
Worker 3: [Chunk101-150] → Embed & Store
```

## API Reference

### File-Based Processing

```python
from rag.merged_pipeline import run_merged_rag_pipeline

stats = run_merged_rag_pipeline(
    directory_path="path/to/documents",
    max_workers=4,           # Number of worker threads
    chunk_size_kb=4,        # Max chunk size in KB
    files_per_batch=5       # Files per worker batch
)
```

### Chunk-Based Processing

```python
from rag.merged_pipeline import run_chunk_based_rag_pipeline

stats = run_chunk_based_rag_pipeline(
    directory_path="path/to/documents", 
    max_workers=4,           # Number of worker threads
    chunk_size_kb=4,        # Max chunk size in KB
    chunks_per_batch=50     # Chunks per worker batch
)
```

### Hybrid Processing (Auto-Select)

```python
from rag.merged_pipeline import run_hybrid_rag_pipeline

stats = run_hybrid_rag_pipeline(
    directory_path="path/to/documents",
    max_workers=4,
    chunk_size_kb=4,
    files_per_batch=5,      # For file-based mode
    chunks_per_batch=50,    # For chunk-based mode
    use_chunk_based=None    # Auto-select based on file count
)
```

## Performance Characteristics

### File-Based Processing

**Advantages:**
- ✅ Simpler coordination between threads
- ✅ Natural file boundaries for error handling
- ✅ Better for large files with many chunks
- ✅ Consistent memory usage per worker

**Disadvantages:**
- ❌ Load imbalance if files have different chunk counts
- ❌ Some workers may finish early (idle time)
- ❌ File I/O happens in parallel (potential contention)

### Chunk-Based Processing

**Advantages:**
- ✅ Perfect load balancing across workers
- ✅ Efficient handling of varying file sizes
- ✅ Separation of I/O (read) and compute (embed/store)
- ✅ More granular progress tracking

**Disadvantages:**
- ❌ Requires reading all files before processing
- ❌ Higher memory usage (all chunks in memory)
- ❌ More complex coordination logic

## Configuration Guidelines

### File Count Based Selection

| File Count | Recommended Approach | Reason |
|------------|---------------------|---------|
| 1-10 files | File-Based | Simple coordination, low overhead |
| 11-50 files | Either approach | Performance similar |
| 50+ files | Chunk-Based | Better load balancing |

### File Size Considerations

| Scenario | Recommended Approach | Configuration |
|----------|---------------------|---------------|
| Many small files | Chunk-Based | `chunks_per_batch=100` |
| Few large files | File-Based | `files_per_batch=2-3` |
| Mixed sizes | Chunk-Based | `chunks_per_batch=50` |
| Very large files | File-Based | `files_per_batch=1` |

### Worker Count Optimization

```python
# For CPU-bound workloads (embedding)
max_workers = min(cpu_count(), 8)

# For mixed I/O and CPU
max_workers = cpu_count() // 2

# For memory-constrained systems
max_workers = 2-4
```

## Example Use Cases

### Use Case 1: Large Document Collection

**Scenario:** 1000 small JSON files (1-5KB each)

```python
# Recommended: Chunk-based processing
stats = run_chunk_based_rag_pipeline(
    directory_path="large_collection/",
    max_workers=6,
    chunk_size_kb=4,
    chunks_per_batch=100  # Higher batch size for small files
)
```

### Use Case 2: Few Large Documents

**Scenario:** 5 large text files (1-10MB each)

```python
# Recommended: File-based processing
stats = run_merged_rag_pipeline(
    directory_path="large_docs/",
    max_workers=4,
    chunk_size_kb=4,
    files_per_batch=1  # One large file per worker
)
```

### Use Case 3: Mixed Document Sizes

**Scenario:** 100 files with varying sizes (1KB - 1MB)

```python
# Recommended: Hybrid auto-selection
stats = run_hybrid_rag_pipeline(
    directory_path="mixed_docs/",
    max_workers=4,
    chunk_size_kb=4,
    use_chunk_based=None  # Auto-select based on file count
)
```

## Statistics Comparison

### File-Based Stats
```python
{
    "files_processed": 50,
    "chunks_created": 1500,
    "embeddings_generated": 1500,
    "embeddings_stored": 1500,
    "errors": 0
}
```

### Chunk-Based Stats
```python
{
    "files_processed": 50,
    "chunks_created": 1500,
    "chunks_processed": 1500,      # Additional metric
    "embeddings_generated": 1500,
    "embeddings_stored": 1500,
    "errors": 0
}
```

## Error Handling Differences

### File-Based Error Handling
- Errors are isolated to individual files
- Failed file doesn't affect other files in the batch
- Worker continues with remaining files

### Chunk-Based Error Handling
- Errors affect entire chunk batches
- Failed batch doesn't affect other batches
- More granular error reporting possible

## Memory Usage Patterns

### File-Based Memory Usage
```
Peak Memory = max_workers × average_file_chunks × chunk_size
```

### Chunk-Based Memory Usage
```
Peak Memory = total_chunks × chunk_size (during read phase)
            + max_workers × chunks_per_batch × chunk_size (during processing)
```

## Monitoring and Debugging

### Progress Tracking

**File-Based:**
- Tracks file batches completed
- Shows files/chunks/stored counts

**Chunk-Based:**
- Tracks chunk batches completed  
- Shows chunks processed/embedded/stored

### Debug Information

Both approaches provide:
- Worker-level logging
- Error details with context
- Processing statistics
- Performance metrics

## Migration Guide

### From File-Based to Chunk-Based

```python
# Old approach
stats = run_merged_rag_pipeline(
    directory_path=path,
    max_workers=4,
    files_per_batch=5
)

# New approach
stats = run_chunk_based_rag_pipeline(
    directory_path=path,
    max_workers=4,
    chunks_per_batch=50  # Roughly 5 files × 10 chunks/file
)
```

### Batch Size Conversion

```python
# Estimate chunks per batch based on average file size
average_chunks_per_file = 10
files_per_batch = 5
chunks_per_batch = files_per_batch * average_chunks_per_file  # = 50
```

## Best Practices

### 1. Choose the Right Approach
- Use file-based for < 20 files
- Use chunk-based for > 50 files
- Use hybrid for automatic selection

### 2. Optimize Batch Sizes
- File-based: 3-10 files per batch
- Chunk-based: 20-100 chunks per batch
- Adjust based on available memory

### 3. Worker Count Tuning
- Start with `cpu_count() // 2`
- Monitor resource usage
- Adjust based on I/O vs CPU bottlenecks

### 4. Memory Management
- Monitor peak memory usage
- Reduce batch sizes if memory constrained
- Consider processing in smaller groups for very large datasets

### 5. Error Recovery
- Implement retry logic for failed batches
- Log detailed error information
- Consider partial processing for large datasets

The chunk-based approach provides better load balancing and scalability for scenarios with many files or varying file sizes, while the file-based approach remains simpler and more efficient for smaller, more uniform datasets.