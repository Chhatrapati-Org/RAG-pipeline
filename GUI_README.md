# 🖥️ PS04 RAG System - Gradio GUI

A user-friendly web interface for the PS04 RAG (Retrieval-Augmented Generation) system.

## 🚀 Quick Start

### Prerequisites

1. **Qdrant** running on `http://localhost:6333`
   ```powershell
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Ollama** running locally
   ```powershell
   ollama serve
   ```

3. **Python Dependencies**
   ```powershell
   pip install gradio
   ```

### Launch the GUI

```powershell
python gui.py
```

The interface will open at `http://localhost:7860`

## 📋 Features

### 🔍 Quick Query
**Most commonly used feature** - Ask a single question and get an immediate AI-generated answer with sources.

- Enter your question in natural language
- Select collection name (from previous embedding runs)
- Choose model and parameters
- Get instant answer with source citations
- View retrieval and generation timing

**Example:**
```
Query: "What are the key features of RAG systems?"
Collection: ps04_10-11-2025_14_30
Model: llama3.1:8b
Temperature: 0.3
```

### 📚 Embed Documents
Index your document collection for retrieval.

**Parameters:**
- **Directory Path**: Path to folder containing documents
- **Workers**: Number of parallel threads (default: 20)
- **Chunk Size**: Size of text chunks in KB (default: 1)
- **Files per Batch**: Number of files to process per batch (default: 20)

**Output:**
- Collection name (auto-generated with timestamp)
- Statistics: files processed, chunks created, embeddings stored
- Processing time

### 🔎 Retrieve Documents
Find relevant documents for multiple queries from a JSON file.

**Input Format** (queries.json):
```json
[
  {"query_num": "1", "query": "What is machine learning?"},
  {"query_num": "2", "query": "Explain neural networks"}
]
```

**Parameters:**
- **Queries File**: Path to JSON file with queries
- **Collection Name**: Qdrant collection to search
- **Top-K Results**: Number of results per query (default: 5)
- **Unique Files Only**: Return one result per unique filename
- **Use BGE Reranker**: Enable reranking for better relevance

**Output:**
- Results JSON file with retrieved chunks
- Preview of first 5 results
- Processing statistics

### 🤖 Generate Answers
Create AI-generated answers from retrieval results.

**Parameters:**
- **Results File**: Output from retrieval step
- **Output File**: Where to save answers (optional)
- **Model**: Ollama model to use
- **Temperature**: 0.0 (deterministic) to 1.0 (creative)
- **Max Chunks**: Maximum context chunks to use

**Output:**
- JSON file with queries and generated answers
- Preview of first 3 answers
- Source citations for each answer

### 🧹 Preprocess Files
Clean and normalize text files before embedding.

**Parameters:**
- **Input Directory**: Raw text files location
- **Output Directory**: Where to save cleaned files
- **File Pattern**: Glob pattern (e.g., `*.txt`, `*`)
- **Workers**: Parallel processing threads

**Operations:**
- Remove extra whitespace
- Normalize line endings
- Clean special characters
- Fix encoding issues

### 📦 Export Results
Export retrieval results to individual JSON files and create a zip archive.

**Parameters:**
- **Results File**: Retrieval results JSON
- **Output Directory**: Where to save individual files
- **Zip Archive Name**: Name for the zip file

**Output:**
- One JSON file per query (named by query number)
- Zip archive containing all JSON files
- Ready for submission or sharing

## 🎯 Typical Workflows

### Workflow 1: First-Time Setup
1. **Preprocess Files** (optional) - Clean your documents
2. **Embed Documents** - Index your corpus
3. **Quick Query** - Test with sample questions

### Workflow 2: Batch Processing
1. **Embed Documents** - Index your corpus
2. **Retrieve Documents** - Process multiple queries from JSON
3. **Generate Answers** - Create answers for all queries
4. **Export Results** - Package for submission

### Workflow 3: Interactive Exploration
1. Use **Quick Query** tab repeatedly
2. Experiment with different models and parameters
3. Compare results across queries

## ⚙️ Configuration

### Qdrant Configuration
- Default URL: `http://localhost:6333`
- Click "Connect" to test connection
- View available collections

### Model Options
- `llama3.1:8b` - Balanced (recommended)
- `llama3.2:3b` - Fast, lighter answers
- `llama3.1:70b` - Best quality (requires powerful GPU)
- `mistral:7b` - Alternative model
- `qwen2.5:7b` - Multilingual support

### Temperature Guide
- **0.0-0.3**: Factual, consistent answers (recommended for RAG)
- **0.4-0.6**: Balanced creativity and consistency
- **0.7-1.0**: More creative, diverse answers

## 🔧 Advanced Features

### Hybrid Search Pipeline
The system uses a sophisticated retrieval pipeline:
1. **Dense Embeddings** (BGE) - Semantic similarity
2. **Sparse Embeddings** (BM25) - Keyword matching
3. **ColBERT Late Interaction** - Fine-grained token matching
4. **BGE Reranker** - Final relevance scoring

### Unique File Filtering
When enabled, returns only the highest-scoring chunk per unique filename, ensuring diverse sources across different documents.

### Progress Tracking
All long-running operations show:
- Real-time progress bars
- Current operation status
- Estimated completion

## 📊 Output Formats

### Quick Query Output
```markdown
## Answer
[AI-generated answer with inline citations]

---
Query: [original question]
Chunks Used: 5
Retrieval Time: 2.3s
Generation Time: 8.7s
Total Time: 11.0s

## Sources
1. document1.txt (score: 0.8542)
2. document2.txt (score: 0.7891)
...
```

### Retrieval Results JSON
```json
[
  {
    "query_num": "1",
    "query": "What is RAG?",
    "response": ["doc1.txt", "doc2.txt"],
    "chunk_1_text": "...",
    "chunk_1_filename": "doc1.txt",
    "chunk_1_score": 0.85,
    "chunk_1_chunk_id": "12345"
  }
]
```

### Generated Answers JSON
```json
[
  {
    "query_num": "1",
    "query": "What is RAG?",
    "answer": "RAG (Retrieval-Augmented Generation) is...",
    "num_chunks_used": 5,
    "sources": ["doc1.txt", "doc2.txt"],
    "chunk_scores": [0.85, 0.78]
  }
]
```

## 🐛 Troubleshooting

### "Failed to connect to Qdrant"
- Ensure Docker container is running: `docker ps`
- Check port 6333 is not blocked by firewall
- Verify URL in configuration section

### "Ollama model not found"
- Pull the model first: `ollama pull llama3.1:8b`
- Check Ollama is running: `ollama list`

### "Collection not found"
- Run embedding pipeline first
- Use exact collection name (case-sensitive)
- Check Qdrant collections: http://localhost:6333/dashboard

### Slow performance
- Reduce number of workers
- Use smaller chunk size
- Try a smaller/faster model
- Enable GPU acceleration (CUDA)

## 💡 Tips

1. **Start with Quick Query** to test your setup
2. **Use meaningful collection names** (include date/version)
3. **Experiment with temperature** to find the right balance
4. **Enable reranker** for better results (slightly slower)
5. **Monitor memory usage** with large document sets
6. **Save outputs regularly** during batch processing

## 🔗 Related Files

- `cli.py` - Command-line interface (alternative to GUI)
- `rag/` - Core RAG modules
- `requirements.txt` - Python dependencies

## 📝 Notes

- All operations are asynchronous with progress tracking
- Results are saved automatically when output paths are provided
- The GUI maintains connection to Qdrant throughout the session
- Multiple tabs can be used simultaneously (be careful with resource usage)

---

**Need help?** Check the Quick Guide section at the bottom of the GUI or refer to the main README.md
