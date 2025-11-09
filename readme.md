# PS04 RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with hybrid search, ColBERT late interaction, BGE reranking, and LLM-powered answer generation.

## 🚀 Quick Start

### GUI (Recommended)
```powershell
# Double-click or run:
launch_gui.bat

# Or manually:
python launch_gui.py
```
Access at: http://localhost:7860

### CLI (Advanced)
```powershell
python cli.py --help
```

## 📋 Prerequisites

1. **Python 3.10+** with virtual environment
2. **Docker** - For Qdrant vector database
3. **Ollama** - For LLM generation

### Setup Steps

```powershell
# 1. Activate virtual environment
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# 4. Start Ollama
ollama serve

# 5. Pull LLM models
ollama pull llama3.1:8b
```

## 🖥️ Using the GUI

The Gradio GUI provides six main features:

### 1. Quick Query 🔍
**Most popular** - Ask a question and get an instant answer with sources.

**Example:**
```
Query: "What are the main components of RAG systems?"
Collection: ps04_10-11-2025_14_30
Model: llama3.1:8b
```

### 2. Embed Documents 📚
Index your document collection for retrieval.

**Example:**
```
Directory: D:\documents\corpus
Workers: 20
Chunk Size: 1 KB
```

### 3. Retrieve Documents 🔎
Process multiple queries from a JSON file.

**Input format (queries.json):**
```json
[
  {"query_num": "1", "query": "What is machine learning?"},
  {"query_num": "2", "query": "Explain neural networks"}
]
```

### 4. Generate Answers 🤖
Create AI answers from retrieval results.

### 5. Preprocess Files 🧹
Clean and normalize text files.

### 6. Export Results 📦
Package results for submission.

👉 **See [GUI_README.md](GUI_README.md) for detailed GUI documentation**

## 💻 Using the CLI

### Common Commands

#### Embed Documents
```powershell
python cli.py embed-main "D:\documents" --max-workers 20 --chunk-size-kb 1
```

#### Retrieve Documents
```powershell
python cli.py retrieve queries.json my_collection -o results.json --rerank
```

#### Generate Answers
```powershell
python cli.py generate results.json -o answers.json -m llama3.1:8b
```

#### Single Query (End-to-End)
```powershell
python cli.py query "What is RAG?" my_collection -m llama3.1:8b
```

#### Complete Pipeline
```powershell
python cli.py embed-retrieve
# Then follow the prompts
```

## 🏗️ Architecture

### Retrieval Pipeline
1. **Dense Embeddings** (BGE-base-en-v1.5) - Semantic similarity
2. **Sparse Embeddings** (BM25) - Keyword matching  
3. **ColBERT Late Interaction** - Token-level matching
4. **BGE Reranker** - Final relevance scoring
5. **Unique Filename Filtering** - Diverse sources

### Generation Pipeline
1. **Context Assembly** - Top-K chunks with metadata
2. **Prompt Engineering** - Grounded, citation-focused
3. **LLM Generation** (Ollama) - Factual answers
4. **Source Attribution** - Explicit citations

## 📁 Project Structure

```
ps04-rag-v2/
├── gui.py                 # Gradio web interface
├── launch_gui.py          # GUI launcher with checks
├── launch_gui.bat         # Windows quick launcher
├── cli.py                 # Command-line interface
├── rag/                   # Core modules
│   ├── pipeline.py        # Embedding & indexing
│   ├── retrieve.py        # Hybrid retrieval
│   ├── generate.py        # Answer generation
│   ├── evaluate.py        # Quality evaluation
│   └── preprocess.py      # Text cleaning
├── requirements.txt       # Python dependencies
├── GUI_README.md          # GUI documentation
└── readme.md             # This file
```

## 🎯 Workflows

### Workflow 1: Interactive Exploration
1. Launch GUI: `launch_gui.bat`
2. Use **Quick Query** tab
3. Experiment with different queries and models

### Workflow 2: Batch Processing
1. Embed documents (GUI or CLI)
2. Prepare queries JSON file
3. Retrieve documents (batch)
4. Generate answers (batch)
5. Export results

### Workflow 3: CLI Automation
```powershell
# Embed
python cli.py embed-main "D:\corpus" --max-workers 20

# Retrieve
python cli.py retrieve queries.json ps04_collection -o results.json

# Generate
python cli.py generate results.json -o answers.json

# Export
python cli.py export answers.json output/ --zip-name submission.zip
```

## ⚙️ Configuration

### Model Options
- `llama3.1:8b` - Balanced (recommended)
- `llama3.2:3b` - Fast, lighter
- `llama3.1:70b` - Best quality (GPU required)
- `mistral:7b` - Alternative
- `qwen2.5:7b` - Multilingual

### Performance Tuning
```powershell
# More workers = faster (but more memory)
--max-workers 30

# Smaller chunks = more granular (but slower)
--chunk-size-kb 0.5

# Higher top-k = more context (but slower generation)
--top-k 10
```

## 🔧 Advanced Features

### Reranking
Enable BGE reranker for better relevance:
```powershell
python cli.py retrieve queries.json collection --rerank
```

### Unique File Filtering
Return only one chunk per file:
```powershell
python cli.py retrieve queries.json collection --unique-files
```

### Custom Temperature
Control answer creativity:
```powershell
python cli.py generate results.json -t 0.1  # More factual
python cli.py generate results.json -t 0.7  # More creative
```

## 📊 Output Formats

### Retrieval Results
```json
[
  {
    "query_num": "1",
    "query": "What is RAG?",
    "response": ["doc1.txt", "doc2.txt"],
    "chunk_1_text": "...",
    "chunk_1_filename": "doc1.txt",
    "chunk_1_score": 0.85
  }
]
```

### Generated Answers
```json
[
  {
    "query_num": "1",
    "query": "What is RAG?",
    "answer": "RAG is...",
    "sources": ["doc1.txt", "doc2.txt"],
    "chunk_scores": [0.85, 0.78]
  }
]
```

## 🐛 Troubleshooting

### Qdrant Issues
```powershell
# Check if running
docker ps

# View logs
docker logs <container_id>

# Restart
docker restart <container_id>
```

### Ollama Issues
```powershell
# Check status
ollama list

# Pull model
ollama pull llama3.1:8b

# Restart service
ollama serve
```

### Memory Issues
- Reduce `--max-workers`
- Use smaller chunk size
- Process in smaller batches

## 📝 Notes & Tips

### Critical Query Numbers
Query nums requiring attention: 21, 29, 32, 57, 88, 7000, 7001, 7002, 7036, 7039, 7041, 7044, 7048, 7055, 7059, 7080, 7084, 7094

### Best Practices
1. **Start with GUI** for exploration
2. **Use CLI** for automation and batch processing
3. **Enable reranker** for better results
4. **Monitor memory** with large corpora
5. **Save collection names** for reuse

## 🔗 Resources

- **Ollama Installation**: https://github.com/ollama/ollama
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **GUI Guide**: See [GUI_README.md](GUI_README.md)

## 📄 License

See repository for licensing information.

---

**Quick Help:**
- GUI: Run `launch_gui.bat` or `python launch_gui.py`
- CLI: Run `python cli.py --help`
- Docs: Read `GUI_README.md` for detailed GUI instructions