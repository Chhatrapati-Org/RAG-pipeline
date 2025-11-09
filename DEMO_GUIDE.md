# 🎬 PS04 RAG System - Demo Guide

Quick demonstration guide for showcasing the system's capabilities.

## 🚀 Quick Demo (5 minutes)

### Prerequisites Check
```powershell
# Check services
docker ps                    # Qdrant should be running
ollama list                  # Models should be available
```

### Launch GUI
```powershell
# Option 1: Double-click
launch_gui.bat

# Option 2: Command line
python launch_gui.py
```

Wait for browser to open at `http://localhost:7860`

---

## 📖 Demo Scenario 1: Quick Query (Fastest)

**Goal**: Show instant question answering with sources

### Steps:

1. **Navigate to "🔍 Quick Query" tab** (default tab)

2. **Enter sample query:**
   ```
   What are the main components of a retrieval-augmented generation system?
   ```

3. **Fill in collection name:**
   - Use existing collection (e.g., `ps04_10-11-2025_14_30`)
   - Or create one first in "Embed Documents" tab

4. **Adjust settings (optional):**
   - Model: `llama3.1:8b`
   - Temperature: `0.3`
   - Max Chunks: `5`

5. **Click "🔍 Search & Generate Answer"**

6. **Observe results:**
   - Answer appears in ~10-20 seconds
   - Sources listed with relevance scores
   - Timing breakdown shown

### Expected Output:
```markdown
## Answer
Retrieval-augmented generation (RAG) systems consist of three main components:
1. **Retrieval Component**: Uses hybrid search combining dense embeddings...
2. **Generation Component**: Employs large language models...
3. **Integration Layer**: Combines retrieved context with query...

[More detailed answer with inline citations]

---
Query: What are the main components...
Chunks Used: 5
Retrieval Time: 2.3s
Generation Time: 8.7s
Total Time: 11.0s

## Sources
1. rag_architecture.txt (score: 0.8542)
2. hybrid_search_guide.txt (score: 0.7891)
3. llm_integration.txt (score: 0.7654)
```

---

## 📚 Demo Scenario 2: Document Indexing

**Goal**: Show how to index a document collection

### Steps:

1. **Navigate to "📚 Embed Documents" tab**

2. **Prepare sample documents:**
   ```powershell
   # Create sample directory
   mkdir demo_docs
   
   # Add some text files
   echo "Machine learning is a subset of AI..." > demo_docs/ml_basics.txt
   echo "Neural networks consist of layers..." > demo_docs/neural_nets.txt
   ```

3. **Fill in form:**
   - Directory Path: `demo_docs`
   - Workers: `10` (for demo)
   - Chunk Size: `1 KB`
   - Files per Batch: `10`

4. **Click "🚀 Start Embedding"**

5. **Watch progress:**
   - Progress bar shows status
   - Statistics update in real-time

6. **Note collection name:**
   - Auto-generated with timestamp
   - Example: `ps04_10-11-2025_15_45`
   - Use this for queries

### Expected Output:
```markdown
✅ Embedding Pipeline Completed!

📊 Statistics:
- Collection Name: ps04_10-11-2025_15_45
- Files Processed: 2
- Chunks Created: 8
- Embeddings Stored: 8
- Time Taken: 0.45 minutes
- Errors: 0
```

---

## 🔎 Demo Scenario 3: Batch Retrieval

**Goal**: Process multiple queries at once

### Steps:

1. **Create queries file** (`demo_queries.json`):
   ```json
   [
     {"query_num": "1", "query": "What is machine learning?"},
     {"query_num": "2", "query": "Explain neural networks"},
     {"query_num": "3", "query": "What are the types of AI?"}
   ]
   ```

2. **Navigate to "🔎 Retrieve Documents" tab**

3. **Fill in form:**
   - Queries File: `demo_queries.json`
   - Collection Name: `ps04_10-11-2025_15_45` (from step 2)
   - Output File: `demo_results.json`
   - Top-K Results: `5`
   - ✓ Unique Files Only
   - ✓ Use BGE Reranker

4. **Click "🔎 Start Retrieval"**

5. **Review results:**
   - Summary shows statistics
   - Preview shows first 5 results
   - Full results saved to JSON

### Expected Output:
```markdown
✅ Retrieval Completed!

📊 Statistics:
- Total Queries: 3
- Time Taken: 0.12 minutes
- Top-K per Query: 5
- Reranker: Enabled ✓
- Unique Files: Enabled ✓
- Output: demo_results.json
```

---

## 🤖 Demo Scenario 4: Answer Generation

**Goal**: Generate AI answers from retrieval results

### Steps:

1. **Navigate to "🤖 Generate Answers" tab**

2. **Fill in form:**
   - Results File: `demo_results.json` (from scenario 3)
   - Output File: `demo_answers.json`
   - Model: `llama3.1:8b`
   - Temperature: `0.3`
   - Max Chunks: `5`

3. **Click "🤖 Generate Answers"**

4. **Wait for generation:**
   - Progress bar tracks completion
   - ~3-10 seconds per query

5. **Review outputs:**
   - Summary with statistics
   - Preview of first 3 answers
   - Full answers saved to JSON

### Expected Output:
```markdown
✅ Answer Generation Completed!

📊 Statistics:
- Queries Processed: 3
- Model: llama3.1:8b
- Temperature: 0.3
- Max Chunks: 5
- Time Taken: 0.52 minutes
- Output: demo_answers.json

[Preview of answers with sources]
```

---

## 📦 Demo Scenario 5: Export Results

**Goal**: Package results for submission

### Steps:

1. **Navigate to "📦 Export Results" tab**

2. **Fill in form:**
   - Results File: `demo_results.json`
   - Output Directory: `export_demo`
   - Zip Archive Name: `demo_submission.zip`

3. **Click "📦 Export & Zip"**

4. **Check outputs:**
   - Individual JSON files created (`1.json`, `2.json`, `3.json`)
   - Zip archive created
   - Ready for submission

### Expected Output:
```markdown
✅ Export Completed!

📊 Statistics:
- Exported Files: 3
- Zip Archive: export_demo\demo_submission.zip
- Archive Size: 3 files
- Output Directory: export_demo
```

---

## 🎯 Complete End-to-End Demo (15 minutes)

### Full Workflow Demonstration

```powershell
# 1. Prepare documents (30 seconds)
mkdir demo_corpus
# Copy 10-20 text files to demo_corpus

# 2. Launch GUI (10 seconds)
python launch_gui.py

# 3. Embed documents (2 minutes)
#    - Tab: "📚 Embed Documents"
#    - Directory: demo_corpus
#    - Click "Start Embedding"
#    - Note collection name

# 4. Quick query test (30 seconds)
#    - Tab: "🔍 Quick Query"
#    - Enter question
#    - Use collection from step 3
#    - Click "Search & Generate Answer"

# 5. Batch processing (5 minutes)
#    - Create queries.json with 5-10 queries
#    - Tab: "🔎 Retrieve Documents"
#    - Process all queries
#    - Tab: "🤖 Generate Answers"
#    - Generate all answers

# 6. Export (1 minute)
#    - Tab: "📦 Export Results"
#    - Create zip archive
```

---

## 💡 Pro Tips for Demos

### Make It Impressive
1. **Use interesting queries** - Show domain-specific questions
2. **Highlight sources** - Emphasize the citation feature
3. **Compare temperatures** - Show 0.1 vs 0.7 creativity
4. **Show speed** - Mention parallel processing capabilities

### Common Demo Queries
```
1. "What are the security considerations for RAG systems?"
2. "Explain the difference between dense and sparse embeddings"
3. "How does ColBERT improve retrieval accuracy?"
4. "What are the main challenges in multi-hop question answering?"
5. "Compare traditional search with semantic search"
```

### Troubleshooting During Demo

**If Qdrant not running:**
```powershell
docker run -p 6333:6333 qdrant/qdrant
```

**If Ollama not responding:**
```powershell
ollama serve
```

**If collection not found:**
- Run embedding first
- Double-check collection name (case-sensitive)

### Performance Expectations

| Operation | Typical Time | Notes |
|-----------|-------------|-------|
| Quick Query | 10-20s | Depends on model |
| Embed 100 docs | 2-5 min | Depends on size |
| Retrieve 50 queries | 30-60s | With reranker |
| Generate 50 answers | 3-10 min | Depends on model |

---

## 📸 Screenshot Checklist

For documentation/presentation:

- [ ] GUI main interface (Quick Query tab)
- [ ] Sample query with answer and sources
- [ ] Embedding progress bar
- [ ] Retrieval results preview
- [ ] Generated answers preview
- [ ] Export summary

---

## 🎤 Demo Script (Verbal)

```
"Today I'll demonstrate our RAG system that combines hybrid search 
with AI-powered answer generation.

[Open GUI]
Here's the interface - we have six main tabs, but let's start with 
the most popular: Quick Query.

[Enter query]
I'll ask: 'What are the main components of RAG systems?'

[Show parameters]
We can choose different models - I'm using Llama 3.1 8B - and adjust
the temperature for more factual or creative answers.

[Click search]
The system is now:
1. Searching our document collection using hybrid search
2. Ranking results with ColBERT and reranker
3. Generating a comprehensive answer with the LLM

[Show results]
Here's our answer - notice it cites specific sources with relevance
scores. The entire process took just 11 seconds.

[Show other tabs briefly]
We also have full batch processing capabilities, document indexing,
preprocessing, and export features - all accessible through this
friendly interface.

Questions?"
```

---

## 🔗 Resources

- **GUI Documentation**: See `GUI_README.md`
- **Technical Details**: See main `readme.md`
- **CLI Commands**: Run `python cli.py --help`

**Happy Demoing! 🎉**
