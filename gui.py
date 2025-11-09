"""
Gradio GUI for PS04 RAG System
Provides a user-friendly interface for all RAG operations:
- Document embedding and indexing
- Query retrieval
- Answer generation
- End-to-end query interface
- Evaluation and export
"""

import json
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from qdrant_client import QdrantClient

from rag.evaluate import RAGEvaluator
from rag.generate import RAGGenerator, generate_single_answer
from rag.pipeline import run_chunk_based_rag_pipeline, run_merged_rag_pipeline
from rag.preprocess import preprocess_chunk_text
from rag.retrieve import run_multithreaded_retrieval, single_retrieval


# Global Qdrant client (will be reinitialized based on user input)
qdrant_client = None


def initialize_qdrant(url: str) -> Tuple[str, str]:
    """Initialize Qdrant client with given URL."""
    global qdrant_client
    try:
        qdrant_client = QdrantClient(url=url)
        # Test connection
        collections = qdrant_client.get_collections()
        return f"✅ Connected to Qdrant at {url}", f"Available collections: {len(collections.collections)}"
    except Exception as e:
        return f"❌ Failed to connect: {str(e)}", ""


def embed_documents(
    directory_path: str,
    qdrant_url: str,
    max_workers: int,
    chunk_size_kb: float,
    files_per_batch: int,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Embed and index documents."""
    try:
        # Initialize client
        global qdrant_client
        qdrant_client = QdrantClient(url=qdrant_url)
        
        progress(0, desc="Starting embedding pipeline...")
        
        start = time.time()
        stats, collection_name = run_merged_rag_pipeline(
            qdrant_client=qdrant_client,
            directory_path=directory_path,
            max_workers=max_workers,
            chunk_size_kb=int(chunk_size_kb),
            files_per_batch=files_per_batch,
        )
        end = time.time()
        
        stats['time_taken'] = (end - start) / 60
        stats['collection_name'] = collection_name
        
        progress(1.0, desc="Embedding completed!")
        
        summary = f"""
✅ **Embedding Pipeline Completed!**

📊 **Statistics:**
- Collection Name: `{collection_name}`
- Files Processed: {stats.get('files_processed', 0)}
- Chunks Created: {stats.get('chunks_created', 0)}
- Embeddings Stored: {stats.get('embeddings_stored', 0)}
- Time Taken: {stats['time_taken']:.2f} minutes
- Errors: {stats.get('errors', 0)}
"""
        
        return summary, json.dumps(stats, indent=2)
        
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


def retrieve_documents(
    queries_file_path: str,
    collection_name: str,
    qdrant_url: str,
    output_file_path: str,
    max_workers: int,
    top_k: int,
    queries_per_batch: int,
    unique_files: bool,
    use_reranker: bool,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Retrieve relevant documents for queries."""
    try:
        # Initialize client
        global qdrant_client
        qdrant_client = QdrantClient(url=qdrant_url)
        
        progress(0, desc="Starting retrieval...")
        
        start = time.time()
        results = run_multithreaded_retrieval(
            COLLECTION_NAME=collection_name,
            qdrant_client=qdrant_client,
            queries_file_path=queries_file_path,
            output_file_path=output_file_path,
            max_workers=max_workers,
            top_k=top_k,
            queries_per_batch=queries_per_batch,
            unique_per_filename=unique_files,
            use_reranker=use_reranker,
        )
        end = time.time()
        
        progress(1.0, desc="Retrieval completed!")
        
        summary = f"""
✅ **Retrieval Completed!**

📊 **Statistics:**
- Total Queries: {len(results)}
- Time Taken: {(end - start) / 60:.2f} minutes
- Top-K per Query: {top_k}
- Reranker: {'Enabled ✓' if use_reranker else 'Disabled ✗'}
- Unique Files: {'Enabled ✓' if unique_files else 'Disabled ✗'}
- Output: `{output_file_path}`
"""
        
        return summary, json.dumps(results[:5], indent=2, ensure_ascii=False) + "\n\n... (showing first 5 results)"
        
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


def generate_answers(
    results_file: str,
    output_file: Optional[str],
    model: str,
    temperature: float,
    max_chunks: int,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Generate answers from retrieval results."""
    try:
        progress(0, desc="Initializing generator...")
        
        start = time.time()
        generator = RAGGenerator(model_name=model, temperature=temperature)
        
        progress(0.2, desc="Generating answers...")
        results = generator.generate(
            data_or_path=results_file,
            output_path=output_file,
            max_chunks=max_chunks
        )
        end = time.time()
        
        progress(1.0, desc="Generation completed!")
        
        summary = f"""
✅ **Answer Generation Completed!**

📊 **Statistics:**
- Queries Processed: {len(results)}
- Model: {model}
- Temperature: {temperature}
- Max Chunks: {max_chunks}
- Time Taken: {(end - start) / 60:.2f} minutes
- Output: `{output_file if output_file else 'Console only'}`
"""
        
        # Show first 3 answers
        preview = []
        for i, result in enumerate(results[:3], 1):
            preview.append(f"**Query {result['query_num']}:** {result['query']}")
            preview.append(f"**Answer:** {result['answer'][:300]}...")
            preview.append(f"**Sources:** {', '.join(result['sources'][:3])}")
            preview.append("---")
        
        return summary, "\n\n".join(preview)
        
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


def query_single(
    query: str,
    collection_name: str,
    qdrant_url: str,
    model: str,
    temperature: float,
    max_chunks: int,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """Process a single query end-to-end."""
    try:
        # Initialize client
        global qdrant_client
        qdrant_client = QdrantClient(url=qdrant_url)
        
        progress(0, desc="Retrieving documents...")
        
        start = time.time()
        retrieval_result = single_retrieval(
            COLLECTION_NAME=collection_name,
            qdrant_client=qdrant_client,
            query=query,
            max_workers=10,
            top_k=5,
            queries_per_batch=20,
        )
        retrieval_time = time.time()
        
        progress(0.5, desc="Generating answer...")
        
        answer_result = generate_single_answer(
            retrieval_result=retrieval_result,
            model_name=model,
            temperature=temperature,
            max_chunks=max_chunks
        )
        
        end = time.time()
        
        progress(1.0, desc="Query completed!")
        
        # Format answer
        answer_text = f"""
## Answer

{answer_result['answer']}

---

**Query:** {query}  
**Chunks Used:** {answer_result['num_chunks_used']}  
**Retrieval Time:** {(retrieval_time - start):.2f}s  
**Generation Time:** {(end - retrieval_time):.2f}s  
**Total Time:** {(end - start):.2f}s
"""
        
        # Format sources
        sources_text = "## Sources\n\n"
        for i, source in enumerate(answer_result.get('sources', []), 1):
            score = answer_result.get('chunk_scores', [0])[i-1] if i-1 < len(answer_result.get('chunk_scores', [])) else 0
            sources_text += f"{i}. **{source}** (score: {score:.4f})\n"
        
        # Format raw JSON
        raw_json = json.dumps(answer_result, indent=2, ensure_ascii=False)
        
        return answer_text, sources_text, raw_json
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, "", ""


def preprocess_files(
    input_dir: str,
    output_dir: str,
    glob_pattern: str,
    max_workers: int,
    files_per_batch: int,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Preprocess text files."""
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        src = Path(input_dir)
        dst = Path(output_dir)
        
        if not src.exists() or not src.is_dir():
            return f"❌ Input directory not found: {input_dir}", ""
        
        dst.mkdir(parents=True, exist_ok=True)
        
        progress(0, desc="Finding files...")
        files = sorted(src.glob(glob_pattern))
        
        if not files:
            return "⚠️ No files matched the pattern.", ""
        
        def _preprocess_batch(batch):
            success_count = 0
            failed_files = []
            
            for fp in batch:
                if not fp.is_file():
                    continue
                
                try:
                    with fp.open("r", encoding="utf-8", errors="ignore") as f:
                        raw = f.read()
                    processed = preprocess_chunk_text(raw)
                    out_fp = dst / fp.name
                    out_fp.parent.mkdir(parents=True, exist_ok=True)
                    with out_fp.open("w", encoding="utf-8") as f:
                        f.write(processed)
                    success_count += 1
                except Exception as e:
                    failed_files.append(f"{fp.name}: {e}")
            
            return success_count, failed_files
        
        batches = [files[i:i + files_per_batch] for i in range(0, len(files), files_per_batch)]
        
        processed_count = 0
        failed_files = []
        
        progress(0.1, desc=f"Processing {len(files)} files in {len(batches)} batches...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(_preprocess_batch, batch): batch 
                for batch in batches
            }
            
            completed = 0
            for future in as_completed(future_to_batch):
                success_count, batch_failed = future.result()
                processed_count += success_count
                failed_files.extend(batch_failed)
                completed += 1
                progress(0.1 + (0.9 * completed / len(batches)), desc=f"Processed {processed_count}/{len(files)} files...")
        
        progress(1.0, desc="Preprocessing completed!")
        
        summary = f"""
✅ **Preprocessing Completed!**

📊 **Statistics:**
- Files Processed: {processed_count}/{len(files)}
- Failed: {len(failed_files)}
- Output Directory: `{output_dir}`
"""
        
        failed_text = ""
        if failed_files:
            failed_text = "\n\n**Failed Files:**\n" + "\n".join(f"- {f}" for f in failed_files[:10])
            if len(failed_files) > 10:
                failed_text += f"\n... and {len(failed_files) - 10} more"
        
        return summary + failed_text, ""
        
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


def export_results(
    results_file: str,
    output_dir: str,
    zip_name: str,
    progress=gr.Progress()
) -> str:
    """Export results to individual JSON files and zip."""
    try:
        import zipfile
        
        results_path = Path(results_file)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        progress(0, desc="Loading results...")
        
        with results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return "❌ Results file must contain a list of results"
        
        progress(0.2, desc=f"Exporting {len(data)} queries...")
        
        exported_count = 0
        for i, item in enumerate(data):
            query_num = item.get("query_num")
            if query_num is None:
                continue
            
            new_item = {
                'query': item.get('query'),
                'response': item.get('response')
            }
            
            output_file = out_dir / f"{query_num}.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(new_item, f, ensure_ascii=False, indent=4)
            
            exported_count += 1
            progress(0.2 + (0.5 * i / len(data)), desc=f"Exported {exported_count} files...")
        
        progress(0.7, desc="Creating zip archive...")
        
        zip_path = out_dir / zip_name
        if not zip_path.name.lower().endswith(".zip"):
            zip_path = zip_path.with_suffix(".zip")
        
        json_files = [f for f in out_dir.iterdir() if f.suffix.lower() == ".json"]
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_ref:
            for i, file in enumerate(json_files):
                zip_ref.write(file, arcname=file.name)
                progress(0.7 + (0.3 * i / len(json_files)), desc=f"Compressing {i+1}/{len(json_files)} files...")
        
        progress(1.0, desc="Export completed!")
        
        return f"""
✅ **Export Completed!**

📊 **Statistics:**
- Exported Files: {exported_count}
- Zip Archive: `{zip_path}`
- Archive Size: {len(json_files)} files
- Output Directory: `{output_dir}`
"""
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Build Gradio Interface
with gr.Blocks(title="PS04 RAG System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 PS04 RAG System
    ### Hybrid Search + ColBERT + BGE Reranker + LLM Generation
    
    A complete Retrieval-Augmented Generation system with document indexing, hybrid search, and answer generation.
    """)
    
    # Qdrant Connection Section
    with gr.Accordion("⚙️ Qdrant Configuration", open=False):
        with gr.Row():
            qdrant_url_input = gr.Textbox(
                label="Qdrant URL",
                value="http://localhost:6333",
                placeholder="http://localhost:6333"
            )
            connect_btn = gr.Button("Connect", variant="primary")
        
        connection_status = gr.Textbox(label="Status", interactive=False)
        connection_details = gr.Textbox(label="Details", interactive=False)
        
        connect_btn.click(
            fn=initialize_qdrant,
            inputs=[qdrant_url_input],
            outputs=[connection_status, connection_details]
        )
    
    # Main Tabs
    with gr.Tabs():
        # Tab 1: Quick Query (Most Used)
        with gr.Tab("🔍 Quick Query"):
            gr.Markdown("### Ask a question and get an AI-generated answer with sources")
            
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Enter your question here...",
                        lines=3
                    )
                    query_collection = gr.Textbox(
                        label="Collection Name",
                        placeholder="e.g., ps04_10-11-2025_14_30"
                    )
                    
                    with gr.Row():
                        query_model = gr.Dropdown(
                            label="Model",
                            choices=["llama3.1:8b", "llama3.2:3b", "llama3.1:70b", "mistral:7b", "qwen2.5:7b"],
                            value="llama3.1:8b"
                        )
                        query_temp = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1
                        )
                        query_max_chunks = gr.Slider(
                            label="Max Chunks",
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1
                        )
                    
                    query_btn = gr.Button("🔍 Search & Generate Answer", variant="primary", size="lg")
                
                with gr.Column(scale=3):
                    query_answer = gr.Markdown(label="Answer")
                    query_sources = gr.Markdown(label="Sources")
            
            with gr.Accordion("Raw JSON Output", open=False):
                query_raw = gr.JSON(label="Raw Result")
            
            query_btn.click(
                fn=query_single,
                inputs=[
                    query_input,
                    query_collection,
                    qdrant_url_input,
                    query_model,
                    query_temp,
                    query_max_chunks
                ],
                outputs=[query_answer, query_sources, query_raw]
            )
        
        # Tab 2: Embed Documents
        with gr.Tab("📚 Embed Documents"):
            gr.Markdown("### Index documents for retrieval (creates embeddings)")
            
            with gr.Row():
                embed_dir = gr.Textbox(
                    label="Directory Path",
                    placeholder="Path to directory containing documents",
                    scale=3
                )
            
            with gr.Row():
                embed_workers = gr.Slider(
                    label="Workers",
                    minimum=1,
                    maximum=50,
                    value=20,
                    step=1
                )
                embed_chunk_size = gr.Slider(
                    label="Chunk Size (KB)",
                    minimum=0.5,
                    maximum=10,
                    value=1,
                    step=0.5
                )
                embed_batch_size = gr.Slider(
                    label="Files per Batch",
                    minimum=5,
                    maximum=100,
                    value=20,
                    step=5
                )
            
            embed_btn = gr.Button("🚀 Start Embedding", variant="primary", size="lg")
            
            embed_summary = gr.Markdown(label="Summary")
            embed_details = gr.JSON(label="Detailed Statistics")
            
            embed_btn.click(
                fn=embed_documents,
                inputs=[
                    embed_dir,
                    qdrant_url_input,
                    embed_workers,
                    embed_chunk_size,
                    embed_batch_size
                ],
                outputs=[embed_summary, embed_details]
            )
        
        # Tab 3: Retrieve Documents
        with gr.Tab("🔎 Retrieve Documents"):
            gr.Markdown("### Retrieve relevant documents for multiple queries")
            
            with gr.Row():
                retrieve_queries_file = gr.Textbox(
                    label="Queries File (JSON)",
                    placeholder="Path to queries JSON file"
                )
                retrieve_collection = gr.Textbox(
                    label="Collection Name",
                    placeholder="e.g., ps04_10-11-2025_14_30"
                )
            
            with gr.Row():
                retrieve_output = gr.Textbox(
                    label="Output File",
                    value="results.json",
                    placeholder="results.json"
                )
            
            with gr.Row():
                retrieve_workers = gr.Slider(
                    label="Workers",
                    minimum=1,
                    maximum=50,
                    value=20,
                    step=1
                )
                retrieve_top_k = gr.Slider(
                    label="Top-K Results",
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1
                )
                retrieve_batch = gr.Slider(
                    label="Queries per Batch",
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=10
                )
            
            with gr.Row():
                retrieve_unique = gr.Checkbox(
                    label="Unique Files Only",
                    value=True
                )
                retrieve_reranker = gr.Checkbox(
                    label="Use BGE Reranker",
                    value=True
                )
            
            retrieve_btn = gr.Button("🔎 Start Retrieval", variant="primary", size="lg")
            
            retrieve_summary = gr.Markdown(label="Summary")
            retrieve_preview = gr.JSON(label="Results Preview (first 5)")
            
            retrieve_btn.click(
                fn=retrieve_documents,
                inputs=[
                    retrieve_queries_file,
                    retrieve_collection,
                    qdrant_url_input,
                    retrieve_output,
                    retrieve_workers,
                    retrieve_top_k,
                    retrieve_batch,
                    retrieve_unique,
                    retrieve_reranker
                ],
                outputs=[retrieve_summary, retrieve_preview]
            )
        
        # Tab 4: Generate Answers
        with gr.Tab("🤖 Generate Answers"):
            gr.Markdown("### Generate answers from retrieval results using LLM")
            
            with gr.Row():
                gen_results_file = gr.Textbox(
                    label="Retrieval Results File (JSON)",
                    placeholder="Path to retrieval results JSON"
                )
                gen_output_file = gr.Textbox(
                    label="Output File (Optional)",
                    placeholder="Leave empty for console only"
                )
            
            with gr.Row():
                gen_model = gr.Dropdown(
                    label="Model",
                    choices=["llama3.1:8b", "llama3.2:3b", "llama3.1:70b", "mistral:7b", "qwen2.5:7b"],
                    value="llama3.1:8b"
                )
                gen_temp = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1
                )
                gen_max_chunks = gr.Slider(
                    label="Max Chunks",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1
                )
            
            gen_btn = gr.Button("🤖 Generate Answers", variant="primary", size="lg")
            
            gen_summary = gr.Markdown(label="Summary")
            gen_preview = gr.Markdown(label="Preview (first 3 answers)")
            
            gen_btn.click(
                fn=generate_answers,
                inputs=[
                    gen_results_file,
                    gen_output_file,
                    gen_model,
                    gen_temp,
                    gen_max_chunks
                ],
                outputs=[gen_summary, gen_preview]
            )
        
        # Tab 5: Preprocess Files
        with gr.Tab("🧹 Preprocess Files"):
            gr.Markdown("### Clean and normalize text files")
            
            with gr.Row():
                preprocess_input = gr.Textbox(
                    label="Input Directory",
                    placeholder="Path to input directory"
                )
                preprocess_output = gr.Textbox(
                    label="Output Directory",
                    placeholder="Path to output directory"
                )
            
            with gr.Row():
                preprocess_glob = gr.Textbox(
                    label="File Pattern (Glob)",
                    value="*",
                    placeholder="e.g., *.txt or *"
                )
                preprocess_workers = gr.Slider(
                    label="Workers",
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1
                )
                preprocess_batch = gr.Slider(
                    label="Files per Batch",
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5
                )
            
            preprocess_btn = gr.Button("🧹 Start Preprocessing", variant="primary", size="lg")
            
            preprocess_summary = gr.Markdown(label="Summary")
            
            preprocess_btn.click(
                fn=preprocess_files,
                inputs=[
                    preprocess_input,
                    preprocess_output,
                    preprocess_glob,
                    preprocess_workers,
                    preprocess_batch
                ],
                outputs=[preprocess_summary, gr.Textbox(visible=False)]
            )
        
        # Tab 6: Export Results
        with gr.Tab("📦 Export Results"):
            gr.Markdown("### Export retrieval results to individual JSON files and zip")
            
            with gr.Row():
                export_input = gr.Textbox(
                    label="Results File (JSON)",
                    placeholder="Path to results JSON file"
                )
                export_output_dir = gr.Textbox(
                    label="Output Directory",
                    placeholder="Path to output directory"
                )
            
            with gr.Row():
                export_zip_name = gr.Textbox(
                    label="Zip Archive Name",
                    value="Astraq Cyber Defence_PS4.zip",
                    placeholder="output.zip"
                )
            
            export_btn = gr.Button("📦 Export & Zip", variant="primary", size="lg")
            
            export_summary = gr.Markdown(label="Summary")
            
            export_btn.click(
                fn=export_results,
                inputs=[
                    export_input,
                    export_output_dir,
                    export_zip_name
                ],
                outputs=[export_summary]
            )
    
    # Footer
    gr.Markdown("""
    ---
    ### 📖 Quick Guide
    
    1. **Quick Query**: Ask a single question and get an immediate answer with sources
    2. **Embed Documents**: Index your document collection for retrieval
    3. **Retrieve Documents**: Find relevant documents for multiple queries
    4. **Generate Answers**: Create AI-generated answers from retrieval results
    5. **Preprocess Files**: Clean and normalize text files before embedding
    6. **Export Results**: Export results to individual JSON files and create zip archive
    
    **Note**: Make sure Qdrant and Ollama services are running before use.
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
