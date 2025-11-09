import json
import re
import zipfile
from pathlib import Path
from typing import List, Optional
import time
import typer
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient

from rag.evaluate import RAGEvaluator
from rag.generate import RAGGenerator, generate_single_answer
from rag.pipeline import run_chunk_based_rag_pipeline, run_merged_rag_pipeline
from rag.preprocess import preprocess_chunk_text
from rag.retrieve import run_multithreaded_retrieval, single_retrieval

app = typer.Typer(help="CLI for PS04 RAG: ingest, retrieve, preprocess, and export.")

qdrant_client = QdrantClient(url="http://localhost:6333")


@app.command("embed-main")
def ingest_merged(
    directory_path: str = typer.Argument(..., help="Directory containing input files"),
    max_workers: int = typer.Option(20, help="Number of worker threads"),
    chunk_size_kb: int = typer.Option(1, help="Max chunk size in KB"),
    files_per_batch: int = typer.Option(20, help="Files per worker batch"),
):
    """
    Embed and index documents using hybrid search (dense + sparse + late interaction).
    This is the main ingestion pipeline for the RAG system.
    """
    typer.echo("🚀 Starting document embedding pipeline...\n")
    typer.echo(f"📁 Input directory: {directory_path}")
    typer.echo(f"⚙️  Workers: {max_workers}")
    typer.echo(f"📦 Chunk size: {chunk_size_kb} KB")
    typer.echo(f"📊 Files per batch: {files_per_batch}\n")
    
    start = time.time()
    stats, _ = run_merged_rag_pipeline(
        qdrant_client=qdrant_client,
        directory_path=directory_path,
        max_workers=max_workers,
        chunk_size_kb=chunk_size_kb,
        files_per_batch=files_per_batch,
    )
    end = time.time()
    stats['time_taken'] = (end - start) / 60  # in minutes
    
    typer.echo("\n✅ Embedding pipeline completed!\n")
    typer.echo("📊 Pipeline Statistics:")
    typer.echo(f"  • Total files processed: {stats.get('total_files', 0)}")
    typer.echo(f"  • Total chunks created: {stats.get('total_chunks', 0)}")
    typer.echo(f"  • Time taken: {stats['time_taken']:.2f} minutes")
    typer.echo(f"\n💾 Full statistics:\n{json.dumps(stats, indent=2)}")


@app.command("embed-chunks")
def ingest_chunks(
    directory_path: str = typer.Argument(..., help="Directory containing input files"),
    max_workers: int = typer.Option(20, help="Number of worker threads"),
    chunk_size_kb: int = typer.Option(1, help="Max chunk size in KB"),
    chunks_per_batch: int = typer.Option(50, help="Chunks per worker batch"),
):
    """
    Embed and index documents using chunk-based processing.
    Alternative pipeline that processes chunks directly instead of files.
    """
    typer.echo("🚀 Starting chunk-based embedding pipeline...\n")
    typer.echo(f"📁 Input directory: {directory_path}")
    typer.echo(f"⚙️  Workers: {max_workers}")
    typer.echo(f"📦 Chunk size: {chunk_size_kb} KB")
    typer.echo(f"📊 Chunks per batch: {chunks_per_batch}\n")
    
    start = time.time()
    stats = run_chunk_based_rag_pipeline(
        qdrant_client=qdrant_client,
        directory_path=directory_path,
        max_workers=max_workers,
        chunk_size_kb=chunk_size_kb,
        chunks_per_batch=chunks_per_batch,
    )
    end = time.time()
    stats['time_taken'] = (end - start) / 60  # in minutes
    
    typer.echo("\n✅ Chunk-based embedding completed!\n")
    typer.echo("📊 Pipeline Statistics:")
    typer.echo(f"  • Total chunks processed: {stats.get('total_chunks', 0)}")
    typer.echo(f"  • Time taken: {stats['time_taken']:.2f} minutes")
    typer.echo(f"\n💾 Full statistics:\n{json.dumps(stats, indent=2)}")

# TODO: Run DOCKER image of Qdrant if not running
@app.command("retrieve")
def retrieve(
    queries_file_path: str = typer.Argument(..., help="Path to queries JSON file"),
    collection_name: str = typer.Argument(..., help="Qdrant collection name"),
    output_file_path: str = typer.Option(
        "results.json", "--output", "-o", help="Path to save retrieval results JSON"
    ),
    max_workers: int = typer.Option(20, help="Number of worker threads"),
    top_k: int = typer.Option(5, help="Top-k similar chunks per query"),
    queries_per_batch: int = typer.Option(50, help="Queries per worker batch"),
    unique_files: bool = typer.Option(
        True, "--unique-files/--allow-duplicates", 
        help="Return only highest scoring chunk per unique filename"
    ),
    use_reranker: bool = typer.Option(
        True, "--rerank/--no-rerank",
        help="Use BGE reranker to improve relevance scoring"
    ),
):
    """
    Retrieve relevant documents for queries using hybrid search.
    
    Retrieval Pipeline:
    1. Hybrid search (dense + sparse embeddings)
    2. ColBERT late interaction scoring
    3. BGE reranker (optional, enabled by default)
    4. Unique filename filtering (optional, enabled by default)
    
    Examples:
        # Default: with reranking and unique files
        python cli.py retrieve queries.json my_collection -o results.json
        
        # Without reranking
        python cli.py retrieve queries.json my_collection --no-rerank
        
        # Allow duplicate filenames
        python cli.py retrieve queries.json my_collection --allow-duplicates
    """
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
    typer.echo(json.dumps(results, indent=2, ensure_ascii=False))
    typer.echo(f"Time taken: {(end - start) / 60:.2f} minutes")

@app.command("embed-retrieve")
def embed_retrieve():
    """
    Combined pipeline: Embed documents and then retrieve results for queries.
    This command runs both ingestion and retrieval in sequence.
    """
    directory_path = input("Enter the Directory containing input files: ").strip('\"')
    queries_file_path = input("Enter the Path to queries JSON file: ").strip('\"')
    output_file_path = input("Enter the Path to save retrieval results JSON (Do not skip): ").strip('\"')

    typer.echo("\n" + "="*60)
    typer.echo("STEP 1: EMBEDDING DOCUMENTS")
    typer.echo("="*60)
    
    start = time.time()
    stats, collection_name = run_merged_rag_pipeline(
        qdrant_client=qdrant_client,
        directory_path=directory_path,
        max_workers=15,
        chunk_size_kb=0.5,
        files_per_batch=20,
    )
    end = time.time()
    stats['time_taken'] = (end - start) / 60  # in minutes
    typer.echo("\n✅ Embedding completed:")
    typer.echo(json.dumps(stats, indent=2))
    
    typer.echo("\n" + "="*60)
    typer.echo("STEP 2: RETRIEVING DOCUMENTS")
    typer.echo("="*60)
    
    init = time.time()
    results = run_multithreaded_retrieval(
        COLLECTION_NAME=collection_name,
        qdrant_client=qdrant_client,
        queries_file_path=queries_file_path,
        output_file_path=output_file_path,
        max_workers=20,
        top_k=5,
        queries_per_batch=20,
    )
    fin = time.time()
    
    typer.echo(f"\n✅ Retrieval completed in {(fin - init) / 60:.2f} minutes")
    typer.echo(f"📊 Total time (embedding + retrieval): {(fin - start) / 60:.2f} minutes")
    typer.echo(f"💾 Results saved to: {output_file_path}")


# TODO: Start the ollama server if not running
@app.command("generate")
def generate(
    results_file: str = typer.Argument(..., help="Path to retrieval results JSON"),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Path to save generated answers JSON"
    ),
    model: str = typer.Option("llama3.1:8b", "--model", "-m", help="Ollama model name"),
    temperature: float = typer.Option(0.3, "--temperature", "-t", help="Sampling temperature (0.0-1.0)"),
    max_chunks: int = typer.Option(5, "--max-chunks", help="Maximum chunks to use per query"),
):
    """
    Generate answers for queries using retrieved chunks and Ollama LLM.
    
    Takes retrieval results JSON and generates comprehensive answers by synthesizing
    information from the retrieved text chunks.
    
    Examples:
        # Generate answers and display in console
        python cli.py generate results.json -m llama3.1:8b
        
        # Generate and save to file
        python cli.py generate results.json -o answers.json
        
        # Use different model with higher temperature
        python cli.py generate results.json -m llama3.2:3b -t 0.5
    """
    typer.echo(f"🤖 Generating answers using model: {model}")
    typer.echo(f"📖 Reading retrieval results from: {results_file}")
    typer.echo(f"🌡️  Temperature: {temperature}")
    typer.echo(f"📊 Max chunks per query: {max_chunks}\n")
    
    start = time.time()
    generator = RAGGenerator(model_name=model, temperature=temperature)
    results = generator.generate(
        data_or_path=results_file,
        output_path=output_file,
        max_chunks=max_chunks
    )
    end = time.time()
    
    # Display summary
    typer.echo(f"\n✅ Generated {len(results)} answers in {(end - start) / 60:.2f} minutes")
    
    if output_file:
        typer.echo(f"💾 Answers saved to: {output_file}")
    else:
        typer.echo("\n📋 Generated Answers:\n")
        for result in results:
            typer.echo(f"Query {result['query_num']}: {result['query']}")
            typer.echo(f"Answer: {result['answer'][:200]}...")
            typer.echo(f"Sources: {', '.join(result['sources'][:3])}")
            typer.echo("-" * 80)


@app.command("query")
def ask_query(    
    query: str = typer.Argument(..., help="Query string to search for"),
    collection_name: str = typer.Argument(..., help="Qdrant collection name"),
    model: str = typer.Option("llama3.1:8b", "--model", "-m", help="Ollama model name"),
    temperature: float = typer.Option(0.3, "--temperature", "-t", help="Sampling temperature (0.0-1.0)"),
    max_chunks: int = typer.Option(5, "--max-chunks", help="Maximum chunks to use per query"),
):
    """
    Combined pipeline: Retrieve results for a query and generate a suitable answer.
    This command runs both retrieval and generation in sequence.
    Works with only one query.
    
    Examples:
        python cli.py query "What is machine learning?" my_collection
        python cli.py query "Explain RAG systems" my_collection -m llama3.2:3b -t 0.5
    """
    typer.echo("\n" + "="*60)
    typer.echo("STEP 1: RETRIEVING DOCUMENTS")
    typer.echo("="*60)
    typer.echo(f"📖 Query: {query}")
    typer.echo(f"📊 Collection: {collection_name}\n")
    
    start = time.time()
    retrieval_result = single_retrieval(
        COLLECTION_NAME=collection_name,
        qdrant_client=qdrant_client,
        query=query,
        max_workers=10,
        top_k=5,
        queries_per_batch=20,
    )    
    typer.echo(f"\n✅ Retrieval completed in {(retrieval_time - start):.2f} seconds")
    typer.echo(f"📂 Found {len(retrieval_result.get('response', []))} relevant documents\n")
    
    typer.echo("="*60)
    typer.echo("STEP 2: GENERATING ANSWER")
    typer.echo("="*60)
    typer.echo(f"🤖 Model: {model}")
    typer.echo(f"🌡️  Temperature: {temperature}\n")
    
    # Generate answer using the new function
    answer_result = generate_single_answer(
        retrieval_result=retrieval_result,
        model_name=model,
        temperature=temperature,
        max_chunks=max_chunks
    )
    
    end = time.time()
    
    typer.echo(f"📊 Total time: {(end - start):.2f} seconds\n")
    
    # Display results
    typer.echo("="*60)
    typer.echo("ANSWER")
    typer.echo("="*60)
    typer.echo(f"\n{answer_result['answer']}\n")
    
    typer.echo("="*60)
    typer.echo("SOURCES")
    typer.echo("="*60)
    for i, source in enumerate(answer_result.get('sources', []), 1):
        score = answer_result.get('chunk_scores', [0])[i-1] if i-1 < len(answer_result.get('chunk_scores', [])) else 0
        typer.echo(f"{i}. {source} (score: {score:.4f})")
    
    typer.echo()



@app.command("evaluate")
def evaluate(
    results_file: str = typer.Argument(..., help="Path to retrieval results JSON"),
    model: str = typer.Option("llama3.1:8b", "--model", "-m", help="Ollama model name"),
):
    """
    Evaluate retrieval quality by judging chunk relevance.
    Uses a small model to assess if retrieved chunks are relevant to queries.
    """
    evaluator = RAGEvaluator(model_name=model)
    with Path(results_file).open("r", encoding="utf-8") as f:
        data = json.load(f)
    evaluations = evaluator.evaluate(data)
    typer.echo(json.dumps(evaluations, ensure_ascii=False, indent=2))


def _write_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


@app.command("export")
def export_results(
    results_file: str = typer.Argument(..., help="Path to retrieval results JSON"),
    output_dir: str = typer.Argument(..., help="Directory to write per-query JSONs"),
    zip_name: Optional[str] = typer.Option(
        "Astraq Cyber Defence_PS4.zip",
        "--zip-name",
        help="Name for the zip archive to create in output_dir",
    ),
):
    """
    Export retrieval results to individual JSON files and optionally create a zip archive.
    Each query gets its own JSON file named by query number.
    """
    results_path = Path(results_file)
    out_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"📖 Loading results from: {results_path}")
    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise typer.BadParameter("Results file must contain a list of results")

    typer.echo(f"📝 Exporting {len(data)} queries to {out_dir}")
    
    exported_count = 0
    with typer.progressbar(data, label="Exporting query files") as progress:
        for item in progress:
            query_num = item.get("query_num")
            if query_num is None:
                continue
            item_copy = dict(item)
            new_item = {}
            new_item['query'] = item_copy['query']
            new_item['response'] = item_copy['response']
            output_file = out_dir / f"{query_num}.json"
            _write_json(new_item, output_file)
            exported_count += 1
    
    typer.echo(f"✅ Exported {exported_count} query files")

    if zip_name:
        zip_path = out_dir / zip_name
        if not zip_path.name.lower().endswith(".zip"):
            zip_path = zip_path.with_suffix(".zip")
        
        typer.echo(f"📦 Creating zip archive: {zip_path.name}")
        
        json_files = [f for f in out_dir.iterdir() if f.suffix.lower() == ".json"]
        with typer.progressbar(json_files, label="Compressing files") as progress:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_ref:
                for file in progress:
                    zip_ref.write(file, arcname=file.name)
        
        typer.echo(f"✅ Created zip archive: {zip_path}")
        typer.echo(f"📊 Archive contains {len(json_files)} files")


def _preprocess_batch(batch: List[Path], dst: Path) -> tuple[int, List[str]]:
    """
    Preprocess a batch of files. Returns (success_count, failed_files).
    """
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


@app.command("preprocess")
def preprocess_dir(
    input_dir: str = typer.Argument(..., help="Directory with input files"),
    output_dir: str = typer.Argument(..., help="Directory to write processed files"),
    glob: str = typer.Option("*", help="Glob to select files within input_dir"),
    max_workers: int = typer.Option(10, help="Number of worker threads"),
    files_per_batch: int = typer.Option(20, help="Files per worker batch"),
):
    """
    Preprocess text files by cleaning and normalizing content (multithreaded with batching).
    Applies text cleaning, whitespace normalization, and other preprocessing steps.
    """
    src = Path(input_dir)
    dst = Path(output_dir)
    start = time.time()
    if not src.exists() or not src.is_dir():
        raise typer.BadParameter(f"Input directory not found: {input_dir}")
    
    # Create output directory if it doesn't exist
    dst.mkdir(parents=True, exist_ok=True)

    files: List[Path] = sorted(src.glob(glob))
    if not files:
        typer.echo("⚠️  No files matched the pattern.")
        raise typer.Exit(code=0)

    # Create batches
    batches = [files[i:i + files_per_batch] for i in range(0, len(files), files_per_batch)]

    typer.echo(f"📁 Input directory: {src}")
    typer.echo(f"📂 Output directory: {dst}")
    typer.echo(f"🔍 Pattern: {glob}")
    typer.echo(f"📊 Found {len(files)} files to process")
    typer.echo(f"📦 Batches: {len(batches)} ({files_per_batch} files per batch)")
    typer.echo(f"⚙️  Workers: {max_workers}\n")
    
    processed_count = 0
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit batch tasks
        future_to_batch = {
            executor.submit(_preprocess_batch, batch, dst): batch 
            for batch in batches
        }
        
        # Process results as they complete
        with typer.progressbar(
            length=len(files), 
            label="Preprocessing files"
        ) as progress:
            for future in as_completed(future_to_batch):
                success_count, batch_failed = future.result()
                processed_count += success_count
                failed_files.extend(batch_failed)
                # Update progress by the batch size
                progress.update(len(future_to_batch[future]))
    
    typer.echo(f"\n✅ Successfully preprocessed {processed_count}/{len(files)} files")
    
    if failed_files:
        typer.echo(f"\n⚠️  Failed to process {len(failed_files)} files:")
        for failed in failed_files[:5]:  # Show first 5 failures
            typer.echo(f"  • {failed}")
        if len(failed_files) > 5:
            typer.echo(f"  ... and {len(failed_files) - 5} more")
    end = time.time()
    typer.echo(f"\n✅ Preprocessed {len(files)} answers in {(end - start) / 60:.2f} minutes")
    typer.echo(f"📂 Output saved to: {dst}")


def main():
    try:
        app()
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
