import json
import re
import zipfile
from pathlib import Path
from typing import List, Optional
import time
import typer
from qdrant_client import QdrantClient

from rag.evaluate import RAGEvaluator
from rag.pipeline import run_chunk_based_rag_pipeline, run_merged_rag_pipeline
from rag.preprocess import preprocess_chunk_text
from rag.retrieve import run_multithreaded_retrieval

app = typer.Typer(help="CLI for PS04 RAG: ingest, retrieve, preprocess, and export.")

qdrant_client = QdrantClient(url="http://localhost:6333")


@app.command("embed-main")
def ingest_merged(
    directory_path: str = typer.Argument(..., help="Directory containing input files"),
    max_workers: int = typer.Option(20, help="Number of worker threads"),
    chunk_size_kb: int = typer.Option(1, help="Max chunk size in KB"),
    files_per_batch: int = typer.Option(20, help="Files per worker batch"),
):
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
    typer.echo(json.dumps(stats, indent=2))


@app.command("embed-chunks")
def ingest_chunks(
    directory_path: str = typer.Argument(..., help="Directory containing input files"),
    max_workers: int = typer.Option(20, help="Number of worker threads"),
    chunk_size_kb: int = typer.Option(1, help="Max chunk size in KB"),
    chunks_per_batch: int = typer.Option(50, help="Chunks per worker batch"),
):
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
    typer.echo(json.dumps(stats, indent=2))

# TODO: Run DOCKER image of Qdrant if not running
@app.command("retrieve")
def retrieve(
    queries_file_path: str = typer.Argument(..., help="Path to queries JSON file"),
    collection_name: str = typer.Argument(..., help="Qdrant collection name"),
    output_file_path: Optional[str] = typer.Option(
        None, "--output", "-o", help="Path to save retrieval results JSON"
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
    directory_path = input("Enter the Directory containing input files:")
    queries_file_path = input("Enter the Path to queries JSON file:")
    output_file_path = input("Enter the Path to save retrieval results JSON (Do not skip):")
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
    typer.echo(json.dumps(stats, indent=2))
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
    typer.echo(f"Time taken for retrieval: {(fin - init) / 60} minutes")
    # typer.echo(json.dumps(results, indent=2, ensure_ascii=False))


# TODO: Start the ollama server if not running
@app.command("evaluate")
def evaluate(
    results_file: str = typer.Argument(..., help="Path to retrieval results JSON"),
    model: str = typer.Option("llama3.1:8b", "--model", "-m", help="Ollama model name"),
):
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
        None,
        "--zip-name",
        help="Optional name for the zip archive to create in output_dir",
    ),
):
    results_path = Path(results_file)
    out_dir = Path(output_dir)

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise typer.BadParameter("Results file must contain a list of results")

    for item in typer.progressbar(data, label="Exporting files"):
        query_num = item.get("query_num")
        if query_num is None:
            continue
        item_copy = dict(item)
        new_item = {}
        new_item['query'] = item_copy['query']
        new_item['response'] = item_copy['response']
        output_file = out_dir / f"{query_num}.json"
        _write_json(new_item, output_file)

    if zip_name:
        zip_path = out_dir / zip_name
        if not zip_path.name.lower().endswith(".zip"):
            zip_path = zip_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w") as zip_ref:
            for file in out_dir.iterdir():
                if file.suffix.lower() == ".json":
                    zip_ref.write(file, arcname=file.name)
        typer.echo(f"Created zip: {zip_path}")


@app.command("preprocess")
def preprocess_dir(
    input_dir: str = typer.Argument(..., help="Directory with input files"),
    output_dir: str = typer.Argument(..., help="Directory to write processed files"),
    glob: str = typer.Option("*", help="Glob to select files within input_dir"),
):
    src = Path(input_dir)
    dst = Path(output_dir)
    if not src.exists() or not src.is_dir():
        raise typer.BadParameter(f"Input directory not found: {input_dir}")

    files: List[Path] = sorted(src.glob(glob))
    if not files:
        typer.echo("No files matched.")
        raise typer.Exit(code=0)

    for fp in typer.progressbar(files, label="Preprocessing files"):
        if not fp.is_file():
            continue
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        processed = preprocess_chunk_text(raw)
        out_fp = dst / fp.name
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        with out_fp.open("w", encoding="utf-8") as f:
            f.write(processed)


def main():
    try:
        app()
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
