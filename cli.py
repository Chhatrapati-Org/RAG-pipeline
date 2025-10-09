import json
import re
import zipfile
from pathlib import Path
from typing import List, Optional

import typer
from qdrant_client import QdrantClient

from rag.evaluate import RAGEvaluator
from rag.pipeline import run_chunk_based_rag_pipeline, run_merged_rag_pipeline
from rag.preprocess import preprocess_chunk_text
from rag.retrieve import run_multithreaded_retrieval

app = typer.Typer(help="CLI for PS04 RAG: ingest, retrieve, preprocess, and export.")

qdrant_client = QdrantClient(url="http://localhost:6333")


@app.command("embed")
def ingest_merged(
    directory_path: str = typer.Argument(..., help="Directory containing input files"),
    max_workers: int = typer.Option(4, help="Number of worker threads"),
    chunk_size_kb: int = typer.Option(4, help="Max chunk size in KB"),
    files_per_batch: int = typer.Option(5, help="Files per worker batch"),
):
    stats = run_merged_rag_pipeline(
        qdrant_client=qdrant_client,
        directory_path=directory_path,
        max_workers=max_workers,
        chunk_size_kb=chunk_size_kb,
        files_per_batch=files_per_batch,
    )
    typer.echo(json.dumps(stats, indent=2))


@app.command("embed-chunks")
def ingest_chunks(
    directory_path: str = typer.Argument(..., help="Directory containing input files"),
    max_workers: int = typer.Option(4, help="Number of worker threads"),
    chunk_size_kb: int = typer.Option(4, help="Max chunk size in KB"),
    chunks_per_batch: int = typer.Option(50, help="Chunks per worker batch"),
):
    stats = run_chunk_based_rag_pipeline(
        qdrant_client=qdrant_client,
        directory_path=directory_path,
        max_workers=max_workers,
        chunk_size_kb=chunk_size_kb,
        chunks_per_batch=chunks_per_batch,
    )
    typer.echo(json.dumps(stats, indent=2))


@app.command("retrieve")
def retrieve(
    queries_file_path: str = typer.Argument(..., help="Path to queries JSON file"),
    output_file_path: Optional[str] = typer.Option(
        None, "--output", "-o", help="Path to save retrieval results JSON"
    ),
    max_workers: int = typer.Option(16, help="Number of worker threads"),
    top_k: int = typer.Option(5, help="Top-k similar chunks per query"),
    queries_per_batch: int = typer.Option(20, help="Queries per worker batch"),
):
    results = run_multithreaded_retrieval(
        qdrant_client=qdrant_client,
        queries_file_path=queries_file_path,
        output_file_path=output_file_path,
        max_workers=max_workers,
        top_k=top_k,
        queries_per_batch=queries_per_batch,
    )
    typer.echo(json.dumps(results, indent=2, ensure_ascii=False))


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
        item_copy.pop("query_num", None)
        output_file = out_dir / f"{query_num}.json"
        _write_json(item_copy, output_file)

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
