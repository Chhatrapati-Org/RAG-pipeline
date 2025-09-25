from rag.chunker import text_chunker
from rag.ingestor import stream_text_batches
from rag.sentencer import make_sentences


def process_directory(directory_path):
    sentences = []

    for batch, filename in stream_text_batches(directory_path, batch_limit_kb=100):
        new_sentences = make_sentences(batch, filename)
        sentences.extend(new_sentences)

    chunks = text_chunker(sentences)
    return 0
