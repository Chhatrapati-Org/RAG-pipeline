import torch
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.qdrant import client

COLLECTION_NAME = "ps04"

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )


def embed_sentences(sentences):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_kwargs = {"device": device}

    encode_kwargs = {"normalize_embeddings": True}

    model = HuggingFaceBgeEmbeddings(
        model_name="jinaai/jina-embeddings-v2-small-en",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    texts_to_embed = [x["combined_sent"] for x in sentences]
    all_embeddings = model.embed_documents(texts_to_embed)

    return all_embeddings


def store_embedding(embedding, payload):
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload=payload,
            )
            for idx, vector in enumerate(embedding)
        ],
    )
