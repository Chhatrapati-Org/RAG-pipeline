import torch
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.utils.math import cosine_similarity
from tqdm import tqdm


def text_chunker(sentences):
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

    print("Embeddings for semantic checking of neighbouring sentences")

    for i, sentence in enumerate(sentences):
        sentence["combined_embed"] = all_embeddings[i]

    distances = []
    for i in tqdm(range(len(sentences) - 1), desc="Calculating distances"):

        current = all_embeddings[i]
        next = all_embeddings[i + 1]

        similarity = cosine_similarity([current], [next])[0][0]

        distances.append(1 - similarity)
        sentences[i]["distance_to_next"] = distances[i]

    breakpoint_distance_threshold = 0.333

    indices_above_thresh = [
        i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
    ]

    chunks = []
    pre = 0
    for i in tqdm(range(len(indices_above_thresh)), desc="Creating text chunks"):
        post = indices_above_thresh[i]
        combined = ""
        for j in range(pre, post):
            combined += " " + sentences[j]["sentence"]

        MAX_LENGTH = 500 * 5
        while len(combined) > MAX_LENGTH:
            chunk = combined[:MAX_LENGTH]
            chunks.append(chunk)
            combined = combined[MAX_LENGTH:]
        chunks.append(combined)

        pre = post

    if pre < len(sentences):
        combined = ""
        for i in range(pre, len(sentences)):
            combined += " " + sentences[i]["sentence"]
        chunks.append(combined)

    return chunks
