import re


# TODO: Merge into a single replace to improve performance
def preprocess_chunk_text(text: str) -> str:
    text = re.sub(r"\"\w+\"\s*:\s*(null|None)", "", text)
    text = re.sub(r"\"\S{,15}\"\s*:", "", text)
    text = re.sub(r"/url\?q=\S+|https?://\S+|www\.\S+", "", text)
    text = re.sub(r"(\\n)|(\\)|(\\\")|(\\\')|(\")", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*,+", r", ", text)
    text = re.sub(r"\s*[\{\}\[\]]\s*,\s*[\{\}\[\]]", r". ", text)
    text = re.sub(r"\s*\.\s*\.+", r". ", text)
    text = re.sub(r"(,\.)|(\.,)", r". ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text
