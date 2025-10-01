import os
from natsort import natsorted
import re

def preprocess_chunk_text(text: str) -> str:
    """Basic text preprocessing before embedding."""
    text = re.sub(r"\"\w+\"\s*:\s*null", "", text) # remove keys with None values
    text = re.sub(r"\"\S{,15}\"\s*:", "", text) # remove keys assuming keys have length <= 15
    text = re.sub(r"/url\?q=\S+|https?://\S+|www\.\S+", "", text) # remove URLs
    text = re.sub(r"\[|\]|\}|\{", " ", text) # remove brackets
    text = re.sub(r"\\n|\\", " ", text) # remove escaped newlines and backslashes
    text = re.sub(r"<[^>]+>", "", text) # remove HTML tags
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    text = re.sub(r",\s*,+", r", ", text) # remove multiple commas
    text = re.sub(r"\"\s*,\s*\"", r". ", text) # make sentence
    text = re.sub(r"\.\s*\.+", r". ", text) # remove multiple dots
    text = text.strip()
    return text

files = os.listdir(r"C:\Users\22bcscs055\Downloads\mock_data")
files = natsorted(files)
for fname in files:
    print(rf"C:\Users\22bcscs055\Downloads\mock_data\{fname}")
    text = ' '
    with open(rf"C:\Users\22bcscs055\Downloads\mock_data\{fname}","r",encoding="utf-8") as g:
        text = preprocess_chunk_text(g.read())
    # os.mkdir(rf"C:\Users\22bcscs055\Downloads\mock_data_processed", exist_ok=True)
    with open(rf"C:\Users\22bcscs055\Downloads\mock_data_processed\{fname}", "w", encoding="utf-8") as f:
        f.write(text)

