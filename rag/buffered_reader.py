import os

from tqdm import tqdm


def stream_text_batches(directory_path, batch_limit_kb=100):
    batch = []
    batch_size = 0
    limit_bytes = batch_limit_kb * 1024
    files_to_process = os.listdir(directory_path)
    i = 0
    for filename in tqdm(files_to_process, desc="Processing files"):
        file_path = os.path.join(directory_path, filename)
        i += 1
        if i > 3:
            break
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                chunklet = f.read(limit_bytes)

                while chunklet:
                    content = chunklet
                    content_size = len(content.encode("utf-8"))
                    while batch_size + content_size > limit_bytes:
                        batch.append(content[: limit_bytes - batch_size])
                        yield batch
                        batch = []
                        batch_size = 0
                        content = content[limit_bytes - batch_size :]
                        content_size = len(content.encode("utf-8"))

                    batch.append(content)
                    batch_size += content_size

                    chunklet = f.read(limit_bytes)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if batch:
        yield batch
