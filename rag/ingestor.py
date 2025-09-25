import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, Tuple, List
from threading import Lock

from tqdm import tqdm


class MultiThreadedFileReader:
    def __init__(self, max_workers=4, chunk_size_kb=4):
        self.max_workers = max_workers
        self.chunk_size_bytes = chunk_size_kb * 1024
        self.lock = Lock()
        self.progress_bar = None

    def _read_file_chunks(self, file_path: str) -> List[Tuple[str, str]]:
        """Read a single file and return chunks with metadata."""
        filename = os.path.basename(file_path)
        chunks = []
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                _, ext = os.path.splitext(filename)
                
                chunk_id = 0
                while True:
                    content = f.read(self.chunk_size_bytes)
                    if not content:
                        break
                    
                    # Clean JSON content
                    if ext.lower() == ".json":
                        content = re.sub(
                            r"[\[\]\{\}\"\n]",
                            "",
                            re.sub(r"\s[\s]+", " ", re.sub(r"\\[nrt]", "", content)),
                        )
                    
                    # Ensure chunk doesn't exceed size limit
                    content_bytes = content.encode("utf-8")
                    while len(content_bytes) > self.chunk_size_bytes:
                        chunklet = content[:self.chunk_size_bytes]
                        content = content[self.chunk_size_bytes:]
                        content_bytes = content.encode("utf-8")
                        chunks.append((chunklet, filename, chunk_id))
                        chunk_id += 1
                    
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
        return chunks

    def read_files_multithreaded(self, directory_path: str) -> Generator[Tuple[str, str, int], None, None]:
        """Read all files in directory using multiple threads."""
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist")
        
        # Get all files to process
        file_paths = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                file_paths.append(file_path)
        
        if not file_paths:
            print("No files found to process")
            return
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file reading tasks
            future_to_file = {
                executor.submit(self._read_file_chunks, file_path): file_path 
                for file_path in file_paths
            }
            
            # Process completed tasks as they finish
            with tqdm(total=len(file_paths), desc="Processing files") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        chunks = future.result()
                        for chunk_content, filename, chunk_id in chunks:
                            yield chunk_content, filename, chunk_id
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                    finally:
                        pbar.update(1)

