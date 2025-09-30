import os
from qdrant_client import QdrantClient

# Multiple Qdrant connection options - choose one:

# Option 1: Qdrant Cloud (Recommended - no local installation needed)
# Sign up at https://cloud.qdrant.io for free cluster
# Uncomment and update these lines:
# QDRANT_URL = "https://your-cluster-url.qdrant.io"
# QDRANT_API_KEY = "your-api-key-here"
# client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Option 2: Local Qdrant installation 
# Download from: https://github.com/qdrant/qdrant/releases
# Extract and run qdrant.exe, then use:
client = QdrantClient(host="localhost", port=6333)

# Option 3: In-memory storage (for testing/development only)
# Uncomment this line to use memory-only storage:
# client = QdrantClient(":memory:")

# Option 4: Local file storage (persistent, no server needed)
# Uncomment this line to use file-based storage:
# client = QdrantClient(path="./qdrant_storage")

print(f"Qdrant client initialized: {type(client)}")