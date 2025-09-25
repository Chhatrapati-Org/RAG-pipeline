# Qdrant Setup Guide - No Docker Required

This guide provides multiple ways to set up Qdrant for the RAG pipeline without using Docker.

## 🎯 Recommended: Qdrant Cloud (Easiest)

### Step 1: Sign Up
1. Go to [https://cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a free account (no credit card required)
3. Create a new cluster (free tier: 1GB storage)

### Step 2: Get Connection Details
1. In your Qdrant Cloud dashboard, copy:
   - **Cluster URL** (e.g., `https://xyz-abc.qdrant.io`)
   - **API Key** (from the "API Keys" section)

### Step 3: Update Configuration
Edit `rag/qdrant.py`:
```python
# Uncomment and update these lines:
QDRANT_URL = "https://your-cluster-url.qdrant.io"  # Your cluster URL
QDRANT_API_KEY = "your-api-key-here"              # Your API key
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Comment out the localhost line:
# client = QdrantClient(host="localhost", port=6333)
```

---

## 🖥️ Option 2: Local Qdrant Binary (Windows)

### Step 1: Download Qdrant
1. Go to [Qdrant Releases](https://github.com/qdrant/qdrant/releases)
2. Download the latest Windows binary (`qdrant-x86_64-pc-windows-msvc.zip`)
3. Extract to a folder (e.g., `C:\qdrant\`)

### Step 2: Run Qdrant
```bash
# Navigate to extracted folder
cd C:\qdrant

# Run Qdrant server
qdrant.exe
```

### Step 3: Verify
- Open browser to [http://localhost:6333](http://localhost:6333)
- You should see the Qdrant dashboard

### Step 4: Configuration
Keep the default configuration in `rag/qdrant.py`:
```python
client = QdrantClient(host="localhost", port=6333)
```

---

## 💾 Option 3: File-Based Storage (No Server)

### Step 1: Update Configuration
Edit `rag/qdrant.py`:
```python
# Comment out the localhost line:
# client = QdrantClient(host="localhost", port=6333)

# Uncomment this line:
client = QdrantClient(path="./qdrant_storage")
```

### Step 2: Run
- No server setup needed
- Data stored in `./qdrant_storage/` folder
- Automatically created on first use

---

## 🧪 Option 4: In-Memory (Testing Only)

### Step 1: Update Configuration
Edit `rag/qdrant.py`:
```python
# Comment out the localhost line:
# client = QdrantClient(host="localhost", port=6333)

# Uncomment this line:
client = QdrantClient(":memory:")
```

### ⚠️ Warning
- Data lost when program ends
- Only for testing/development
- Not suitable for production

---

## 📦 Alternative: Install Qdrant via Python

### Option A: Using pip
```bash
pip install qdrant-client[fastembed]
```

### Option B: Using conda
```bash
conda install -c conda-forge qdrant-client
```

---

## 🧪 Test Your Setup

Create a test script (`test_qdrant.py`):
```python
from rag.qdrant import client
from qdrant_client.models import Distance, VectorParams

# Test connection
try:
    # Try to create a test collection
    client.create_collection(
        collection_name="test_connection",
        vectors_config=VectorParams(size=10, distance=Distance.COSINE),
    )
    print("✅ Qdrant connection successful!")
    
    # Clean up test collection
    client.delete_collection("test_connection")
    
except Exception as e:
    print(f"❌ Qdrant connection failed: {e}")
    print("Please check your configuration in rag/qdrant.py")
```

Run the test:
```bash
python test_qdrant.py
```

---

## 🎯 Which Option Should You Choose?

| Option | Best For | Pros | Cons |
|--------|----------|------|------|
| **Qdrant Cloud** | Production, ease of use | No setup, scalable, free tier | Requires internet |
| **Local Binary** | Development, full control | Fast, private, full features | Manual setup |
| **File-Based** | Simple projects | No server needed | Limited performance |
| **In-Memory** | Testing only | Fastest setup | Data not persistent |

---

## 🏃‍♂️ Quick Start Commands

### For Qdrant Cloud:
```bash
# 1. Update rag/qdrant.py with your credentials
# 2. Run your pipeline
python main.py
```

### For Local Binary:
```bash
# 1. Download and extract Qdrant
# 2. Run in one terminal:
qdrant.exe

# 3. Run pipeline in another terminal:
python main.py
```

### For File-Based:
```bash
# 1. Update rag/qdrant.py to use file storage
# 2. Run pipeline:
python main.py
```

---

## 🔧 Troubleshooting

### "Import qdrant_client could not be resolved"
```bash
pip install qdrant-client
```

### "Connection refused" or "Connection timeout"
- **Qdrant Cloud**: Check URL and API key
- **Local**: Ensure qdrant.exe is running
- **File-based**: Check folder permissions

### "Collection already exists" errors
```python
# Add this to handle existing collections:
if client.collection_exists("ps04"):
    client.delete_collection("ps04")
```

Choose the option that best fits your needs and follow the corresponding setup steps!