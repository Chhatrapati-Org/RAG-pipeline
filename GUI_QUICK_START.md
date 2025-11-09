# 🚀 GUI Quick Start - 2 Minutes

## Installation

```powershell
# 1. Install Gradio
pip install gradio

# Or install all requirements
pip install -r requirements.txt
```

## Launch

```powershell
# Easy way (Windows)
launch_gui.bat

# Or
python launch_gui.py

# Or directly
python gui.py
```

Opens at: **http://localhost:7860**

## First Use

### 1️⃣ Connect to Qdrant (Optional)
- Click "⚙️ Qdrant Configuration" 
- URL: `http://localhost:6333` (default)
- Click "Connect"

### 2️⃣ Try Quick Query
- Go to "🔍 Quick Query" tab
- Enter: `"What is machine learning?"`
- Collection: Enter your collection name
- Click "🔍 Search & Generate Answer"

### 3️⃣ Or Index Documents First
- Go to "📚 Embed Documents" tab
- Directory: Path to your docs
- Click "🚀 Start Embedding"
- Note the collection name
- Use it in Quick Query

## Tabs Overview

| Tab | Purpose | Time |
|-----|---------|------|
| 🔍 Quick Query | Single question → answer | 10-20s |
| 📚 Embed Documents | Index corpus | 2-5 min |
| 🔎 Retrieve Documents | Batch retrieval | 30-60s |
| 🤖 Generate Answers | Batch generation | 3-10 min |
| 🧹 Preprocess Files | Clean text | 1-2 min |
| 📦 Export Results | Package for submit | 10-30s |

## Troubleshooting

**"Failed to connect to Qdrant"**
```powershell
docker run -p 6333:6333 qdrant/qdrant
```

**"Ollama model not found"**
```powershell
ollama serve
ollama pull llama3.1:8b
```

**"Collection not found"**
- Run embedding first (📚 tab)
- Check collection name spelling

## Tips

✅ **Start with Quick Query** - fastest way to test  
✅ **Enable Reranker** - better results  
✅ **Use Temperature 0.3** - balanced answers  
✅ **Save collection names** - for reuse  

## Next Steps

📖 Read **GUI_README.md** for detailed docs  
🎬 Check **DEMO_GUIDE.md** for examples  
💻 Try **cli.py** for automation  

**Enjoy! 🎉**
