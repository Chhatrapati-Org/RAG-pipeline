"""
Quick launcher for PS04 RAG GUI
Checks prerequisites and launches the Gradio interface
"""

import subprocess
import sys
import time
from pathlib import Path


def check_python_packages():
    """Check if required Python packages are installed."""
    print("🔍 Checking Python packages...")
    
    required = ['gradio', 'qdrant_client', 'typer']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install them with: pip install gradio qdrant-client typer")
        return False
    
    return True


def check_qdrant():
    """Check if Qdrant is running."""
    print("\n🔍 Checking Qdrant connection...")
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        collections = client.get_collections()
        print(f"  ✓ Qdrant is running ({len(collections.collections)} collections)")
        return True
    except Exception as e:
        print(f"  ✗ Qdrant connection failed: {e}")
        print("\nTo start Qdrant:")
        print("  docker run -p 6333:6333 qdrant/qdrant")
        return False


def check_ollama():
    """Check if Ollama is running."""
    print("\n🔍 Checking Ollama...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"  ✓ Ollama is running ({len(models)} models available)")
            if models:
                print(f"    Available models: {', '.join([m['name'] for m in models[:5]])}")
            return True
        else:
            print("  ✗ Ollama responded but with error")
            return False
    except Exception as e:
        print(f"  ✗ Ollama connection failed: {e}")
        print("\nTo start Ollama:")
        print("  ollama serve")
        print("\nTo install models:")
        print("  ollama pull llama3.1:8b")
        return False


def launch_gui():
    """Launch the Gradio GUI."""
    print("\n" + "="*60)
    print("🚀 LAUNCHING PS04 RAG GUI")
    print("="*60)
    
    gui_path = Path(__file__).parent / "gui.py"
    
    if not gui_path.exists():
        print(f"❌ GUI file not found: {gui_path}")
        return False
    
    print(f"\n📂 Starting from: {gui_path}")
    print("🌐 Opening in browser at: http://localhost:7860")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    time.sleep(1)
    
    try:
        subprocess.run([sys.executable, str(gui_path)], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 GUI stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching GUI: {e}")
        return False
    
    return True


def main():
    """Main launcher function."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                  PS04 RAG SYSTEM - GUI LAUNCHER           ║
║            Hybrid Search + ColBERT + BGE Reranker         ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    # Check prerequisites
    packages_ok = check_python_packages()
    qdrant_ok = check_qdrant()
    ollama_ok = check_ollama()
    
    print("\n" + "="*60)
    print("PREREQUISITES SUMMARY")
    print("="*60)
    print(f"Python Packages: {'✓ OK' if packages_ok else '✗ MISSING'}")
    print(f"Qdrant:          {'✓ RUNNING' if qdrant_ok else '✗ NOT RUNNING'}")
    print(f"Ollama:          {'✓ RUNNING' if ollama_ok else '✗ NOT RUNNING'}")
    
    # Decide whether to launch
    if not packages_ok:
        print("\n❌ Cannot launch: Missing Python packages")
        print("Install with: pip install -r requirements.txt")
        return
    
    if not qdrant_ok:
        print("\n⚠️  Warning: Qdrant is not running")
        print("Some features will not work until Qdrant is started.")
        response = input("\nLaunch GUI anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            return
    
    if not ollama_ok:
        print("\n⚠️  Warning: Ollama is not running")
        print("Answer generation will not work until Ollama is started.")
        response = input("\nLaunch GUI anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            return
    
    # Launch
    launch_gui()


if __name__ == "__main__":
    main()
