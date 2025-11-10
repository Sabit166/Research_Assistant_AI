"""Installation verification script for Research Assistant AI.

Run this script to check if all dependencies and requirements are properly installed.
"""
import sys
from pathlib import Path

print("=" * 60)
print("🔍 Research Assistant AI - Installation Check")
print("=" * 60)
print()

# Check Python version
print("1️⃣ Checking Python version...")
python_version = sys.version_info
if python_version.major >= 3 and python_version.minor >= 11:
    print(f"   ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    print(f"   ❌ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"   ⚠️  Python 3.11+ required")
print()

# Check required modules
print("2️⃣ Checking required Python packages...")
required_packages = {
    "langchain": "LangChain",
    "langchain_ollama": "LangChain Ollama",
    "langgraph": "LangGraph",
    "streamlit": "Streamlit",
    "chromadb": "ChromaDB",
    "sentence_transformers": "Sentence Transformers",
    "fitz": "PyMuPDF",
    "pdfplumber": "pdfplumber"
}

missing_packages = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} (not installed)")
        missing_packages.append(name)

print()

# Check project structure
print("3️⃣ Checking project structure...")
required_dirs = {
    "src": "Source code directory",
    "src/utils": "Utilities directory",
    "uploads": "PDF uploads directory",
    "data": "Data directory",
    "data/sessions": "Session cache directory",
    "data/chromadb": "Vector database directory"
}

missing_dirs = []
for dir_path, description in required_dirs.items():
    path = Path(dir_path)
    if path.exists():
        print(f"   ✅ {description} ({dir_path})")
    else:
        print(f"   ❌ {description} ({dir_path}) - missing")
        missing_dirs.append(dir_path)

print()

# Check required files
print("4️⃣ Checking required files...")
required_files = {
    "app.py": "Main application",
    "src/graph.py": "Graph definition",
    "src/state.py": "State definitions",
    "src/configuration.py": "Configuration",
    "src/utils/pdf_utils.py": "PDF utilities",
    "src/utils/embedding_utils.py": "Embedding utilities",
    "src/utils/vector_db_utils.py": "Vector DB utilities",
    "src/utils/session_utils.py": "Session utilities"
}

missing_files = []
for file_path, description in required_files.items():
    path = Path(file_path)
    if path.exists():
        print(f"   ✅ {description} ({file_path})")
    else:
        print(f"   ❌ {description} ({file_path}) - missing")
        missing_files.append(file_path)

print()

# Check Ollama
print("5️⃣ Checking Ollama...")
try:
    import subprocess
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        print("   ✅ Ollama is installed")
        if "qwen2.5:7b" in result.stdout or "qwen2.5" in result.stdout:
            print("   ✅ qwen2.5 model available")
        else:
            print("   ⚠️  qwen2.5:7b model not found")
            print("   💡 Run: ollama pull qwen2.5:7b")
    else:
        print("   ❌ Ollama not responding")
        print("   💡 Install from: https://ollama.ai")
except FileNotFoundError:
    print("   ❌ Ollama not installed")
    print("   💡 Install from: https://ollama.ai")
except subprocess.TimeoutExpired:
    print("   ⚠️  Ollama not responding (timeout)")
    print("   💡 Run: ollama serve")
except Exception as e:
    print(f"   ⚠️  Error checking Ollama: {e}")

print()

# Summary
print("=" * 60)
print("📊 SUMMARY")
print("=" * 60)

all_good = True

if missing_packages:
    all_good = False
    print(f"❌ Missing packages: {', '.join(missing_packages)}")
    print("   💡 Run: pip install -r requirements.txt")
    print()

if missing_dirs:
    all_good = False
    print(f"❌ Missing directories: {', '.join(missing_dirs)}")
    print("   💡 Directories will be created automatically on first run")
    print()

if missing_files:
    all_good = False
    print(f"❌ Missing files: {', '.join(missing_files)}")
    print("   💡 Ensure all source files are present")
    print()

if all_good:
    print("✅ All checks passed!")
    print()
    print("🚀 You're ready to launch:")
    print("   streamlit run app.py")
else:
    print("⚠️  Some issues found. Please fix them before running.")
    print()
    print("📚 See QUICKSTART.md for detailed setup instructions")

print("=" * 60)
