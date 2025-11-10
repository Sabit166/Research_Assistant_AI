# 🚀 Quick Start Guide - Research Assistant AI

## Step-by-Step Setup (5 minutes)

### 1️⃣ Install Ollama

**Windows:**
- Download from: https://ollama.ai/download
- Run installer
- Ollama will start automatically

**Verify installation:**
```powershell
ollama --version
```

### 2️⃣ Download LLM Model

Open PowerShell and run:
```powershell
ollama pull qwen2.5:7b
```

Wait for download to complete (~4GB).

### 3️⃣ Install Python Dependencies

In your project directory:
```powershell
cd d:\Research_Assistant_AI
pip install -r requirements.txt
```

This will install:
- LangChain/LangGraph (agent framework)
- ChromaDB (vector database)
- Streamlit (web UI)
- Sentence Transformers (embeddings)
- PDF processing libraries

### 4️⃣ Verify Ollama is Running

```powershell
ollama serve
```

You should see: `Ollama is running`

**Leave this terminal open!**

### 5️⃣ Launch the Application

**Option A: Web UI (Streamlit)** - Best for regular use
```powershell
cd d:\Research_Assistant_AI
streamlit run app.py
```

Your browser will open automatically to: `http://localhost:8501`

**Option B: CLI Interface** - Best for development/debugging
```powershell
cd d:\Research_Assistant_AI
python cli.py
```

Interactive command-line interface with detailed logging.

---

## 📖 First Use Tutorial

### Using Web UI (Streamlit)

#### Upload Your First PDF

1. **Click "Browse files"** in the left sidebar
2. **Select a PDF** (e.g., research paper, book chapter)
3. **Click "🚀 Process PDF"**
4. **Wait** for processing (shows progress bar)
5. **See confirmation** when complete

### Ask Your First Question

1. **Type a question** in the chat box at the bottom:
   - "What is this document about?"
   - "Summarize the main findings"
   - "Explain the methodology"

2. **Press Enter** or click send

3. **View the answer** with sources shown below

### Continue the Conversation

4. **Ask follow-up questions**:
   - "Tell me more about..."
   - "What did the author say about..."
   - "Compare this to..."

5. **Check session info** in the sidebar:
   - Interaction count increases with each Q&A
   - After 3 interactions, session persists to long-term memory

### Using CLI Interface

#### Launch CLI

```powershell
python cli.py
```

You'll see a welcome banner with available commands.

#### CLI Commands

- `/upload <filepath>` - Upload and process a PDF
- `/list` - List all uploaded PDFs
- `/session` - Show session information
- `/debug on|off` - Toggle debug logging
- `/clear` - Clear session and start fresh
- `/help` - Show help
- `/exit` - Exit CLI

#### Upload a PDF via CLI

```
>>> /upload "D:\Documents\research_paper.pdf"
```

Wait for processing confirmation.

#### Ask Questions via CLI

Just type your question (no command prefix):

```
>>> What is this paper about?
```

View answer with sources and relevance scores.

#### Debug Mode

Enable detailed logging:

```
>>> /debug on
```

Check `logs/cli.log` for detailed execution traces.

---

## ✅ Verification Checklist

Before using, verify:

- [ ] Ollama installed and running (`ollama serve` in terminal)
- [ ] Model downloaded (`ollama list` shows `qwen2.5:7b`)
- [ ] Python packages installed (`pip list | findstr langchain`)
- [ ] Streamlit launches (`streamlit run app.py`)
- [ ] Browser opens to application
- [ ] Can see sidebar with upload button
- [ ] Can type in chat input

---

## 🎯 Common First-Time Issues

### Issue: "Connection refused to localhost:11434"

**Cause**: Ollama is not running

**Fix**:
```powershell
ollama serve
```

Keep this terminal open.

---

### Issue: "Model qwen2.5:7b not found"

**Cause**: Model not downloaded

**Fix**:
```powershell
ollama pull qwen2.5:7b
```

Wait for download (~4GB).

---

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Cause**: Dependencies not installed

**Fix**:
```powershell
pip install -r requirements.txt
```

---

### Issue: PDF upload fails

**Cause**: PDF might be scanned image without text

**Fix**: Try a different PDF with selectable text.

---

### Issue: Application is slow

**Causes**: 
- Large PDF (>100 pages)
- First-time model loading
- Limited RAM

**Fixes**:
- Wait longer (first run is slowest)
- Use smaller PDFs initially
- Close other applications
- Use smaller model: `ollama pull qwen2.5:3b`

---

## 🎓 Learning Path

### Beginner (Day 1)
1. ✅ Upload 1 PDF
2. ✅ Ask 5 questions
3. ✅ View sources
4. ✅ Check session persistence

### Intermediate (Week 1)
1. ✅ Upload multiple PDFs
2. ✅ Compare documents
3. ✅ Build long-term knowledge base
4. ✅ Use previous session knowledge

### Advanced (Month 1)
1. ✅ Customize configuration
2. ✅ Adjust chunk sizes
3. ✅ Experiment with different models
4. ✅ Analyze session data

---

## 📊 Expected Performance

### Processing Times

| Task | Small PDF (5 pages) | Medium PDF (50 pages) | Large PDF (500 pages) |
|------|---------------------|------------------------|------------------------|
| Upload & Process | ~10 seconds | ~30 seconds | ~5 minutes |
| First Query | ~5 seconds | ~5 seconds | ~5 seconds |
| Follow-up Query | ~3 seconds | ~3 seconds | ~3 seconds |

*Times on typical consumer laptop (8GB RAM, no GPU)*

### Resource Usage

- **RAM**: 2-4 GB (model + embeddings)
- **Disk**: ~500 MB per 100-page PDF processed
- **CPU**: 20-50% during query processing

---

## 🆘 Getting Help

1. **Check logs**: `app.log` in project directory
2. **View errors**: Red messages in web UI
3. **Test Ollama**: `ollama run qwen2.5:7b "Hello"`
4. **Verify files**: Ensure all `src/` files exist
5. **Restart app**: Stop (Ctrl+C) and run again

---

## 🎉 You're Ready!

**Your Research Assistant AI is now running!**

- 📄 Upload PDFs
- 💬 Ask questions  
- 🧠 Build knowledge
- 🚀 Boost productivity

**Pro tip**: Upload academic papers, documentation, or books you're studying. Ask synthesis questions that require understanding multiple sections!

---

**Need more help?** Check the main README.md for detailed documentation.

**Happy researching!** 🎓✨
