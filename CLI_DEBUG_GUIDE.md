# 🐛 CLI Debugging Guide

## Overview

The CLI interface (`cli.py`) provides a command-line alternative to the Streamlit web UI, designed specifically for development, testing, and debugging with enhanced logging and direct control.

---

## Why Use CLI?

### ✅ Benefits

1. **Detailed Logging**: See every step of execution
2. **Faster Iteration**: No browser required
3. **Copilot Agent Mode**: Better integration with AI assistants
4. **Terminal Output**: Easy to copy/paste logs
5. **Debug Commands**: Built-in debugging tools
6. **Scriptable**: Can be automated for testing

### 🎯 Best For

- Development and debugging
- Testing new features
- Analyzing graph execution
- Troubleshooting errors
- Performance profiling
- Automated testing scripts

---

## Getting Started

### Launch CLI

```powershell
cd d:\Research_Assistant_AI
python cli.py
```

### First Commands

```
>>> /help              # Show all commands
>>> /session           # View current session info
>>> /debug on          # Enable detailed logging
```

---

## Commands Reference

### File Management

#### `/upload <filepath>`

Upload and process a PDF document.

**Example:**
```
>>> /upload "D:\Documents\research_paper.pdf"
```

**Output:**
- Processing status
- File size
- Success/error message
- Time taken

**Debugging:**
- Check `logs/cli.log` for PDF parsing details
- Look for text extraction issues
- Verify chunk creation

---

#### `/list`

List all uploaded PDFs in current session.

**Example:**
```
>>> /list
```

**Output:**
- Number of PDFs
- Filename, size, timestamp for each

---

### Session Management

#### `/session`

Show detailed session information.

**Example:**
```
>>> /session
```

**Output:**
- Session ID
- Interaction count
- Number of PDFs uploaded
- Model configuration
- Persist threshold

**Use Case:**
- Verify session state
- Check if persistence will trigger
- Confirm configuration

---

#### `/clear`

Clear current session and start fresh.

**Example:**
```
>>> /clear
>>> yes
```

**Use Case:**
- Test fresh session behavior
- Reset after errors
- Start new conversation context

---

### Debugging

#### `/debug on|off`

Toggle debug-level logging.

**Example:**
```
>>> /debug on
```

**What It Does:**
- Changes log level to DEBUG
- Shows every function call
- Logs all state transitions
- Displays LLM prompts and responses
- Records vector DB queries

**When Enabled:**
- Check `logs/cli.log` for detailed traces
- See graph node execution order
- View embedding generation
- Track memory retrieval

**Example Debug Output:**
```
DEBUG - Entering node: process_query
DEBUG - Query: What is the main topic?
DEBUG - Retrieved 5 chunks from vector DB
DEBUG - Chunk scores: [0.89, 0.85, 0.82, 0.79, 0.76]
DEBUG - Generating LLM response...
DEBUG - LLM tokens: 342
DEBUG - Exiting node: process_query
```

---

### Utility

#### `/help`

Show help with all available commands.

#### `/exit` or `/quit`

Exit CLI (saves session automatically).

---

## Query Processing

### Ask Questions

Simply type your question without any command prefix:

```
>>> What is the main finding of this paper?
```

### Output Format

1. **Processing Indicator**
   ```
   🤔 Processing query...
   ```

2. **Answer Section**
   ```
   💡 Answer:
   ----------------------------------------------------------------------
   [The AI's detailed answer appears here]
   ----------------------------------------------------------------------
   ```

3. **Sources Section**
   ```
   📚 Sources (3):
      1. 🟢 Relevance: 0.923
         Source: research_paper.pdf
         Page: 5
         Preview: The study examined the effects of...
      2. 🟡 Relevance: 0.847
         Source: research_paper.pdf
         Page: 12
         Preview: Results showed a significant correlation...
      3. 🔴 Relevance: 0.693
         Source: research_paper.pdf
         Page: 23
         Preview: Discussion of limitations includes...
   ```

4. **Persistence Status**
   ```
   💾 Session will be persisted to long-term memory
   ```

5. **Interaction Count**
   ```
   📊 Interaction: 3
   ```

---

## Debugging Workflows

### Workflow 1: Debug PDF Processing

**Scenario:** PDF upload fails or extracts no text

**Steps:**

1. Enable debug logging:
   ```
   >>> /debug on
   ```

2. Upload PDF:
   ```
   >>> /upload "problematic.pdf"
   ```

3. Check logs:
   ```powershell
   Get-Content logs\cli.log -Tail 100
   ```

4. Look for:
   - `Error reading PDF`
   - `No text extracted`
   - `Chunk count: 0`
   - `pdfplumber` or `PyMuPDF` errors

5. Common fixes:
   - PDF is scanned image → Use OCR
   - PDF is encrypted → Unlock first
   - PDF is corrupted → Try different file

---

### Workflow 2: Debug Query Processing

**Scenario:** Answers are irrelevant or wrong

**Steps:**

1. Enable debug logging:
   ```
   >>> /debug on
   ```

2. Ask question:
   ```
   >>> What is the methodology?
   ```

3. Check logs for:
   - **Vector DB retrieval:**
     ```
     DEBUG - ChromaDB query: "What is the methodology?"
     DEBUG - Retrieved chunks: 5
     DEBUG - Top score: 0.234  ← Too low!
     ```

   - **Embedding quality:**
     ```
     DEBUG - Query embedding dimension: 384
     DEBUG - Similarity scores: [0.89, 0.85, ...]
     ```

   - **LLM prompt:**
     ```
     DEBUG - Prompt sent to LLM:
     Context: [Retrieved chunks]
     Question: What is the methodology?
     ```

4. Common issues:
   - Low relevance scores → Chunk size too small
   - No relevant chunks → PDF not properly indexed
   - LLM hallucinating → Prompt needs improvement

---

### Workflow 3: Debug Memory Persistence

**Scenario:** Session not persisting to long-term memory

**Steps:**

1. Check session info:
   ```
   >>> /session
   ```

2. Note interaction count and persist threshold:
   ```
   Interaction count: 2
   Persist threshold: 3
   ```

3. Ask one more question:
   ```
   >>> Tell me more about the results
   ```

4. Look for persistence message:
   ```
   💾 Session will be persisted to long-term memory
   ```

5. Check ChromaDB:
   ```powershell
   ls data\chromadb\
   ```

6. Should see:
   - `chroma.sqlite3`
   - `.bin` files (embeddings)

---

### Workflow 4: Compare Web UI vs CLI

**Scenario:** Behavior differs between interfaces

**Steps:**

1. Run same query in both:

   **CLI:**
   ```
   >>> /debug on
   >>> What is the conclusion?
   ```

   **Web UI:** Type same question

2. Compare logs:
   - CLI: `logs/cli.log`
   - Web UI: `logs/app.log`

3. Check for differences in:
   - Session ID format
   - Interaction count
   - State initialization
   - Graph execution path

---

## Log Analysis

### Log Locations

- **CLI logs**: `logs/cli.log`
- **Web UI logs**: `logs/app.log`
- **System logs**: Console output

### Read Recent Logs

```powershell
# Last 50 lines
Get-Content logs\cli.log -Tail 50

# Search for errors
Select-String -Path logs\cli.log -Pattern "ERROR|EXCEPTION"

# Filter by timestamp
Get-Content logs\cli.log | Select-String "2024-11-10"
```

### Key Log Patterns

**Successful Query:**
```
INFO - CLI initialized with session: cli_20241110_143022
INFO - Query: What is the main topic?
DEBUG - Entering node: process_query
DEBUG - Retrieved 5 relevant chunks
DEBUG - Chunk scores: [0.89, 0.85, 0.82, 0.79, 0.76]
INFO - Query processed successfully
```

**Failed Query:**
```
ERROR - Error processing query: Connection refused
ERROR - Cannot connect to Ollama at localhost:11434
EXCEPTION - Traceback follows...
```

**Memory Persistence:**
```
INFO - Interaction count: 3
INFO - Threshold reached: 3
DEBUG - Persisting session to long-term memory
DEBUG - ChromaDB collection: research_sessions
INFO - Session persisted successfully
```

---

## Performance Profiling

### Measure Query Time

1. Enable debug:
   ```
   >>> /debug on
   ```

2. Ask question (note timestamps in log):
   ```
   >>> What are the key findings?
   ```

3. Check log timestamps:
   ```
   2024-11-10 14:30:22 - INFO - Query: What are the key findings?
   2024-11-10 14:30:23 - DEBUG - Vector DB query: 0.2s
   2024-11-10 14:30:25 - DEBUG - LLM generation: 2.1s
   2024-11-10 14:30:25 - INFO - Total time: 2.3s
   ```

### Bottleneck Identification

Common bottlenecks:

1. **Vector DB Query** (should be <0.5s)
   - If slow: Check collection size
   - Solution: Limit chunks per PDF

2. **LLM Generation** (typically 2-5s)
   - If slow: Model loading or RAM issue
   - Solution: Preload model, increase RAM

3. **PDF Processing** (varies by size)
   - If slow: Large PDF or OCR needed
   - Solution: Process smaller sections

---

## Integration with GitHub Copilot

### Use CLI with Copilot Agent Mode

The CLI is designed to work seamlessly with GitHub Copilot's agent mode:

1. **Share Context**: Copilot can read CLI output directly
2. **Iterate Quickly**: Make changes, test immediately
3. **Analyze Errors**: Copilot sees full stack traces
4. **Generate Fixes**: Copilot suggests code improvements

### Example Workflow

1. Run CLI with debug enabled:
   ```
   >>> /debug on
   >>> What is the methodology?
   ```

2. Copy error or unexpected output

3. In Copilot chat:
   ```
   @workspace I got this error in CLI:
   [paste error]
   
   What's wrong and how to fix?
   ```

4. Copilot analyzes:
   - Reads source code
   - Checks logs
   - Suggests specific fix

5. Apply fix and test immediately in CLI

---

## Advanced Debugging

### Custom Debug Commands

Add your own debug commands to `cli.py`:

```python
elif cmd == '/dump_state':
    # Dump current state
    print(json.dumps(result, indent=2))

elif cmd == '/test_embedding':
    # Test embedding generation
    from src.utils.embedding_utils import EmbeddingGenerator
    gen = EmbeddingGenerator()
    emb = gen.generate_embedding(arg)
    print(f"Embedding dimension: {len(emb)}")
```

### Breakpoint Debugging

Use Python debugger with CLI:

```python
# In cli.py, add:
import pdb; pdb.set_trace()
```

Run with debugger:
```powershell
python -m pdb cli.py
```

### Trace Graph Execution

Enable LangGraph tracing:

```python
# In cli.py, before graph.invoke():
import langchain
langchain.debug = True

result = self.graph.invoke(initial_state)
```

---

## Testing Strategies

### Unit Testing

Test individual nodes via CLI:

```
>>> /debug on
>>> /upload test_document.pdf
>>> What is on page 1?
```

Check each node executes correctly.

### Integration Testing

Test full workflows:

```python
# test_cli.py
def test_pdf_query_workflow():
    cli = ResearchAssistantCLI()
    cli.upload_pdf("test.pdf")
    result = cli.process_query("What is this about?")
    assert result is not None
```

### Regression Testing

Keep test queries and expected outputs:

```python
test_cases = [
    {"query": "What is the main topic?", "expected_keywords": ["research", "study"]},
    {"query": "Who are the authors?", "expected_keywords": ["Smith", "Jones"]},
]
```

---

## Common Issues & Solutions

### Issue: "Cannot connect to Ollama"

**Debug:**
```
>>> /debug on
>>> Test connection
```

**Check logs for:**
```
ERROR - Connection refused: localhost:11434
```

**Solution:**
```powershell
ollama serve
```

---

### Issue: No sources returned

**Debug:**
```
>>> /debug on
>>> /session  # Check if PDFs uploaded
>>> /list     # Verify PDF list
```

**Check logs for:**
```
DEBUG - ChromaDB query returned 0 results
```

**Solution:**
- Upload PDF first
- Verify PDF has text content

---

### Issue: Wrong answers

**Debug:**
```
>>> /debug on
>>> [ask same question]
```

**Check logs for:**
```
DEBUG - Top relevance scores: [0.234, 0.189, 0.156]  ← All too low!
```

**Solution:**
- Improve query phrasing
- Upload more relevant documents
- Adjust chunk size in config

---

## Best Practices

1. **Always enable debug for development:**
   ```
   >>> /debug on
   ```

2. **Use consistent session for testing:**
   - Don't `/clear` unless needed
   - Test persistence behavior

3. **Monitor logs actively:**
   - Keep `logs/cli.log` open in editor
   - Use `tail -f` equivalent

4. **Test edge cases:**
   - Empty PDFs
   - Very long queries
   - Multiple PDFs with conflicting info

5. **Document test results:**
   - Keep notes of what works
   - Log successful configurations

---

## Next Steps

- Explore `src/graph.py` to understand node execution
- Modify `src/configuration.py` to tune parameters
- Add custom commands to `cli.py`
- Build automated test scripts
- Integrate with CI/CD pipelines

**Happy debugging!** 🐛🔧

