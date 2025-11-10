"""Research Assistant Agent Graph Implementation.

This module implements the complete Research Assistant AI agent with dual memory
system (short-term session memory + long-term vector database) and PDF processing.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from state import ResearchState, ResearchStateInput, ResearchStateOutput
from configuration import ResearchAgentConfig
from utils.pdf_utils import process_pdf
from utils.embedding_utils import EmbeddingGenerator, add_embeddings_to_chunks
from utils.vector_db_utils import create_vector_db
from utils.session_utils import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# NODE 1: session_init
# ============================================================================

def session_init(state: ResearchState, config: RunnableConfig) -> Dict:
    """Initialize session ID and clear short-term memory.
    
    This node:
    - Generates a unique session ID
    - Initializes empty short-term memory structures
    - Sets initial timestamp
    
    Args:
        state: Current graph state
        config: Runnable configuration
        
    Returns:
        Updated state dict
    """
    
    user_query = state.get("user_query")
    uploaded_pdfs = state.get("uploaded_pdfs")
    
    if user_query is None:
        raise ValueError("user_query is a required input field.")
    if uploaded_pdfs is None:
        raise ValueError("uploaded_pdfs is a required input field.")
    
    logger.info("NODE: session_init - Initializing session")
    
    # Generate unique session ID
    session_id = f"session_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return {
        "session_id": session_id,
        "session_messages": [],
        "session_docs": [],
        "timestamp": datetime.now().isoformat(),
        "pdf_chunks_count": 0,
        "retrieval_scores": {},
        "should_persist": False
    }


# ============================================================================
# NODE 2: check_input (Conditional routing node)
# ============================================================================

def check_input(state: ResearchState, config: RunnableConfig) -> Dict:
    """Determine if input is a PDF upload or a query.
    
    This node analyzes the input and sets the routing flag.
    
    Args:
        state: Current graph state
        config: Runnable configuration
        
    Returns:
        Updated state dict with input_type
    """
    logger.info("NODE: check_input - Analyzing input type")
    
    uploaded_pdfs = state.get("uploaded_pdfs", [])
    user_query = state.get("user_query", "")
    
    # Determine input type
    if uploaded_pdfs and len(uploaded_pdfs) > 0:
        input_type = "pdf_upload"
        logger.info(f"Input type: PDF upload ({len(uploaded_pdfs)} files)")
    elif user_query:
        input_type = "query"
        logger.info(f"Input type: Query - '{user_query[:50]}...'")
    else:
        input_type = "unknown"
        logger.warning("Input type: Unknown (no PDFs or query provided)")
    
    return {"input_type": input_type}


def route_after_check_input(state: ResearchState) -> Literal["ingest_pdf", "process_query"]:
    """Route to either PDF ingestion or query processing.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node name
    """
    input_type = state.get("input_type", "unknown")
    
    if input_type == "pdf_upload":
        logger.info("ROUTE: check_input -> ingest_pdf")
        return "ingest_pdf"
    else:
        logger.info("ROUTE: check_input -> process_query")
        return "process_query"


# ============================================================================
# NODE 3: ingest_pdf
# ============================================================================

def ingest_pdf(state: ResearchState, config: RunnableConfig) -> Dict:
    """Process PDF files and store in both session and vector DB.
    
    This node:
    - Extracts text from PDFs
    - Chunks text into manageable pieces
    - Generates embeddings for chunks
    - Stores chunks in session memory (short-term)
    - Stores chunks in vector database (long-term)
    
    Args:
        state: Current graph state
        config: Runnable configuration
        
    Returns:
        Updated state dict
    """
    logger.info("NODE: ingest_pdf - Processing PDF files")
    
    # Get configuration
    cfg = ResearchAgentConfig.from_runnable_config(config)
    
    # Initialize utilities
    embedding_gen = EmbeddingGenerator(cfg.embedding_model)
    vector_db = create_vector_db(cfg.vector_db_type, cfg.model_dump())
    
    # Get PDFs to process
    uploaded_pdfs = state.get("uploaded_pdfs", [])
    session_id = state.get("session_id")
    
    all_chunks = []
    
    try:
        # Process each PDF
        if uploaded_pdfs:
            for pdf_path in uploaded_pdfs:
                logger.info(f"Processing PDF: {pdf_path}")
                
                # Extract and chunk
                full_text, chunks = process_pdf(
                    pdf_path,
                    chunk_size=cfg.pdf_chunk_size,
                    chunk_overlap=cfg.pdf_chunk_overlap
                )
                
                # Add session metadata
                for chunk in chunks:
                    chunk["session_id"] = session_id
                    chunk["timestamp"] = datetime.now().isoformat()
                
                all_chunks.extend(chunks)
        
        # Generate embeddings for all chunks
        chunks_with_embeddings = add_embeddings_to_chunks(all_chunks, embedding_gen)
        
        # Store in session memory (short-term)
        logger.info(f"Storing {len(chunks_with_embeddings)} chunks in session memory")
        
        # Store in vector database (long-term)
        logger.info(f"Storing {len(chunks_with_embeddings)} chunks in vector DB")
        vector_db.add_documents(
            chunks_with_embeddings,
            collection_name=f"session_{session_id}"
        )
        
        return {
            "session_docs": chunks_with_embeddings,
            "pdf_chunks_count": len(chunks_with_embeddings)
        }
        
    except Exception as e:
        logger.error(f"Error in ingest_pdf: {str(e)}")
        return {
            "answer": f"Error processing PDFs: {str(e)}",
            "sources": []
        }


# ============================================================================
# NODE 4: confirm_ingest
# ============================================================================

def confirm_ingest(state: ResearchState, config: RunnableConfig) -> Dict:
    """Confirm PDF ingestion to user.
    
    Args:
        state: Current graph state
        config: Runnable configuration
        
    Returns:
        Updated state dict
    """
    logger.info("NODE: confirm_ingest - Confirming PDF ingestion")
    
    #pdf_count = len(state.get("uploaded_pdfs", []))
    chunks_count = state.get("pdf_chunks_count", 0)
    
    confirmation_message = (
        #f"✅ Successfully processed {pdf_count} PDF file(s).\n"
        f"📄 Created {chunks_count} text chunks.\n"
        f"💾 Stored in both session memory and long-term database.\n"
        f"🔍 You can now ask questions about the content!"
    )
    
    return {
        "answer": confirmation_message,
        "sources": []
    }


# ============================================================================
# NODE 5: process_query
# ============================================================================

def process_query(state: ResearchState, config: RunnableConfig) -> Dict:
    """Parse and process user query with optional LLM-based intent classification.
    
    This node:
    - Cleans and normalizes the query
    - Optionally uses LLM to identify query intent and extract key topics
    - Prepares query for retrieval
    
    Args:
        state: Current graph state
        config: Runnable configuration
        
    Returns:
        Updated state dict
    """
    logger.info("NODE: process_query - Processing user query")
    
    cfg = ResearchAgentConfig.from_runnable_config(config)
    user_query = state.get("user_query", "")
    
    # Clean query
    parsed_query = user_query.strip()
    
    # Optional: Use LLM to enhance query understanding
    # This can be enabled/disabled based on query complexity
    use_intent_analysis = len(parsed_query.split()) > 3  # Only for complex queries
    
    if use_intent_analysis:
        try:
            llm = ChatOllama(
                model=cfg.local_llm,
                base_url=cfg.ollama_base_url,
                temperature=0.0
            )
            
            intent_prompt = f"""Analyze this research query and extract key information for better document retrieval.

**Query:** {parsed_query}

**Task:** Identify:
1. Query intent: Is this asking for facts, analysis, comparison, or clarification?
2. Key topics: What are the 2-3 main concepts or keywords?
3. Temporal aspect: Does it reference "previous", "recent", or specific time periods?

Provide a brief analysis (2-3 sentences) focusing on what information would best answer this query.

**Analysis:**"""

            messages = [HumanMessage(content=intent_prompt)]
            response = llm.invoke(messages)
            
            query_analysis = str(response.content).strip()
            logger.info(f"Query analysis: {query_analysis[:150]}...")
            
            return {
                "parsed_query": parsed_query,
                "query_metadata": {
                    "analysis": query_analysis,
                    "original_query": user_query
                }
            }
            
        except Exception as e:
            logger.warning(f"Query analysis failed, using simple parsing: {str(e)}")
    
    logger.info(f"Parsed query: {parsed_query}")
    return {"parsed_query": parsed_query}


# ============================================================================
# NODE 6: retrieve_memory
# ============================================================================

def retrieve_memory(state: ResearchState, config: RunnableConfig) -> Dict:
    """Retrieve relevant information from both memory systems.
    
    This node:
    - Retrieves from short-term session memory
    - Retrieves from long-term vector database
    - Returns separate results from both sources
    
    Args:
        state: Current graph state
        config: Runnable configuration
        
    Returns:
        Updated state dict
    """
    logger.info("NODE: retrieve_memory - Retrieving from dual memory")
    
    # Get configuration
    cfg = ResearchAgentConfig.from_runnable_config(config)
    
    # Initialize utilities
    embedding_gen = EmbeddingGenerator(cfg.embedding_model)
    vector_db = create_vector_db(cfg.vector_db_type, cfg.model_dump())
    
    parsed_query = state.get("parsed_query", "")
    session_id = state.get("session_id")
    session_docs = state.get("session_docs", [])
    
    try:
        # Generate query embedding
        query_embedding = embedding_gen.generate_embedding(parsed_query)
        
        # Retrieve from short-term memory (session docs)
        logger.info("Retrieving from short-term session memory")
        short_term_results = []
        
        if session_docs:
            # Compute similarities
            for doc in session_docs:
                doc_embedding = doc.get("embedding", [])
                if doc_embedding:
                    similarity = embedding_gen.compute_similarity(
                        query_embedding,
                        doc_embedding
                    )
                    if similarity >= cfg.similarity_threshold:
                        short_term_results.append({
                            "text": doc["text"],
                            "score": similarity,
                            "source": doc.get("source", "session"),
                            "metadata": {k: v for k, v in doc.items() 
                                       if k not in ["text", "embedding"]}
                        })
            
            # Sort by score and take top_k
            short_term_results.sort(key=lambda x: x["score"], reverse=True)
            short_term_results = short_term_results[:cfg.top_k_short_term]
        
        logger.info(f"Found {len(short_term_results)} results from short-term memory")
        
        # Retrieve from long-term memory (vector DB)
        logger.info("Retrieving from long-term vector database")
        long_term_results = []
        
        try:
            long_term_results = vector_db.search(
                query_embedding=query_embedding,
                collection_name="global",  # Search global collection
                top_k=cfg.top_k_long_term
            )
            
            # Filter by similarity threshold
            long_term_results = [
                doc for doc in long_term_results
                if doc.get("score", 0) >= cfg.similarity_threshold
            ]
            
        except Exception as e:
            logger.warning(f"Error retrieving from long-term memory: {str(e)}")
        
        logger.info(f"Found {len(long_term_results)} results from long-term memory")
        
        # Store retrieval scores
        retrieval_scores = {
            "short_term_count": len(short_term_results),
            "long_term_count": len(long_term_results),
            "short_term_avg_score": (
                sum(d["score"] for d in short_term_results) / len(short_term_results)
                if short_term_results else 0.0
            ),
            "long_term_avg_score": (
                sum(d["score"] for d in long_term_results) / len(long_term_results)
                if long_term_results else 0.0
            )
        }
        
        return {
            "retrieved_short_term": short_term_results,
            "retrieved_long_term": long_term_results,
            "retrieval_scores": retrieval_scores
        }
        
    except Exception as e:
        logger.error(f"Error in retrieve_memory: {str(e)}")
        return {
            "retrieved_short_term": [],
            "retrieved_long_term": [],
            "retrieval_scores": {"error": str(e)}
        }


# ============================================================================
# NODE 7: merge_context
# ============================================================================

def merge_context(state: ResearchState, config: RunnableConfig) -> Dict:
    """Merge and rank results from both memory systems.
    
    This node:
    - Combines short-term and long-term results
    - Deduplicates similar content
    - Ranks by relevance score
    - Limits total context by token budget
    
    Args:
        state: Current graph state
        config: Runnable configuration
        
    Returns:
        Updated state dict
    """
    logger.info("NODE: merge_context - Merging dual memory results")
    
    # Get configuration
    cfg = ResearchAgentConfig.from_runnable_config(config)
    
    short_term = state.get("retrieved_short_term", [])
    long_term = state.get("retrieved_long_term", [])
    
    # Combine results
    all_results = []
    
    # Add short-term with source tag
    for doc in short_term:
        doc_copy = doc.copy()
        doc_copy["memory_type"] = "short_term"
        all_results.append(doc_copy)
    
    # Add long-term with source tag
    for doc in long_term:
        doc_copy = doc.copy()
        doc_copy["memory_type"] = "long_term"
        all_results.append(doc_copy)
    
    # Sort by score (highest first)
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Deduplicate based on text similarity
    merged_results = []
    seen_texts = set()
    
    for doc in all_results:
        text = doc["text"]
        # Simple deduplication by exact match (could use fuzzy matching)
        text_key = text[:100]  # Use first 100 chars as key
        
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            merged_results.append(doc)
    
    # Limit by token budget (approximate: 4 chars ≈ 1 token)
    total_chars = 0
    max_chars = cfg.max_context_tokens * 4
    limited_results = []
    
    for doc in merged_results:
        doc_chars = len(doc["text"])
        if total_chars + doc_chars <= max_chars:
            limited_results.append(doc)
            total_chars += doc_chars
        else:
            break
    
    logger.info(
        f"Merged context: {len(limited_results)} docs "
        f"(~{total_chars // 4} tokens) from {len(all_results)} total"
    )
    
    return {"merged_context": limited_results}


# ============================================================================
# NODE 8: generate_answer
# ============================================================================

def generate_answer(state: ResearchState, config: RunnableConfig) -> Dict:
    """Generate answer using LLM with retrieved context.
    
    This node:
    - Builds prompt with query and context
    - Calls LLM to generate answer
    - Extracts sources used in the answer
    
    Args:
        state: Current graph state
        config: Runnable configuration
        
    Returns:
        Updated state dict
    """
    logger.info("NODE: generate_answer - Generating LLM response")
    
    # Get configuration
    cfg = ResearchAgentConfig.from_runnable_config(config)
    
    # Initialize LLM (Ollama)
    llm = ChatOllama(
        model=cfg.local_llm,
        base_url=cfg.ollama_base_url,
        temperature=cfg.temperature
    )
    
    parsed_query = state.get("parsed_query", "")
    merged_context = state.get("merged_context", [])
    
    # Build context string
    context_parts = []
    sources_used = []
    
    for i, doc in enumerate(merged_context):
        context_parts.append(
            f"[{i+1}] (from {doc.get('memory_type', 'unknown')} memory, "
            f"source: {doc.get('source', 'unknown')})\n{doc['text']}\n"
        )
        sources_used.append({
            "id": i+1,
            "source": doc.get("source", "unknown"),
            "memory_type": doc.get("memory_type", "unknown"),
            "score": doc.get("score", 0.0),
            "text_preview": doc["text"][:200]
        })
    
    context_str = "\n".join(context_parts)
    
    # Build enhanced prompt
    system_prompt = """You are an advanced Research Assistant AI with expertise in analyzing academic papers and technical documents.

Your capabilities:
- Deep understanding of scientific and technical content
- Ability to synthesize information from multiple sources
- Clear explanation of complex concepts
- Critical analysis and comparison of different viewpoints

Guidelines for responses:
1. **Context-based answers**: Base your response ONLY on the provided context from document memory
2. **Source citation**: Always cite sources using [1], [2], etc. notation when referencing specific information
3. **Clarity**: If the context doesn't contain enough information to fully answer the question, explicitly state what's missing
4. **Accuracy**: Be precise and avoid speculation beyond the provided context
5. **Memory awareness**: 
   - Short-term memory contains recently uploaded documents from this session
   - Long-term memory contains knowledge from previous sessions
   - Note any differences or complementary information between them
6. **Structure**: For complex questions, organize your answer with clear sections
7. **Brevity**: Be comprehensive but concise - avoid unnecessary repetition

Response format:
- Start with a direct answer
- Provide supporting details with citations
- End with any caveats or limitations based on available context"""
    
    user_prompt = f"""**Context from Memory:**

{context_str if context_str else "[No relevant context found in memory. Unable to answer based on available documents.]"}

**User Question:**
{parsed_query}

**Instructions:**
Please provide a well-structured, evidence-based answer using the context above. Cite all sources using [number] notation."""
    
    try:
        # Call LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        answer = response.content
        
        logger.info(f"Generated answer ({len(answer)} chars)")
        
        return {
            "answer": answer,
            "sources": sources_used
        }
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": []
        }


# ============================================================================
# NODE 9: update_session
# ============================================================================

def update_session(state: ResearchState, config: RunnableConfig) -> Dict:
    """Update short-term session memory with query and answer.
    
    This node:
    - Adds user query to session messages
    - Adds assistant answer to session messages
    - Updates conversation history
    
    Args:
        state: Current graph state
        config: Runnable configuration
        
    Returns:
        Updated state dict
    """
    logger.info("NODE: update_session - Updating session memory")
    
    # Get configuration
    cfg = ResearchAgentConfig.from_runnable_config(config)
    
    session_id = state.get("session_id")
    parsed_query = state.get("parsed_query", "")
    answer = state.get("answer", "")
    session_messages = list(state.get("session_messages", []))
    
    # Add query and answer to messages
    session_messages.append({
        "type": "human",
        "content": parsed_query,
        "timestamp": datetime.now().isoformat()
    })
    
    session_messages.append({
        "type": "ai",
        "content": answer,
        "timestamp": datetime.now().isoformat()
    })
    
    # Save session to cache
    try:
        session_manager = SessionManager(cfg.session_cache_dir)
        session_data = {
            "session_id": session_id,
            "messages": session_messages,
            "docs_count": len(state.get("session_docs", [])),
            "interaction_count": len([m for m in session_messages if m["type"] == "human"])
        }
        session_manager.save_session(session_id, session_data)
    except Exception as e:
        logger.warning(f"Error saving session: {str(e)}")
    
    logger.info(f"Session updated: {len(session_messages)} messages total")
    
    return {"session_messages": session_messages}


# ============================================================================
# NODE 10: should_persist (Conditional routing node)
# ============================================================================

def should_persist(state: ResearchState, config: RunnableConfig) -> Dict:
    """Determine if session should be persisted to long-term memory.
    
    This node decides based on:
    - Number of interactions in session
    - Configured persistence threshold
    
    Args:
        state: Current graph state
        config: Runnable configuration
        
    Returns:
        Updated state dict
    """
    logger.info("NODE: should_persist - Checking persistence criteria")
    
    # Get configuration
    cfg = ResearchAgentConfig.from_runnable_config(config)
    
    session_messages = state.get("session_messages", [])
    interaction_count = len([m for m in session_messages if m.get("type") == "human"])
    
    # Decide if should persist
    persist = interaction_count >= cfg.persist_threshold
    
    logger.info(
        f"Interactions: {interaction_count}, Threshold: {cfg.persist_threshold}, "
        f"Persist: {persist}"
    )
    
    return {"should_persist": persist}


def route_after_should_persist(state: ResearchState) -> Literal["persist_to_longterm", "finalize_response"]:
    """Route to persistence or finalize based on decision.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node name
    """
    should_persist_flag = state.get("should_persist", False)
    
    if should_persist_flag:
        logger.info("ROUTE: should_persist -> persist_to_longterm")
        return "persist_to_longterm"
    else:
        logger.info("ROUTE: should_persist -> finalize_response")
        return "finalize_response"


# ============================================================================
# NODE 11: persist_to_longterm
# ============================================================================

def persist_to_longterm(state: ResearchState, config: RunnableConfig) -> Dict:
    """Persist session summary to long-term vector database.
    
    This node:
    - Summarizes the session conversation
    - Generates embedding for the summary
    - Stores in global long-term collection
    - Adds metadata about the session
    
    Args:
        state: Current graph state
        config: Runnable configuration
        
    Returns:
        Updated state dict
    """
    logger.info("NODE: persist_to_longterm - Persisting to long-term memory")
    
    # Get configuration
    cfg = ResearchAgentConfig.from_runnable_config(config)
    
    # Initialize utilities
    embedding_gen = EmbeddingGenerator(cfg.embedding_model)
    vector_db = create_vector_db(cfg.vector_db_type, cfg.model_dump())
    session_manager = SessionManager(cfg.session_cache_dir)
    
    session_id = state.get("session_id")
    session_messages = state.get("session_messages", [])
    session_docs = state.get("session_docs", [])
    
    try:
        # Use LLM to create intelligent session summary
        llm = ChatOllama(
            model=cfg.local_llm,
            base_url=cfg.ollama_base_url,
            temperature=0.3  # Slightly higher for more natural summaries
        )
        
        # Build conversation transcript
        conversation_transcript = "\n".join([
            f"{msg['type'].upper()}: {msg['content'][:200]}..." 
            if len(msg['content']) > 200 else f"{msg['type'].upper()}: {msg['content']}"
            for msg in session_messages[-10:]  # Last 10 messages
        ])
        
        # Build document context
        doc_context = f"Documents in session: {len(session_docs)}"
        if session_docs:
            doc_titles = [doc.get("metadata", {}).get("source", "Unknown") for doc in session_docs[:5]]
            doc_context += f"\nKey documents: {', '.join(doc_titles)}"
        
        # Create summarization prompt
        summary_prompt = f"""You are a session summarizer for a Research Assistant AI. Create a concise, informative summary of this research session.

**Session ID:** {session_id}
**Interaction Count:** {len([m for m in session_messages if m['type'] == 'human'])}
**Document Count:** {len(session_docs)}

**Conversation Transcript:**
{conversation_transcript}

**Document Context:**
{doc_context}

**Task:** Create a 150-200 word summary that captures:
1. Main research topics discussed
2. Key questions asked by the user
3. Important findings or insights shared
4. Documents referenced or analyzed
5. Overall context and purpose of the session

Write the summary in third person, focusing on what was researched and discovered. This summary will be used for future retrieval to help recall this research session.

**Session Summary:**"""

        messages = [HumanMessage(content=summary_prompt)]
        response = llm.invoke(messages)
        
        summary = str(response.content).strip()
        logger.info(f"Generated intelligent session summary ({len(summary)} chars)")
        
        # Generate embedding for summary
        summary_embedding = embedding_gen.generate_embedding(summary)
        
        # Create document for long-term storage
        summary_doc = {
            "id": f"summary_{session_id}",
            "text": summary,
            "embedding": summary_embedding,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "interaction_count": len([m for m in session_messages if m["type"] == "human"]),
            "doc_count": len(session_docs),
            "type": "session_summary"
        }
        
        # Store in global long-term collection
        vector_db.add_documents(
            [summary_doc],
            collection_name="global"
        )
        
        logger.info(f"Session {session_id} persisted to long-term memory")
        
    except Exception as e:
        logger.error(f"Error persisting to long-term memory: {str(e)}")
    
    return {}


# ============================================================================
# NODE 12: finalize_response
# ============================================================================

def finalize_response(state: ResearchState, config: RunnableConfig) -> Dict:
    """Format final response for output.
    
    This node:
    - Formats the answer
    - Includes sources
    - Adds metadata
    - Prepares final output
    
    Args:
        state: Current graph state
        config: Runnable configuration
        
    Returns:
        Updated state dict (no changes needed, state is already finalized)
    """
    logger.info("NODE: finalize_response - Finalizing response")
    
    answer = state.get("answer", "")
    sources = state.get("sources", [])
    
    logger.info(f"Response finalized: answer length = {len(answer)}, sources = {len(sources)}")
    
    # State already contains answer and sources, no changes needed
    return {}


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_graph():
    """Construct and compile the Research Assistant graph.
    
    Returns:
        Compiled StateGraph
    """
    logger.info("Constructing Research Assistant graph...")
    
    # Create graph
    graph = StateGraph(ResearchState)
    
    # Add all nodes
    graph.add_node("session_init", session_init)
    graph.add_node("check_input", check_input)
    graph.add_node("ingest_pdf", ingest_pdf)
    graph.add_node("confirm_ingest", confirm_ingest)
    graph.add_node("process_query", process_query)
    graph.add_node("retrieve_memory", retrieve_memory)
    graph.add_node("merge_context", merge_context)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("update_session", update_session)
    graph.add_node("should_persist", should_persist)
    graph.add_node("persist_to_longterm", persist_to_longterm)
    graph.add_node("finalize_response", finalize_response)
    
    # Add edges
    # Start -> session_init -> check_input
    graph.add_edge(START, "session_init")
    graph.add_edge("session_init", "check_input")
    
    # check_input -> [ingest_pdf | process_query] (conditional)
    graph.add_conditional_edges(
        "check_input",
        route_after_check_input,
        {
            "ingest_pdf": "ingest_pdf",
            "process_query": "process_query"
        }
    )
    
    # ingest_pdf -> confirm_ingest -> merge_context
    graph.add_edge("ingest_pdf", "confirm_ingest")
    graph.add_edge("confirm_ingest", "merge_context")
    
    # process_query -> retrieve_memory -> merge_context
    graph.add_edge("process_query", "retrieve_memory")
    graph.add_edge("retrieve_memory", "merge_context")
    
    # merge_context -> generate_answer -> update_session -> should_persist
    graph.add_edge("merge_context", "generate_answer")
    graph.add_edge("generate_answer", "update_session")
    graph.add_edge("update_session", "should_persist")
    
    # should_persist -> [persist_to_longterm | finalize_response] (conditional)
    graph.add_conditional_edges(
        "should_persist",
        route_after_should_persist,
        {
            "persist_to_longterm": "persist_to_longterm",
            "finalize_response": "finalize_response"
        }
    )
    
    # persist_to_longterm -> finalize_response -> END
    graph.add_edge("persist_to_longterm", "finalize_response")
    graph.add_edge("finalize_response", END)
    
    # Compile graph
    compiled_graph = graph.compile()
    
    logger.info("Research Assistant graph compiled successfully!")
    return compiled_graph


# ============================================================================
# GRAPH EXPORT
# ============================================================================

# Create and export the compiled graph
graph = create_graph()

__all__ = ["graph", "create_graph"]
