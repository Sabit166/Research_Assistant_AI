"""Utility functions for PDF processing."""
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text()
            
        doc.close()
        logger.info(f"Extracted {len(text)} characters from {pdf_path}")
        return text
        
    except ImportError:
        logger.warning("PyMuPDF not available, falling back to pdfplumber")
        try:
            import pdfplumber
            
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                        
            logger.info(f"Extracted {len(text)} characters from {pdf_path}")
            return text
            
        except ImportError:
            logger.error("Neither PyMuPDF nor pdfplumber is installed")
            raise ImportError(
                "Please install either PyMuPDF (pip install pymupdf) "
                "or pdfplumber (pip install pdfplumber)"
            )
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        raise


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, any]]:
    """Split text into chunks with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        
        # Create chunk dictionaries with metadata
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dicts.append({
                "text": chunk,
                "chunk_id": i,
                "chunk_size": len(chunk)
            })
            
        logger.info(f"Created {len(chunk_dicts)} chunks from text")
        return chunk_dicts
        
    except ImportError:
        logger.error("langchain not installed")
        raise ImportError("Please install langchain (pip install langchain)")
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        raise


def process_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> tuple[str, List[Dict[str, any]]]:
    """Extract text and chunk a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        Tuple of (full_text, chunks)
    """
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    if not text or len(text.strip()) == 0:
        raise ValueError(f"No text extracted from {pdf_path}")
    
    # Chunk text
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # Add PDF metadata to chunks
    pdf_name = Path(pdf_path).name
    for chunk in chunks:
        chunk["source"] = pdf_name
        chunk["source_path"] = pdf_path
    
    logger.info(f"Successfully processed {pdf_name}: {len(chunks)} chunks")
    return text, chunks
