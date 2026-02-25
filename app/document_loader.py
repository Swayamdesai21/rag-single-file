import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader
)

def load_document(file_path: str) -> List[Document]:
    """
    Loads documents based on file extension.
    Supports .pdf, .docx, .pptx, .txt, .md
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        elif ext in [".pptx", ".ppt"]:
            loader = UnstructuredPowerPointLoader(file_path)
        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        return loader.load()
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        # Fallback to a simple text load if it's text-like but unknown
        try:
            loader = TextLoader(file_path)
            return loader.load()
        except:
            return []
