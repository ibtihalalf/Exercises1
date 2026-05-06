from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def load_documents():
    files = list(Path(DOCS_DIR).glob("*"))

    if not files:
        raise FileNotFoundError(f"No files found in {DOCS_DIR}")

    loader = DoclingLoader(
        file_path=[str(file) for file in files],
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(),
    )

    return loader.load()


def ingest():
    docs = load_documents()

    # FIX CHROMA METADATA ERROR
    docs = filter_complex_metadata(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        collection_name="rag_docs",
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    vectorstore.add_documents(docs)

    print(f"Ingested {len(docs)} chunks into ChromaDB")


if __name__ == "__main__":
    ingest()