"""Loads backend configuration from environment variables."""

import os
from rich import print as rprint
from dotenv import load_dotenv

load_dotenv()

errors = []


def load_or_error(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        errors.append(f"Environment variable [bold red]{key}[/bold red] is not set.")
    return value


MODEL_NAME = load_or_error("MODEL_NAME")
OPENROUTER_API_KEY = load_or_error("OPENROUTER_API_KEY")

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if errors:
    for error in errors:
        rprint(error)
    raise ValueError("Environment variables are not set")