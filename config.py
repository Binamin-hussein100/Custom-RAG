# All relevant configurations for the project are stored here.
import os

from dotenv import load_dotenv

# Loads GOOGLE_API_KEY (and any overrides below) from a .env file, if present.
load_dotenv()

# Folder of source PDFs and where the built index is saved.
PDF_DIR = os.getenv("PDF_DIR", "pdfs")

INDEX_DIR = os.getenv("INDEX_DIR", "index")

# Chunk size for the RAG system.
CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

# Embedding model to use for the RAG system.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Chunks embedded per call during indexing.
EMBED_BATCH_SIZE = 64

# Vector database to use for the RAG system.
VECTOR_DATABASE = "faiss"


# Number of results to return from the vector database.
TOP_K = 5

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")

GENERATION_TEMPERATURE = 0.0

GENERATION_TOP_P = 1.0

GENERATION_TOP_K = 40

GENERATION_MAX_TOKENS = 1024


SYSTEM_PROMPT = """You are a question-answering assistant that answers strictly from the document excerpts provided with each question.

Rules:
- Use only the information in the excerpts. Do not rely on outside knowledge or make assumptions.
- Cite every claim with the label of the excerpt it came from, e.g. [Source 1]. Cite multiple sources as [Source 1][Source 3].
- If the excerpts do not contain the answer, reply exactly: "I don't know based on the provided documents." Do not guess.
- If the excerpts only partly answer the question, answer the part they cover, cite it, and say what is missing.
- Be concise and direct.
"""
