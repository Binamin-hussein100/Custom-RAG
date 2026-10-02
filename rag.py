import hashlib
import json
import os

import numpy as np

from chunker import chunk_page_text
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    INDEX_DIR,
    PDF_DIR,
    TOP_K,
)
from embedder import embed
from generator import generate
from pdf_loader import load_pdfs
from vector_store import VectorStore

MANIFEST_FILE = "manifest.json"

_store = None


def _file_hash(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _build_manifest(pdf_dir: str) -> dict:
    """Describe everything the index depends on, so any change triggers a rebuild."""
    files = {
        name: _file_hash(os.path.join(pdf_dir, name))
        for name in sorted(os.listdir(pdf_dir))
        if name.endswith(".pdf")
    }
    return {
        "files": files,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def _read_manifest(index_dir: str) -> dict | None:
    path = os.path.join(index_dir, MANIFEST_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_index(pdf_dir: str = PDF_DIR, index_dir: str = INDEX_DIR, force: bool = False) -> bool:
    """Load PDFs, chunk, embed and save the index. Returns True if it rebuilt."""
    global _store

    if not os.path.isdir(pdf_dir):
        raise SystemExit(f"PDF folder '{pdf_dir}' not found.")

    manifest = _build_manifest(pdf_dir)
    if not manifest["files"]:
        raise SystemExit(f"No .pdf files found in '{pdf_dir}'.")

    if not force and _read_manifest(index_dir) == manifest:
        print("Index up to date.")
        return False

    print(f"Indexing {len(manifest['files'])} PDF(s) from '{pdf_dir}'...")
    pages = load_pdfs(pdf_dir)

    chunks = []
    for page in pages:
        for text in chunk_page_text(page["text"]):
            chunks.append({
                "text": text,
                "source": page["source"],
                "page": page["page"],
                "chunk_id": len(chunks),
            })
    if not chunks:
        raise SystemExit(f"No extractable text found in the PDFs in '{pdf_dir}'.")

    texts = [c["text"] for c in chunks]
    vectors = np.vstack([
        embed(texts[i:i + EMBED_BATCH_SIZE])
        for i in range(0, len(texts), EMBED_BATCH_SIZE)
    ])

    store = VectorStore(dimensions=vectors.shape[1])
    store.add_vector(vectors, chunks)
    store.save(index_dir)

    # Written last so a crash mid-build never leaves a manifest that looks valid.
    with open(os.path.join(index_dir, MANIFEST_FILE), "w") as f:
        json.dump(manifest, f, indent=2)

    _store = store
    print(f"Indexed {len(manifest['files'])} file(s), {len(pages)} page(s), {len(chunks)} chunk(s) into '{index_dir}'.")
    return True


def _get_store(index_dir: str = INDEX_DIR) -> VectorStore:
    global _store
    if _store is None:
        if not os.path.exists(os.path.join(index_dir, "index.faiss")):
            raise SystemExit(f"No index found in '{index_dir}'. Run `python main.py ingest` first.")
        _store = VectorStore.load(index_dir)
    return _store


def answer(question: str, top_k: int = TOP_K) -> dict:
    """Retrieve the top_k chunks for a question and generate a grounded answer.

    sources[N-1] is the excerpt the model sees as [Source N].
    """
    store = _get_store()
    hits = store.search(embed([question]), top_k)
    text = generate(question, hits)
    return {
        "answer": text,
        "sources": [
            {"source": meta["source"], "page": meta["page"], "score": score, "text": meta["text"]}
            for score, meta in hits
        ],
    }
