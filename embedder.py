from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads it)...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed(texts):
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

if __name__ == "__main__":
    # Test with 3 sentences
    texts = ["I love programming in Python.", "Coding in Python is enjoyable.", "It is a sunny day."]
    embeddings = embed(texts)
    
    # Check shape
    print(f"\nSuccessfully embedded {len(texts)} sentences.")
    print(f"Resulting shape: {embeddings.shape}")
    
    # Assert check
    assert embeddings.shape == (3, 384), f"Expected (3, 384), got {embeddings.shape}"
    print("Test passed: Embeddings are the correct shape (384 dimensions).")