import numpy as np  # Import NumPy for array operations and type conversions
import faiss  # Import FAISS for efficient similarity search on dense vectors
import os  # Import os for filesystem path and directory operations
import json  # Import json for serializing and deserializing metadata

class VectorStore:  # Define a wrapper around a FAISS index plus chunk metadata
    def __init__(self, dimensions):  # Initialize the store with a fixed embedding dimensionality
        self.index = faiss.IndexFlatL2(dimensions)  # Create a flat L2-distance FAISS index
        self.metadata = []  # Keep metadata aligned by vector position in the index

    def add_vector(self, vectors, metadata_list):  # Insert new embeddings and their metadata
        self.index.add(np.array(vectors).astype('float32'))  # Convert vectors to float32 and add to FAISS
        self.metadata.extend(metadata_list)  # Append metadata entries in the same order as added vectors

    def search(self, query_vector, top_k):  # Find the nearest vectors to a query embedding
        query_vector = np.array(query_vector).astype('float32')  # Normalize the query into a float32 NumPy array
        scores, indices = self.index.search(query_vector, top_k)  # Run FAISS search and get distances plus indices

        results = []  # Collect matched metadata paired with similarity scores
        for score, idx in zip(scores[0], indices[0]):  # Iterate over the top-k results from the first query row
            if idx == -1:  # Skip placeholder slots FAISS uses when fewer than top_k matches exist
                continue  # Move on to the next search result
            results.append((float(score), self.metadata[idx]))  # Store the distance score with the matching metadata

        return results  # Return the list of search hits

    def save(self, file_path):  # Persist the FAISS index and metadata to disk
        os.makedirs(file_path, exist_ok=True)  # Create the output directory if it does not already exist
        faiss.write_index(self.index, os.path.join(file_path, 'index.faiss'))  # Write the FAISS index to a file
        with open(os.path.join(file_path, 'metadata.json'), 'w') as f:  # Open the metadata JSON file for writing
            json.dump(self.metadata, f)  # Serialize the metadata list into JSON

    @classmethod  # Allow construction from saved files without calling __init__ directly
    def load(cls, file_path):  # Restore a VectorStore instance from a saved directory
        index = faiss.read_index(os.path.join(file_path, 'index.faiss'))  # Load the FAISS index from disk
        with open(os.path.join(file_path, 'metadata.json'), 'r') as f:  # Open the metadata JSON file for reading
            metadata = json.load(f)  # Deserialize metadata from JSON into a Python list

        store = cls(dimensions=index.d)  # Create a store using the dimensionality stored in the loaded index
        store.index = index  # Replace the empty index with the loaded FAISS index
        store.metadata = metadata  # Attach the loaded metadata list
        return store  # Return the fully restored VectorStore instance


if __name__ == "__main__":
    import tempfile

    from embedder import embed

    texts = [
        "I love programming in Python.",
        "It is a sunny day at the beach.",
        "Machine learning models need good data.",
    ]
    metadata = [{"text": t, "id": i} for i, t in enumerate(texts)]
    vectors = embed(texts)

    store = VectorStore(dimensions=vectors.shape[1])
    store.add_vector(vectors, metadata)
    print(f"Indexed {len(store.metadata)} vectors ({vectors.shape[1]} dimensions each)")

    query = embed(["Coding in Python is enjoyable."])
    results = store.search(query, top_k=2)
    print(f"\nSearch returned {len(results)} hits:")
    for score, meta in results:
        print(f"  score={score:.4f}  id={meta['id']}  text={meta['text']!r}")

    assert results[0][1]["id"] == 0, "Closest match should be the Python sentence"

    with tempfile.TemporaryDirectory() as tmp:
        store.save(tmp)
        loaded = VectorStore.load(tmp)
        reloaded = loaded.search(query, top_k=2)
        assert reloaded[0][1]["id"] == results[0][1]["id"]
        print(f"\nSave/load OK — reloaded top match: {reloaded[0][1]['text']!r}")

    print("Test passed.")
