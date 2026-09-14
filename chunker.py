from config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_page_text(text: str) -> list[str]:
    """Split one page's text into overlapping chunks for embedding.

    Embedding models work best on small, focused passages. A full page
    mixes many ideas into one vector and hurts retrieval. CHUNK_SIZE is
    tuned for roughly 150-250 words per chunk so each vector captures
    one coherent idea.
    """
    if not text:
        return []

    if len(text) <= CHUNK_SIZE:
        return [text]

    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
 
    return chunks


if __name__ == "__main__":
    # Long dummy string — sanity-check chunk count and overlap before real PDFs.
    dummy = "word " * 600  # ~3000 chars, ~600 words
    chunks = chunk_page_text(dummy)

    step = CHUNK_SIZE - CHUNK_OVERLAP
    print(f"CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}, step={step}")
    print(f"Input: {len(dummy)} chars, ~{len(dummy.split())} words")
    print(f"Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  [{i}] {len(chunk)} chars, ~{len(chunk.split())} words")

    if len(chunks) >= 2:
        overlap = chunks[0][-CHUNK_OVERLAP:]
        assert overlap == chunks[1][:CHUNK_OVERLAP], "overlap mismatch"
        print(f"Overlap OK: last {CHUNK_OVERLAP} chars of chunk 0 == first {CHUNK_OVERLAP} of chunk 1")
