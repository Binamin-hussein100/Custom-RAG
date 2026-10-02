import os

from google import genai
from google.genai import types

from config import (
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    GENERATION_TOP_K,
    GENERATION_TOP_P,
    SYSTEM_PROMPT,
)

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _normalize_chunks(chunks: list) -> list[dict]:
    """Accept metadata dicts or (score, metadata) tuples from vector search."""
    normalized = []
    
    for item in chunks:
        if isinstance(item, tuple) and len(item) == 2:
            normalized.append(item[1])
        else:
            normalized.append(item)
    return normalized


def _source_label(index: int, chunk: dict) -> str:
    source = chunk.get("source", "unknown")
    page = chunk.get("page")
    if page is not None:
        return f"[Source {index}: {source} | page {page}]"
    return f"[Source {index}: {source}]"


def format_prompt(question: str, chunks: list) -> str:
    """Build one user prompt from a question and retrieved chunk metadata."""
    normalized = _normalize_chunks(chunks)

    if not normalized:
        return (
            "No relevant document excerpts were retrieved.\n\n"
            f"QUESTION:\n{question.strip()}"
        )

    excerpts = []
    for i, chunk in enumerate(normalized, start=1):
        text = chunk.get("text", "").strip()
        excerpts.append(f"{_source_label(i, chunk)}\n{text}")

    context = "\n\n".join(excerpts)
    return (
        "Answer the question using only the retrieved excerpts below. "
        "Cite the source labels, e.g. [Source 1], for every claim.\n\n"
        f"RETRIEVED EXCERPTS:\n{context}\n\n"
        f"QUESTION:\n{question.strip()}"
    )


def generate(question: str, chunks: list) -> str:
    """Format retrieved chunks into a prompt and send it to the LLM."""
    prompt = format_prompt(question, chunks)
    client = _get_client()
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            top_k=GENERATION_TOP_K,
            max_output_tokens=GENERATION_MAX_TOKENS,
        ),
    )
    return response.text


if __name__ == "__main__":
    sample_chunks = [
        {
            "text": "All build pipelines must fail if static analysis errors or unit test failures are encountered.",
            "source": "b_cubed_engineering_policy.pdf",
            "page": 2,
        },
        {
            "text": "Root-level SSH access is prohibited on edge cluster nodes.",
            "source": "b_cubed_data_security_policy.pdf",
            "page": 5,
        },
    ]
    question = "What happens if our CI pipeline has failing unit tests?"

    prompt = format_prompt(question, sample_chunks)
    print("Formatted prompt:\n")
    print(prompt)

    assert "[Source 1: b_cubed_engineering_policy.pdf | page 2]" in prompt
    assert "[Source 2: b_cubed_data_security_policy.pdf | page 5]" in prompt
    assert question in prompt
    print("\nformat_prompt test passed.")

    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        answer = generate(question, sample_chunks)
        print("\nModel answer:\n")
        print(answer)
    else:
        print("\nSet GOOGLE_API_KEY or GEMINI_API_KEY to run a live generation test.")
