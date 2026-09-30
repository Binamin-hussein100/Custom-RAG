# Custom RAG

A from-scratch Retrieval-Augmented Generation pipeline. It indexes PDF policy documents, retrieves the most relevant passages for a question, and asks Gemini to answer using only those excerpts.

The sample corpus is three **B-Cubed** policy PDFs:

- `b_cubed_engineering_policy.pdf`
- `b_cubed_data_security_policy.pdf`
- `b_cubed_operations_leasing_policy.pdf`

## How it works

RAG here is two stages: **indexing** (once, when documents change) and **query** (every question).

```
PDFs
  → extract text per page
  → split into overlapping chunks
  → embed with SentenceTransformers
  → store vectors in FAISS

Question
  → embed the question
  → search FAISS for top-k nearest chunks
  → send question + excerpts to Gemini
  → grounded answer with source labels
```

### Indexing

1. **PDF load** (`pdf_loader.py`) — reads every `.pdf` in a folder and returns one dict per page: `{text, source, page}`.
2. **Chunking** (`chunker.py`) — splits each page into overlapping windows so embeddings stay focused on one idea instead of a whole mixed page.
3. **Embedding** (`embedder.py`) — converts chunk text to 384-dimensional vectors with `all-MiniLM-L6-v2`. Similar meaning lands nearby in vector space. Embeddings are L2-normalized.
4. **Vector store** (`vector_store.py`) — FAISS `IndexFlatL2` plus aligned metadata. Can save/load `index.faiss` and `metadata.json`.

### Query

1. Embed the user question with the same model.
2. Retrieve `TOP_K` nearest chunks (default 5), including source filename and page.
3. Format a prompt that lists excerpts as `[Source N: file | page P]` (`generator.py`).
4. Generate with Gemini (`gemini-2.5-flash`) using the grounded system prompt in `config.py`: answer only from the excerpts, cite `[Source N]`, and say "I don't know" when the excerpts don't cover the question.

If retrieval returns nothing, the generator still asks the model, but tells it that no excerpts were found.

## Project layout

| File | Role |
| --- | --- |
| `config.py` | Chunk size, overlap, models, retrieval `TOP_K`, generation settings, system prompt |
| `pdf_loader.py` | PDF → page dicts |
| `chunker.py` | Overlapping character chunks |
| `embedder.py` | SentenceTransformer embeddings |
| `vector_store.py` | FAISS index + metadata persistence |
| `generator.py` | Prompt assembly + Gemini generation |
| `main.py` | Entry point (stub: `Hello from custom-rag!`) |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` (or export in your shell) with a Gemini key. `config.py` loads it with `python-dotenv`; the generator looks for `GOOGLE_API_KEY` or `GEMINI_API_KEY`. Model names can be overridden the same way:

```bash
GOOGLE_API_KEY=your-key-here
# optional
GENERATION_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Put PDFs in a `pdfs/` directory (that path is what `pdf_loader.py` uses in its `__main__` block).

The first embedding run downloads `all-MiniLM-L6-v2` from Hugging Face.

## Tunable settings

All knobs live in `config.py`. `GENERATION_MODEL` and `EMBEDDING_MODEL` can also be set via environment variables.

| Setting | Default | Why it matters |
| --- | --- | --- |
| `CHUNK_SIZE` | `1000` | Large enough to keep a full idea in one vector |
| `CHUNK_OVERLAP` | `200` | Keeps context that would otherwise sit on a chunk boundary |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local, 384-dim; must match query and index |
| `VECTOR_DATABASE` | `faiss` | In-memory / on-disk similarity search |
| `TOP_K` | `5` | How many chunks the LLM sees per question |
| `GENERATION_MODEL` | `gemini-2.5-flash` | Answer model |
| `GENERATION_TEMPERATURE` | `0.0` | Low temperature → more grounded, less creative |
| `GENERATION_MAX_TOKENS` | `1024` | Cap on answer length |

## Module smoke tests

Each pipeline file can be run on its own:

```bash
uv run python pdf_loader.py      # needs a pdfs/ folder
uv run python chunker.py         # overlap sanity check
uv run python embedder.py        # shape check (3, 384)
uv run python vector_store.py    # add / search / save / load
uv run python generator.py       # prompt format; live call if API key is set
```

## Notes

- Query and index **must** use the same embedding model. Re-index after changing `EMBEDDING_MODEL`.
- FAISS search returns L2 distances (lower is closer) paired with chunk metadata.
- Answers are instructed to stay inside retrieved excerpts, cite `[Source N]` labels, and say "I don't know" when the excerpts don't cover the question.
