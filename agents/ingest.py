import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

from config import DATA_DIR, INDEX_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf_pages(filepath: str) -> list[str]:
    reader = PdfReader(filepath)
    return [page.extract_text() or "" for page in reader.pages]


def load_pdf_text(filepath: str) -> str:
    return "".join(t + "\n" for t in load_pdf_pages(filepath) if t)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _embed_and_write(chunks: list[str], sources: list[str]):
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, f"{INDEX_PATH}.bin")
    with open(f"{INDEX_PATH}_meta.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "sources": sources}, f)

    print(f"Index built: {len(chunks)} chunks from {len(set(sources))} documents.")


def build_index(data_dir: str = DATA_DIR):
    all_chunks = []
    all_sources = []

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            print(f"Processing: {filename}")
            text = load_pdf_text(filepath)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            all_sources.extend([filename] * len(chunks))

    if not all_chunks:
        raise ValueError(f"No PDFs found in {data_dir}. Add at least one PDF before indexing.")

    _embed_and_write(all_chunks, all_sources)
    return all_chunks, all_sources


MIN_TEXT_WORDS = 50  # below this, a PDF counts as having no text layer (e.g. scanned images)
MIN_TEXT_PAGE_RATIO = 0.2  # at least this share of pages must contain text
MIN_WORDS_PER_PAGE = 5  # averaged over all pages; scanned books sit far below this


def build_index_for_file(pdf_path: str) -> int:
    """Build the index from a single PDF (the app's upload path).

    Returns the number of chunks indexed. Raises ValueError — before the
    existing index files are touched — when the PDF has almost no
    extractable text, which almost always means it is scanned images.
    """
    pages = load_pdf_pages(pdf_path)
    words = sum(len(t.split()) for t in pages)
    pages_with_text = sum(1 for t in pages if t.split())
    if (
        words < MIN_TEXT_WORDS
        or (pages and pages_with_text / len(pages) < MIN_TEXT_PAGE_RATIO)
        or (pages and words / len(pages) < MIN_WORDS_PER_PAGE)
    ):
        raise ValueError(
            f"Only {words} words of extractable text on {pages_with_text} of {len(pages)} pages "
            f"in {os.path.basename(pdf_path)}. DocMind reads a PDF's text layer; "
            "scanned-image PDFs do not have one. Please upload a text-based PDF."
        )
    chunks = chunk_text("".join(t + "\n" for t in pages if t))
    _embed_and_write(chunks, [os.path.basename(pdf_path)] * len(chunks))
    return len(chunks)


if __name__ == "__main__":
    build_index()
