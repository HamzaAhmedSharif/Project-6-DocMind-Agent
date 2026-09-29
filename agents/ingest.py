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


def load_pdf_text(filepath: str) -> str:
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


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


def build_index(data_dir: str = DATA_DIR):
    model = SentenceTransformer(EMBEDDING_MODEL)
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

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, f"{INDEX_PATH}.bin")
    with open(f"{INDEX_PATH}_meta.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "sources": all_sources}, f)

    print(f"Index built: {len(all_chunks)} chunks from {len(set(all_sources))} documents.")
    return index, all_chunks, all_sources


if __name__ == "__main__":
    build_index()