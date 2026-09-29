import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

from config import INDEX_PATH, EMBEDDING_MODEL, TOP_K_RESULTS


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.read_index(f"{INDEX_PATH}.bin")
        with open(f"{INDEX_PATH}_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.chunks = meta["chunks"]
        self.sources = meta["sources"]

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "source": self.sources[idx],
                "score": float(dist)
            })
        return results