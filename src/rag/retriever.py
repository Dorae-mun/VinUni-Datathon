from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class RagRetriever:
    def __init__(self, rag_dir: Path):
        self.rag_dir = Path(rag_dir)
        self.vectorizer = None
        self.matrix = None
        self.metadatas: List[Dict[str, str]] = []
        self.texts: List[str] = []

    def load(self) -> bool:
        vectorizer_path = self.rag_dir / "vectorizer.joblib"
        matrix_path = self.rag_dir / "matrix.joblib"
        metadata_path = self.rag_dir / "metadata.json"
        if not (vectorizer_path.exists() and matrix_path.exists() and metadata_path.exists()):
            return False

        self.vectorizer = joblib.load(vectorizer_path)
        self.matrix = joblib.load(matrix_path)
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.metadatas = payload.get("metadatas", [])
        self.texts = payload.get("texts", [])
        return True

    def is_ready(self) -> bool:
        return self.vectorizer is not None and self.matrix is not None and len(self.texts) > 0

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, str]]:
        if not self.is_ready() or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        if sims.size == 0:
            return []

        top_indices = np.argsort(sims)[::-1][:top_k]
        hits: List[Dict[str, str]] = []
        for idx in top_indices:
            score = float(sims[idx])
            if score <= 0:
                continue
            meta = self.metadatas[idx] if idx < len(self.metadatas) else {}
            text = self.texts[idx] if idx < len(self.texts) else ""
            hits.append(
                {
                    "source": meta.get("source", ""),
                    "chunk_id": meta.get("chunk_id", ""),
                    "content": text,
                    "score": f"{score:.4f}",
                }
            )
        return hits

