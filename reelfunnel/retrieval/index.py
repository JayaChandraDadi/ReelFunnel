"""ANN serving for item embeddings (FAISS HNSW), plus an exact index for checking it."""
from __future__ import annotations

import numpy as np

import faiss


class ItemIndex:
    """HNSW over normalised item vectors; inner product equals cosine similarity.

    `ids` maps index rows back to video vocab ids, so a corpus can be any subset
    (for example, only videos that are live at serving time).
    """

    def __init__(self, vectors: np.ndarray, ids: np.ndarray, m: int = 32, ef_construction: int = 200,
                 ef_search: int = 128, exact: bool = False):
        self.ids = np.asarray(ids)
        d = vectors.shape[1]
        if exact:
            self.index = faiss.IndexFlatIP(d)
        else:
            self.index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self.index, "hnsw") and self.index.hnsw.efSearch < k:
            self.index.hnsw.efSearch = k  # HNSW cannot return more good results than its search beam
        scores, rows = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        # FAISS pads with -1 when fewer than k results exist; keep the padding visible.
        return np.where(rows >= 0, self.ids[np.clip(rows, 0, None)], -1), scores

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)
        np.save(path + ".ids.npy", self.ids)

    @classmethod
    def load(cls, path: str, ef_search: int = 128) -> "ItemIndex":
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(path)
        if hasattr(obj.index, "hnsw"):
            obj.index.hnsw.efSearch = ef_search
        obj.ids = np.load(path + ".ids.npy")
        return obj


def ann_recall(ann: ItemIndex, exact: ItemIndex, queries: np.ndarray, k: int) -> float:
    """Share of the exact top-k that HNSW also returns."""
    a, _ = ann.search(queries, k)
    e, _ = exact.search(queries, k)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, e)]))
