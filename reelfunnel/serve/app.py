"""FastAPI recommendation service: retrieve with HNSW, rank with DCN v2, return top-k.

The request path follows the usual split between a feed service and a retrieval
service: look up the user's precomputed embedding (the offline job's output),
query the ANN index, hydrate counter features from in-memory tables (the online
store's contents), score all candidates in one batched call, sort.

    uvicorn reelfunnel.serve.app:app --port 8000      (ARTIFACTS=artifacts by default)
    curl 'localhost:8000/recommend?user_id=42&k=10'
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from ..pipeline import DENSE, ranker_inputs
from ..ranking.features import COUNTERS, RATIOS
from ..retrieval.index import ItemIndex

DAY = 86400


class Recommender:
    def __init__(self, art: str | Path, candidates: int = 500):
        import tensorflow as tf  # noqa: F401  (registers the custom layer before load)
        import keras

        from ..ranking.dcn import Cross, make_predictor  # noqa: F401

        art = Path(art)
        self.meta = json.loads((art / "meta.json").read_text())
        self.users = {int(k): v for k, v in self.meta["users"].items()}
        self.videos_inv = {int(k): v for k, v in self.meta["videos_inv"].items()}
        self.seen = {int(k): set(v) for k, v in self.meta["seen"].items()}
        self.user_emb = np.load(art / "user_emb.npy")
        self.item_emb = np.load(art / "item_emb.npy")
        self.tables = dict(np.load(art / "feature_tables.npz"))
        it = np.load(art / "item_table.npz")
        self.items = {k: it[k] for k in ("author", "tag", "dur", "upload_ts")}
        self.user_active = it["user_active"]
        self.index = ItemIndex.load(str(art / "items.hnsw"))
        self.ranker = keras.models.load_model(art / "ranker.keras")
        self.predict = make_predictor(self.ranker, len(DENSE))
        self.candidates = candidates
        self.lat = defaultdict(lambda: deque(maxlen=5000))
        self.recommend(next(iter(self.users)), 10)  # trace the TF graph before traffic

    def _features(self, u: int, v: np.ndarray):
        import pandas as pd
        a = self.items["author"][v]
        cols = {}
        for name, (ent, *_rest) in COUNTERS.items():
            key = {"user": np.full(len(v), u), "video": v, "author": a}[ent]
            cols[name] = self.tables[name][key]
        for name, (num, den, k, prior) in RATIOS.items():
            cols[name] = (cols[num] + k * prior) / (cols[den] + k)
        return pd.DataFrame(cols), a

    def recommend(self, raw_user_id: int, k: int = 10, now: float | None = None) -> dict:
        t0 = time.perf_counter()
        u = self.users.get(raw_user_id)
        if u is None:
            raise KeyError(raw_user_id)
        q = self.user_emb[u: u + 1]
        cand, _ = self.index.search(q, self.candidates + 200)
        seen = self.seen.get(u, set())
        v = np.array([x for x in cand[0] if x > 0 and x not in seen][: self.candidates], dtype=np.int64)
        t1 = time.perf_counter()

        now = now or float(self.meta["snapshot_ts"])
        feats, a = self._features(u, v)
        tt = self.item_emb[v] @ q[0]
        x = ranker_inputs(feats, np.full(len(v), u), v, a, self.items, self.user_active, np.full(len(v), now), tt)
        t2 = time.perf_counter()
        scores = self.predict(x).numpy()
        order = np.argsort(-scores)[:k]
        t3 = time.perf_counter()

        timing = {"retrieve_ms": (t1 - t0) * 1e3, "features_ms": (t2 - t1) * 1e3,
                  "rank_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3}
        for name, ms in timing.items():
            self.lat[name].append(ms)
        return {"user_id": raw_user_id, "candidates": int(len(v)),
                "items": [{"video_id": self.videos_inv[int(v[i])], "score": round(float(scores[i]), 4)} for i in order],
                "timing_ms": {k_: round(t, 2) for k_, t in timing.items()}}

    def latency_summary(self) -> dict:
        return {n: {"p50": round(float(np.percentile(d, 50)), 2), "p95": round(float(np.percentile(d, 95)), 2),
                    "n": len(d)} for n, d in self.lat.items() if d}


app = FastAPI(title="ReelFunnel")
_rec: Recommender | None = None


def get_rec() -> Recommender:
    global _rec
    if _rec is None:
        _rec = Recommender(os.environ.get("ARTIFACTS", "artifacts"), int(os.environ.get("CANDIDATES", 500)))
    return _rec


@app.get("/healthz")
def healthz():
    get_rec()
    return {"ok": True}


@app.get("/recommend")
def recommend(user_id: int, k: int = 10):
    try:
        return get_rec().recommend(user_id, min(max(k, 1), 100))
    except KeyError:
        raise HTTPException(404, f"unknown user {user_id}")


@app.get("/metrics")
def metrics():
    return get_rec().latency_summary()
