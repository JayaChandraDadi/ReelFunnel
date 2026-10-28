"""Non-learned candidate sources the two-tower model has to beat."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def popularity(train: pd.DataFrame, vocab_videos: dict, recent_days: int = 7, k: int = 500) -> np.ndarray:
    """Most-engaged videos over the last `recent_days` of training."""
    last = np.sort(train.date.unique())[-recent_days:]
    pos = train[(train.positive == 1) & train.date.isin(last)]
    top = pos.video_id.value_counts().index[:k]
    return np.array([vocab_videos[v] for v in top], dtype=np.int64)


class ItemCF:
    """Item-to-item co-occurrence from user positive histories, cosine-normalised."""

    def __init__(self, train: pd.DataFrame, vocab_videos: dict, max_hist: int = 50, top_neighbors: int = 50):
        pos = train[train.positive == 1].sort_values("ts")
        seqs = pos.groupby("user_id").video_id.apply(lambda s: [vocab_videos[v] for v in s.tail(max_hist)])
        co: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        cnt: dict[int, int] = defaultdict(int)
        for seq in seqs:
            uniq = list(dict.fromkeys(seq))
            for a in uniq:
                cnt[a] += 1
            w = 1.0 / np.log1p(len(uniq) + 1)  # down-weight very active users
            for i, a in enumerate(uniq):
                for b in uniq[i + 1:]:
                    co[a][b] += w
                    co[b][a] += w
        self.nn = {}
        for a, nbrs in co.items():
            items = sorted(((b, s / np.sqrt(cnt[a] * cnt[b])) for b, s in nbrs.items()), key=lambda x: -x[1])
            self.nn[a] = items[:top_neighbors]

    def recommend(self, history: np.ndarray, k: int, exclude: set | None = None) -> np.ndarray:
        scores: dict[int, float] = defaultdict(float)
        for rank, h in enumerate(history[history > 0]):
            decay = 0.9 ** rank  # newer history counts more
            for b, s in self.nn.get(int(h), []):
                scores[b] += decay * s
        if exclude:
            for e in exclude:
                scores.pop(e, None)
        return np.array([b for b, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]], dtype=np.int64)
