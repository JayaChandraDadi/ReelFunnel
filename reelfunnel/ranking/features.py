"""Point-in-time counter features for the ranker.

Same feature names and semantics as WindowForge's kuairand.yaml (day buckets,
features for day d use days [d - W, d - 1]), implemented with dense cumulative
sums so they are fast enough to recompute inside one training run. Point the
trainer at WindowForge output instead with `--features` to use the Spark pipeline.

The same object answers serving lookups: `table(entity, day)` returns every key's
values as of the start of `day`, which is what the online store holds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86400

# name -> (entity, kind, source column, window days)
COUNTERS = {
    "user_views_1d": ("user", "count", None, 1),
    "user_views_7d": ("user", "count", None, 7),
    "user_clicks_7d": ("user", "sum", "is_click", 7),
    "user_long_views_7d": ("user", "sum", "long_view", 7),
    "user_likes_14d": ("user", "sum", "is_like", 14),
    "user_watch_ratio_7d": ("user", "mean", "watch_ratio", 7),
    "video_views_1d": ("video", "count", None, 1),
    "video_views_7d": ("video", "count", None, 7),
    "video_long_views_7d": ("video", "sum", "long_view", 7),
    "video_likes_7d": ("video", "sum", "is_like", 7),
    "video_watch_ratio_7d": ("video", "mean", "watch_ratio", 7),
    "author_views_7d": ("author", "count", None, 7),
    "author_long_views_7d": ("author", "sum", "long_view", 7),
    "author_follows_14d": ("author", "sum", "is_follow", 14),
}
# name -> (numerator, denominator, smoothing, prior)
RATIOS = {
    "user_long_view_rate_7d": ("user_long_views_7d", "user_views_7d", 5, 0.3),
    "video_long_view_rate_7d": ("video_long_views_7d", "video_views_7d", 20, 0.3),
    "video_like_rate_7d": ("video_likes_7d", "video_views_7d", 20, 0.05),
    "author_long_view_rate_7d": ("author_long_views_7d", "author_views_7d", 50, 0.3),
}
FEATURES = list(COUNTERS) + list(RATIOS)


class CounterFeatures:
    def __init__(self, log: pd.DataFrame, keys: dict[str, np.ndarray], sizes: dict[str, int]):
        """`keys[entity]` is the vocab id of that entity for every log row."""
        self.day0 = int(log.ts.min() // DAY)
        day = (log.ts.to_numpy() // DAY).astype(np.int64) - self.day0
        self.n_days = int(day.max()) + 2
        self.cum: dict[tuple[str, str], np.ndarray] = {}
        cols = {"_one": np.ones(len(log))}
        for _, kind, col, _ in COUNTERS.values():
            if col and col not in cols:
                cols[col] = log[col].fillna(0).to_numpy(float)
        needed: dict[str, set] = {}
        for ent, _, col, _ in COUNTERS.values():
            needed.setdefault(ent, {"_one"}).add(col or "_one")
        for ent, k in keys.items():
            for name in needed.get(ent, ()):
                arr = cols[name]
                grid = np.zeros((sizes[ent], self.n_days))
                np.add.at(grid, (k, day), arr)
                # cum[:, t] = sum of days < t, so a window ending before day d is cum[d] - cum[d - W]
                self.cum[(ent, name)] = np.concatenate([np.zeros((sizes[ent], 1)), grid.cumsum(1)], 1)

    def _window(self, ent: str, col: str, key: np.ndarray, day: np.ndarray, w: int) -> np.ndarray:
        c = self.cum[(ent, col)]
        d = np.clip(day, 0, c.shape[1] - 1)
        lo = np.clip(d - w, 0, None)
        return c[key, d] - c[key, lo]

    def at(self, keys: dict[str, np.ndarray], ts: np.ndarray) -> pd.DataFrame:
        """Features for rows at timestamps `ts` (seconds), using whole days before each row's day."""
        day = (np.asarray(ts) // DAY).astype(np.int64) - self.day0
        return self.at_day(keys, day)

    def at_day(self, keys: dict[str, np.ndarray], day: np.ndarray) -> pd.DataFrame:
        out = {}
        for name, (ent, kind, col, w) in COUNTERS.items():
            k = keys[ent]
            if kind == "count":
                out[name] = self._window(ent, "_one", k, day, w)
            elif kind == "sum":
                out[name] = self._window(ent, col, k, day, w)
            else:
                s = self._window(ent, col, k, day, w)
                n = self._window(ent, "_one", k, day, w)
                out[name] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        for name, (num, den, a, prior) in RATIOS.items():
            out[name] = (out[num] + a * prior) / (out[den] + a)
        return pd.DataFrame(out)

    def snapshot_day(self) -> int:
        """Day index for 'now': the first day after the last logged day."""
        return self.n_days - 1
