"""Load KuaiRand-Pure logs and turn them into retrieval and ranking datasets.

Everything here is time-aware. A user's history for an event only contains
positives that happened strictly before it, and the train/test split is by date,
so offline numbers mean what they would mean in production.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

POSITIVE_COLS = ["long_view", "is_like", "is_follow", "is_forward"]
DUR_EDGES = np.array([7_000, 15_000, 30_000, 60_000, 120_000, 300_000])  # ms


@dataclass
class Vocab:
    """Contiguous ids. Index 0 is reserved for padding / unknown everywhere."""
    users: dict
    videos: dict
    authors: dict
    tags: dict

    @classmethod
    def build(cls, log: pd.DataFrame, videos: pd.DataFrame) -> "Vocab":
        def m(values):
            return {v: i + 1 for i, v in enumerate(pd.unique(values))}
        return cls(m(log.user_id), m(videos.video_id), m(videos.author_id), m(videos.tag.astype(str)))

    def sizes(self) -> dict:
        return {k: len(getattr(self, k)) + 1 for k in ("users", "videos", "authors", "tags")}


def load_kuairand(data_dir: str | Path, include_random: bool = True):
    data_dir = Path(data_dir)
    parts = sorted(data_dir.glob("log_standard_*.csv"))
    if include_random:
        parts += sorted(data_dir.glob("log_random_*.csv"))
    if not parts:
        raise FileNotFoundError(f"no KuaiRand logs in {data_dir}; run `python -m reelfunnel.cli synth`")
    log = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    if "is_rand" not in log:
        log["is_rand"] = 0
    log = log.sort_values("time_ms", kind="stable").reset_index(drop=True)
    log["ts"] = log.time_ms / 1000.0
    log["positive"] = log[[c for c in POSITIVE_COLS if c in log]].max(axis=1).astype(np.int8)
    log["watch_ratio"] = (log.play_time_ms / log.duration_ms.clip(lower=1)).clip(upper=5.0)

    videos = pd.read_csv(data_dir / "video_features_basic_pure.csv")
    videos["tag"] = videos["tag"].astype(str)
    dur = videos["video_duration"] if "video_duration" in videos else pd.Series(0, index=videos.index)
    videos["dur_bucket"] = np.searchsorted(DUR_EDGES, dur.fillna(0).to_numpy()) + 1
    users_path = data_dir / "user_features_pure.csv"
    users = pd.read_csv(users_path) if users_path.exists() else pd.DataFrame({"user_id": log.user_id.unique()})
    if "user_active_degree" not in users:
        users["user_active_degree"] = "unknown"
    log = log.merge(videos[["video_id", "author_id", "tag"]], on="video_id", how="left")
    return log, videos, users


def date_split(log: pd.DataFrame, test_days: int = 3, val_days: int = 1):
    dates = np.sort(log.date.unique())
    test_start = dates[-test_days]
    val_start = dates[-(test_days + val_days)]
    return val_start, test_start


def history_matrix(log: pd.DataFrame, vocab: Vocab, max_len: int = 20) -> np.ndarray:
    """For every row, the user's last `max_len` positive videos strictly before it.

    One pass in time order per user. Row i never sees its own label.
    """
    out = np.zeros((len(log), max_len), dtype=np.int32)
    vid = log.video_id.map(vocab.videos).fillna(0).to_numpy(np.int32)
    pos = log.positive.to_numpy()
    ts = log.ts.to_numpy()
    for _, idx in log.groupby("user_id", sort=False).indices.items():
        buf: list[int] = []
        j = 0  # positives with ts < current ts are moved into the buffer lazily
        for i in idx:
            while j < len(idx) and ts[idx[j]] < ts[i]:
                if pos[idx[j]]:
                    buf.append(vid[idx[j]])
                j += 1
            h = buf[-max_len:][::-1]  # newest first
            out[i, : len(h)] = h
    return out


def history_as_of(log: pd.DataFrame, vocab: Vocab, cutoff_ts: float, users=None, max_len: int = 20) -> dict:
    """Each user's newest positives before `cutoff_ts`: the state an offline user-embedding job sees."""
    sub = log[(log.ts < cutoff_ts) & (log.positive == 1)]
    if users is not None:
        sub = sub[sub.user_id.isin(users)]
    sub = sub.sort_values("ts")
    hist = {}
    for u, g in sub.groupby("user_id"):
        v = g.video_id.map(vocab.videos).fillna(0).astype(np.int32).to_numpy()[-max_len:][::-1]
        row = np.zeros(max_len, np.int32)
        row[: len(v)] = v
        hist[u] = row
    return hist


def item_feature_table(videos: pd.DataFrame, vocab: Vocab) -> dict[str, np.ndarray]:
    """Dense arrays indexed by video vocab id, used by the item tower and the ranker."""
    n = len(vocab.videos) + 1
    author = np.zeros(n, np.int32)
    tag = np.zeros(n, np.int32)
    dur = np.zeros(n, np.int32)
    ids = videos.video_id.map(vocab.videos).to_numpy()
    author[ids] = videos.author_id.map(vocab.authors).to_numpy()
    tag[ids] = videos.tag.map(vocab.tags).to_numpy()
    dur[ids] = videos.dur_bucket.to_numpy()
    upload = np.full(n, np.nan)
    if "upload_dt" in videos:
        upload[ids] = pd.to_datetime(videos.upload_dt, errors="coerce").astype("int64").to_numpy() / 1e9
    return {"author": author, "tag": tag, "dur": dur, "upload_ts": upload}
