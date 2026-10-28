"""Synthetic short-video logs with the KuaiRand-Pure schema.

Real KuaiRand is the target dataset. This generator exists so tests and smoke runs
work offline. It plants real structure (latent user tastes, author styles, video
quality, popularity skew, new uploads over time) so models have something to learn
and leakage has something to inflate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd


@dataclass
class SynthConfig:
    n_users: int = 2000
    n_videos: int = 5000
    n_authors: int = 600
    n_tags: int = 30
    days: int = 21
    exposures_per_user_day: int = 30
    random_exposure_share: float = 0.05
    dim: int = 16
    start: date = date(2022, 4, 8)
    seed: int = 7


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(cfg: SynthConfig = SynthConfig()):
    rng = np.random.default_rng(cfg.seed)
    d = cfg.dim

    tag_vec = rng.normal(0, 1, (cfg.n_tags, d))
    author_tag = rng.integers(0, cfg.n_tags, cfg.n_authors)
    author_vec = tag_vec[author_tag] + rng.normal(0, 0.5, (cfg.n_authors, d))

    video_author = rng.integers(0, cfg.n_authors, cfg.n_videos)
    video_tag = np.where(rng.random(cfg.n_videos) < 0.8,
                         author_tag[video_author], rng.integers(0, cfg.n_tags, cfg.n_videos))
    video_vec = 0.6 * author_vec[video_author] + 0.6 * tag_vec[video_tag] \
        + rng.normal(0, 0.4, (cfg.n_videos, d))
    video_vec /= np.linalg.norm(video_vec, axis=1, keepdims=True)
    quality = rng.normal(0, 0.6, cfg.n_videos)
    duration_ms = (rng.lognormal(10.1, 0.6, cfg.n_videos)).astype(np.int64)  # ~25s median
    # A third of videos upload during the window, so cold start is real.
    upload_day = np.where(rng.random(cfg.n_videos) < 0.66, -rng.integers(1, 60, cfg.n_videos),
                          rng.integers(0, cfg.days, cfg.n_videos))
    video_type = np.where(rng.random(cfg.n_videos) < 0.9, "NORMAL", "AD")

    n_pref = rng.integers(1, 4, cfg.n_users)
    user_vec = np.zeros((cfg.n_users, d))
    for u in range(cfg.n_users):
        tags = rng.choice(cfg.n_tags, n_pref[u], replace=False)
        user_vec[u] = tag_vec[tags].sum(0) + rng.normal(0, 0.5, d)
    user_vec /= np.linalg.norm(user_vec, axis=1, keepdims=True)
    activity = rng.lognormal(0, 0.5, cfg.n_users)
    activity /= activity.mean()

    pop_logit = 1.5 * quality + rng.gumbel(0, 1, cfg.n_videos)

    frames = []
    for day in range(cfg.days):
        live = np.flatnonzero(upload_day <= day)
        n_exp = np.maximum(1, rng.poisson(cfg.exposures_per_user_day * activity)).astype(int)
        users = np.repeat(np.arange(cfg.n_users), n_exp)
        n = len(users)
        is_rand = rng.random(n) < cfg.random_exposure_share

        # Logged policy: personalised softmax over a random slate of live videos.
        slate = rng.choice(live, size=(n, 40))
        aff = np.einsum("nd,nkd->nk", user_vec[users], video_vec[slate]) * 4 + 0.3 * pop_logit[slate]
        aff -= aff.max(1, keepdims=True)
        p = np.exp(aff)
        p /= p.sum(1, keepdims=True)
        pick = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
        videos = slate[np.arange(n), pick]
        videos[is_rand] = rng.choice(live, is_rand.sum())

        affinity = (user_vec[users] * video_vec[videos]).sum(1)
        freshness = np.exp(-(day - np.maximum(upload_day[videos], 0)) / 10)
        base = 3.0 * affinity + 0.8 * quality[videos] + 0.3 * freshness - 1.6
        is_click = rng.random(n) < _sigmoid(base + 0.6)
        watch_frac = np.clip(_sigmoid(base + rng.normal(0, 0.8, n)) * 1.3, 0.02, 1.0)
        play = (watch_frac * duration_ms[videos]).astype(np.int64)
        long_view = (watch_frac > 0.65) | (play > 18_000)
        is_like = long_view & (rng.random(n) < _sigmoid(base - 1.2))
        is_follow = is_like & (rng.random(n) < 0.08)
        is_comment = long_view & (rng.random(n) < _sigmoid(base - 3.0))
        is_forward = long_view & (rng.random(n) < _sigmoid(base - 3.3))

        secs = rng.integers(0, 86_400, n)
        day0 = pd.Timestamp(cfg.start + timedelta(days=day))
        time_ms = (day0.value // 1_000_000) + secs * 1000
        frames.append(pd.DataFrame({
            "user_id": users,
            "video_id": videos,
            "date": int((cfg.start + timedelta(days=day)).strftime("%Y%m%d")),
            "hourmin": (secs // 3600) * 100 + (secs % 3600) // 60,
            "time_ms": time_ms,
            "is_click": is_click.astype(np.int8),
            "is_like": is_like.astype(np.int8),
            "is_follow": is_follow.astype(np.int8),
            "is_comment": is_comment.astype(np.int8),
            "is_forward": is_forward.astype(np.int8),
            "long_view": long_view.astype(np.int8),
            "play_time_ms": play,
            "duration_ms": duration_ms[videos],
            "is_rand": is_rand.astype(np.int8),
            "tab": rng.integers(0, 4, n),
        }))
    log = pd.concat(frames, ignore_index=True).sort_values("time_ms", kind="stable").reset_index(drop=True)

    videos_df = pd.DataFrame({
        "video_id": np.arange(cfg.n_videos),
        "author_id": video_author,
        "video_type": video_type,
        "upload_dt": [(cfg.start + timedelta(days=int(x))).isoformat() for x in upload_day],
        "video_duration": duration_ms,
        "tag": video_tag.astype(str),
    })
    users_df = pd.DataFrame({
        "user_id": np.arange(cfg.n_users),
        "user_active_degree": pd.cut(activity, [0, 0.6, 1.2, np.inf],
                                     labels=["low", "middle", "high"]).astype(str),
        "register_days_range": rng.choice(["7-30", "31-60", "61-90", "91-180", "181-365", "366-730"],
                                          cfg.n_users),
    })
    return log, videos_df, users_df


def write_kuairand_layout(out_dir, cfg: SynthConfig = SynthConfig()) -> None:
    """Write CSVs with the same file names as KuaiRand-Pure, so loaders are shared."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log, videos, users = generate(cfg)
    std = log[log.is_rand == 0]
    rnd = log[log.is_rand == 1]
    std.to_csv(out / "log_standard_4_08_to_4_21_pure.csv", index=False)
    rnd.to_csv(out / "log_random_4_22_to_5_08_pure.csv", index=False)
    videos.to_csv(out / "video_features_basic_pure.csv", index=False)
    users.to_csv(out / "user_features_pure.csv", index=False)
