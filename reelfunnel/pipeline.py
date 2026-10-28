"""End-to-end training and evaluation: retrieval, ranking, funnel, serving artifacts.

    python -m reelfunnel.cli train --data data/synth --out artifacts

Time layout (days are UTC buckets of the log timestamps):

    | burn-in | ranker train ............ | val | test (eval only) |
    | retrieval train ......................... |

Every model sees only data from before the period it is scored on.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import Vocab, history_as_of, history_matrix, item_feature_table, load_kuairand
from .metrics import gauc, mean_metric, ndcg_at_k, recall_at_k
from .ranking.features import COUNTERS, FEATURES, CounterFeatures
from .retrieval.baselines import ItemCF, popularity
from .retrieval.index import ItemIndex, ann_recall
from .retrieval.two_tower import TowerConfig, TwoTower, embed_items, embed_users, train_two_tower

DAY = 86400
DENSE = FEATURES + ["video_age_days", "tt_score"]
LOG_COLS = np.array([i for i, n in enumerate(DENSE)
                     if (n in COUNTERS and COUNTERS[n][1] in ("count", "sum")) or n == "video_age_days"])


@dataclass
class TrainConfig:
    data: str
    out: str = "artifacts"
    test_days: int = 3
    val_days: int = 1
    burn_in_days: int = 7
    hist_len: int = 20
    tt_epochs: int = 3
    tt_batch: int = 1024
    rank_epochs: int = 2
    rank_batch: int = 2048
    rank_max_rows: int = 1_500_000
    candidates: int = 500
    eval_users: int = 3000
    content_emb: str | None = None
    seed: int = 0


class Timer:
    def __init__(self):
        self.t = time.time()
        self.marks = {}

    def mark(self, name):
        now = time.time()
        self.marks[name] = round(now - self.t, 1)
        self.t = now
        print(f"[{name}] {self.marks[name]}s", flush=True)


def _load_content(path: str | None, vocab: Vocab, n: int):
    if not path:
        return None
    z = np.load(path)
    emb = np.zeros((n, z["emb"].shape[1]), np.float32)
    for vid, e in zip(z["video_id"], z["emb"]):
        i = vocab.videos.get(int(vid))
        if i:
            emb[i] = e
    return emb


def user_embeddings_at(model, log, vocab, cutoff_ts, users_idx, active, hist_len):
    """User vectors from each user's history before `cutoff_ts`: the offline daily job."""
    inv = {i: u for u, i in vocab.users.items()}
    hist_map = history_as_of(log, vocab, cutoff_ts, users=[inv[i] for i in users_idx], max_len=hist_len)
    H = np.zeros((len(users_idx), hist_len), np.int32)
    for row, i in enumerate(users_idx):
        h = hist_map.get(inv[i])
        if h is not None:
            H[row] = h
    return embed_users(model, np.asarray(users_idx, np.int64), active[users_idx], H)


def ranker_inputs(feats: pd.DataFrame, u, v, a, items, user_active, ts, tt_score):
    from .ranking.dcn import dense_transform
    f = feats.copy()
    f["video_age_days"] = np.clip((ts - items["upload_ts"][v]) / DAY, 0, None)
    f["tt_score"] = tt_score
    dense = dense_transform(f[DENSE].to_numpy(np.float32), LOG_COLS)
    return {"user": u.astype(np.int32), "video": v.astype(np.int32), "author": a.astype(np.int32),
            "tag": items["tag"][v].astype(np.int32), "dur": items["dur"][v].astype(np.int32),
            "active": user_active[u].astype(np.int32), "hour": ((ts % DAY) // 3600).astype(np.int32),
            "dense": dense}


def run(cfg: TrainConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    T = Timer()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    log, videos, users = load_kuairand(cfg.data)
    vocab = Vocab.build(log, videos)
    sizes = vocab.sizes()
    items = item_feature_table(videos, vocab)
    active_levels = sorted(users.user_active_degree.astype(str).unique())
    active_map = {lvl: i + 1 for i, lvl in enumerate(active_levels)}
    user_active = np.zeros(sizes["users"], np.int32)
    known = users[users.user_id.isin(vocab.users)]
    user_active[known.user_id.map(vocab.users).to_numpy()] = known.user_active_degree.astype(str).map(active_map).to_numpy()

    log["u"] = log.user_id.map(vocab.users).astype(np.int64)
    log["v"] = log.video_id.map(vocab.videos).fillna(0).astype(np.int64)
    log["a"] = items["author"][log.v.to_numpy()]
    log["day"] = (log.ts // DAY).astype(np.int64)
    days = np.sort(log.day.unique())
    test_day = days[-cfg.test_days]
    val_day = days[-(cfg.test_days + cfg.val_days)]
    first_rank_day = days[min(cfg.burn_in_days, len(days) - cfg.test_days - cfg.val_days - 1)]
    T.mark("load")

    hist = history_matrix(log, vocab, cfg.hist_len)
    T.mark("history")

    # ---------------- retrieval: two-tower ----------------
    tr_pos = np.flatnonzero(((log.day < val_day) & (log.positive == 1)).to_numpy())
    freq = np.bincount(log.v.to_numpy()[tr_pos], minlength=sizes["videos"]) + 1.0
    log_q = np.log(freq / freq.sum())
    U, V = log.u.to_numpy(), log.v.to_numpy()

    def batches():
        perm = rng.permutation(tr_pos)
        for s in range(0, len(perm) - cfg.tt_batch + 1, cfg.tt_batch):
            idx = perm[s: s + cfg.tt_batch]
            yield U[idx], user_active[U[idx]], hist[idx], V[idx]

    content = _load_content(cfg.content_emb, vocab, sizes["videos"])
    tcfg = TowerConfig(content_dim=0 if content is None else content.shape[1])
    model = TwoTower(sizes, len(active_map), items, tcfg, content)
    print(f"two-tower: {len(tr_pos):,} positive pairs, {sizes['videos']:,} videos", flush=True)
    train_two_tower(model, batches, log_q, epochs=cfg.tt_epochs)
    item_emb = embed_items(model, sizes["videos"])
    T.mark("train_two_tower")

    # ---------------- retrieval evaluation ----------------
    before_test = log[log.day < test_day]
    seen = before_test[before_test.positive == 1].groupby("u").v.apply(set).to_dict()
    test_pos = log[(log.day >= test_day) & (log.positive == 1)]
    gt = {u: s - seen.get(u, set()) for u, s in test_pos.groupby("u").v.apply(set).items()}
    gt = {u: s for u, s in gt.items() if s}
    eval_u = np.array(sorted(gt))
    if len(eval_u) > cfg.eval_users:
        eval_u = np.sort(rng.choice(eval_u, cfg.eval_users, replace=False))
    q = user_embeddings_at(model, log, vocab, test_day * DAY, eval_u, user_active, cfg.hist_len)

    end_ts = log.ts.max()
    live = np.flatnonzero(~(items["upload_ts"] > end_ts))
    live = live[live > 0]
    ann = ItemIndex(item_emb[live], live)
    exact = ItemIndex(item_emb[live], live, exact=True)
    K = cfg.candidates

    def topk_filtered(index, queries, users_, k):
        res, _ = index.search(queries, k + 200)
        return [[x for x in row if x > 0 and x not in seen.get(u, ())][:k] for row, u in zip(res, users_)]

    t_ann = time.time()
    tt_lists = topk_filtered(ann, q, eval_u, K)
    ann_ms = (time.time() - t_ann) * 1000 / len(eval_u)
    tt_exact = topk_filtered(exact, q, eval_u, K)
    pop = popularity(before_test, vocab.videos, k=K + 500)
    pop_lists = [[x for x in pop if x not in seen.get(u, ())][:K] for u in eval_u]
    cf = ItemCF(before_test, vocab.videos)
    inv = {i: u for u, i in vocab.users.items()}
    hist_eval = history_as_of(log, vocab, test_day * DAY, users=[inv[u] for u in eval_u], max_len=cfg.hist_len)
    cf_lists = [list(cf.recommend(hist_eval.get(inv[u], np.zeros(cfg.hist_len, np.int32)), K, seen.get(u)))
                for u in eval_u]
    rel = [gt[u] for u in eval_u]

    retrieval = {}
    for name, lists in [("popularity", pop_lists), ("item_cf", cf_lists),
                        ("two_tower_hnsw", tt_lists), ("two_tower_exact", tt_exact)]:
        retrieval[name] = {f"recall@{k}": round(mean_metric(recall_at_k, lists, rel, k), 4) for k in (50, 200, K)}
        retrieval[name]["ndcg@50"] = round(mean_metric(ndcg_at_k, lists, rel, 50), 4)
    retrieval["hnsw_vs_exact_recall@100"] = round(ann_recall(ann, exact, q, 100), 4)
    retrieval["hnsw_ms_per_query"] = round(ann_ms, 3)
    retrieval["eval_users"] = int(len(eval_u))
    print(json.dumps(retrieval, indent=1), flush=True)
    T.mark("eval_retrieval")

    # ---------------- ranking: DCN v2 ----------------
    import tensorflow as tf
    import keras

    from .ranking.dcn import CAT, build_ranker, make_predictor

    tf.random.set_seed(cfg.seed)
    cf_feats = CounterFeatures(log, {"user": log.u.to_numpy(), "video": log.v.to_numpy(), "author": log.a.to_numpy()},
                               {"user": sizes["users"], "video": sizes["videos"], "author": sizes["authors"]})

    rank_days = [d for d in days if d >= first_rank_day]
    all_users = np.arange(1, sizes["users"])
    day_user_emb = {}
    for d in rank_days:
        emb = np.zeros((sizes["users"], tcfg.dim), np.float32)
        emb[all_users] = user_embeddings_at(model, log, vocab, d * DAY, all_users, user_active, cfg.hist_len)
        day_user_emb[d] = emb
    T.mark("daily_user_embeddings")

    def rows_to_inputs(df: pd.DataFrame):
        u, v, a, ts, d = (df[c].to_numpy() for c in ("u", "v", "a", "ts", "day"))
        feats = cf_feats.at(dict(user=u, video=v, author=a), ts)
        tt = np.empty(len(df), np.float32)
        for dd in np.unique(d):
            m = d == dd
            tt[m] = (day_user_emb[dd][u[m]] * item_emb[v[m]]).sum(1)
        return ranker_inputs(feats, u, v, a, items, user_active, ts, tt), df.long_view.to_numpy(np.float32)

    tr = log[(log.day >= first_rank_day) & (log.day < val_day)]
    if len(tr) > cfg.rank_max_rows:
        tr = tr.sample(cfg.rank_max_rows, random_state=cfg.seed)
    va = log[(log.day >= val_day) & (log.day < test_day)]
    te_all = log[log.day >= test_day]
    te = te_all[te_all.is_rand == 1] if (te_all.is_rand == 1).sum() >= 2000 else te_all
    Xtr, ytr = rows_to_inputs(tr)
    Xva, yva = rows_to_inputs(va)
    Xte, yte = rows_to_inputs(te)
    T.mark("ranker_features")

    card = {"user": sizes["users"], "video": sizes["videos"], "author": sizes["authors"], "tag": sizes["tags"],
            "dur": 16, "active": len(active_map) + 1, "hour": 24}
    sample = Xtr["dense"][rng.choice(len(ytr), min(50_000, len(ytr)), replace=False)]
    ranker = build_ranker(card, len(DENSE), sample)
    ranker.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=cfg.rank_epochs, batch_size=cfg.rank_batch, verbose=2)
    T.mark("train_ranker")

    from sklearn.metrics import log_loss, roc_auc_score
    p_te = ranker.predict(Xte, batch_size=8192, verbose=0).ravel()
    # Baseline ranker: retrieval similarity alone, to show what the heavy model adds.
    ranking = {
        "test_rows": int(len(yte)),
        "test_is_random_exposure": bool((te.is_rand == 1).all()),
        "positive_rate": round(float(yte.mean()), 4),
        "auc_dcn": round(roc_auc_score(yte, p_te), 4),
        "gauc_dcn": round(gauc(te.u.to_numpy(), yte, p_te), 4),
        "logloss_dcn": round(log_loss(yte, np.clip(p_te, 1e-6, 1 - 1e-6)), 4),
        "auc_two_tower_score_only": round(roc_auc_score(yte, Xte["dense"][:, DENSE.index("tt_score")]), 4),
    }
    print(json.dumps(ranking, indent=1), flush=True)

    # ---------------- ranking stage: re-order each user's real test impressions ----------------
    # Retrieval is judged on recall over the whole corpus; the ranker's job is ordering what is
    # shown, so it is judged per user on the impressions that user actually received in the test
    # window. (Scoring the ranker against logged engagements on retrieved candidates would reward
    # copying the old policy's exposure, not predicting engagement.)
    Xall, yall = rows_to_inputs(te_all)
    scorers = {
        "dcn_v2": ranker.predict(Xall, batch_size=8192, verbose=0).ravel(),
        "two_tower_score": Xall["dense"][:, DENSE.index("tt_score")],
        "video_long_view_rate": Xall["dense"][:, DENSE.index("video_long_view_rate_7d")],
        "random": rng.random(len(yall)),
    }
    users_all = te_all.u.to_numpy()
    order = np.argsort(users_all, kind="stable")
    bounds = np.flatnonzero(np.diff(users_all[order])) + 1
    groups = [g for g in np.split(order, bounds) if len(g) >= 10 and 0 < yall[g].sum() < len(g)]
    impression_ranking = {"users": len(groups), "impressions": int(sum(len(g) for g in groups))}
    for name, s in scorers.items():
        ndcg = [ndcg_at_k(list(g[np.argsort(-s[g])]), set(g[yall[g] > 0]), 10) for g in groups]
        impression_ranking[f"ndcg@10_{name}"] = round(float(np.mean(ndcg)), 4)
    lift = impression_ranking["ndcg@10_dcn_v2"] / impression_ranking["ndcg@10_two_tower_score"] - 1
    impression_ranking["dcn_lift_over_two_tower_order"] = f"{lift:+.1%}"
    funnel = impression_ranking
    print(json.dumps(funnel, indent=1), flush=True)
    T.mark("eval_ranking_stage")

    # ---------------- serving artifacts: state as of the end of the log ----------------
    snap_day = int(days[-1]) + 1
    user_emb_now = np.zeros((sizes["users"], tcfg.dim), np.float32)
    user_emb_now[all_users] = user_embeddings_at(model, log, vocab, snap_day * DAY, all_users, user_active, cfg.hist_len)
    zeros = lambda n: np.zeros(n, np.int64)  # noqa: E731
    tables = {}
    for ent, n in (("user", sizes["users"]), ("video", sizes["videos"]), ("author", sizes["authors"])):
        keys = {"user": zeros(n), "video": zeros(n), "author": zeros(n)}
        keys[ent] = np.arange(n)
        df = cf_feats.at_day(keys, np.full(n, snap_day - cf_feats.day0))
        for col in df.columns:
            if col.startswith(ent + "_"):
                tables[col] = df[col].to_numpy(np.float32)
    np.savez_compressed(out / "feature_tables.npz", **tables)
    np.save(out / "item_emb.npy", item_emb)
    np.save(out / "user_emb.npy", user_emb_now)
    ann.save(str(out / "items.hnsw"))
    ranker.save(str(out / "ranker.keras"))
    torch.save(model.state_dict(), out / "two_tower.pt")
    np.savez_compressed(out / "item_table.npz", **items, user_active=user_active)
    seen_now = log[log.positive == 1].groupby("u").v.apply(lambda s: list(map(int, s.tail(200)))).to_dict()
    meta = {
        "config": asdict(cfg), "dense": DENSE, "log_cols": LOG_COLS.tolist(), "cat": CAT,
        "snapshot_ts": snap_day * DAY, "dim": tcfg.dim,
        "users": {str(k): int(v) for k, v in vocab.users.items()},
        "videos_inv": {int(v): int(k) for k, v in vocab.videos.items()},
        "seen": {str(k): v for k, v in seen_now.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta))
    metrics = {"retrieval": retrieval, "ranking": ranking, "ranking_stage": funnel, "timings_s": T.marks,
               "data": {"rows": int(len(log)), "users": int(sizes["users"] - 1), "videos": int(sizes["videos"] - 1),
                        "days": int(len(days))}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    T.mark("export")
    return metrics
