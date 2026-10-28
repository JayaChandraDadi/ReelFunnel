from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from reelfunnel.data import Vocab, history_matrix, item_feature_table, load_kuairand
from reelfunnel.metrics import gauc, ndcg_at_k, recall_at_k
from reelfunnel.ranking.features import CounterFeatures
from reelfunnel.retrieval.index import ItemIndex, ann_recall
from reelfunnel.retrieval.two_tower import TowerConfig, TwoTower, embed_items
from reelfunnel.synth import SynthConfig, write_kuairand_layout


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    d = tmp_path_factory.mktemp("kr")
    write_kuairand_layout(d, SynthConfig(n_users=60, n_videos=120, n_authors=20, days=8, exposures_per_user_day=6))
    log, videos, users = load_kuairand(d)
    return log, videos, users, Vocab.build(log, videos)


def test_metrics():
    assert recall_at_k([1, 2, 3], {3, 9}, 3) == 0.5
    assert ndcg_at_k([9, 1], {9}, 2) == pytest.approx(1.0)
    assert ndcg_at_k([1, 9], {9}, 2) == pytest.approx(1 / np.log2(3))
    users = np.array([1, 1, 1, 2, 2])
    assert gauc(users, np.array([1, 0, 0, 1, 0]), np.array([.9, .1, .2, .3, .4])) == pytest.approx(3 / 5)


def test_history_never_contains_current_or_future(small):
    log, _, _, vocab = small
    H = history_matrix(log, vocab, max_len=5)
    for i in np.random.default_rng(0).choice(len(log), 200, replace=False):
        u, t = log.user_id[i], log.ts[i]
        before = log[(log.user_id == u) & (log.ts < t) & (log.positive == 1)]
        expect = before.video_id.map(vocab.videos).to_numpy()[-5:][::-1]
        assert list(H[i][: len(expect)]) == list(expect)
        assert (H[i][len(expect):] == 0).all()


def test_counter_features_point_in_time(small):
    log, videos, _, vocab = small
    items = item_feature_table(videos, vocab)
    u = log.user_id.map(vocab.users).to_numpy()
    v = log.video_id.map(vocab.videos).to_numpy()
    a = items["author"][v]
    sizes = vocab.sizes()
    cf = CounterFeatures(log, {"user": u, "video": v, "author": a},
                         {"user": sizes["users"], "video": sizes["videos"], "author": sizes["authors"]})
    i = len(log) - 1
    f = cf.at({"user": u[i:i + 1], "video": v[i:i + 1], "author": a[i:i + 1]}, log.ts.to_numpy()[i:i + 1])
    day_start = (log.ts[i] // 86400) * 86400
    ref = log[(log.user_id == log.user_id[i]) & (log.ts < day_start) & (log.ts >= day_start - 7 * 86400)]
    assert f.user_views_7d[0] == len(ref)
    assert f.user_long_views_7d[0] == ref.long_view.sum()


def test_two_tower_learns_and_index_matches_exact(small):
    log, videos, users, vocab = small
    sizes = vocab.sizes()
    items = item_feature_table(videos, vocab)
    m = TwoTower(sizes, 3, items, TowerConfig(dim=16, emb=16, hidden=32))
    uid = torch.as_tensor(log.user_id.map(vocab.users).to_numpy()[:256])
    vid = torch.as_tensor(log.video_id.map(vocab.videos).to_numpy()[:256])
    act, hist, lq = torch.zeros_like(uid), torch.zeros(256, 5, dtype=torch.long), torch.zeros(sizes["videos"])
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    first = None
    for _ in range(30):
        loss = m.loss(uid, act, hist, vid, lq)
        first = first or float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < first
    emb = embed_items(m, sizes["videos"])
    ids = np.arange(1, sizes["videos"])
    ann, exact = ItemIndex(emb[ids], ids), ItemIndex(emb[ids], ids, exact=True)
    q = np.random.default_rng(0).normal(size=(20, 16)).astype(np.float32)
    assert ann_recall(ann, exact, q, 10) > 0.9
    out, _ = ann.search(q, 500)  # more than the corpus: padding must not alias a real id
    assert (out[:, len(ids):] == -1).all()


def test_api_returns_ranked_items(tmp_path):
    """Full train on a tiny dataset, then hit the FastAPI app."""
    import os
    from fastapi.testclient import TestClient

    data = tmp_path / "d"
    write_kuairand_layout(data, SynthConfig(n_users=80, n_videos=150, n_authors=25, days=12, exposures_per_user_day=8))
    from reelfunnel.pipeline import TrainConfig, run
    run(TrainConfig(data=str(data), out=str(tmp_path / "art"), tt_epochs=1, rank_epochs=1, burn_in_days=3,
                    candidates=50, eval_users=50, tt_batch=128))
    os.environ["ARTIFACTS"] = str(tmp_path / "art")
    os.environ["CANDIDATES"] = "50"
    from reelfunnel.serve import app as app_mod
    app_mod._rec = None
    client = TestClient(app_mod.app)
    r = client.get("/recommend", params={"user_id": 0, "k": 5})
    assert r.status_code == 200
    body = r.json()
    scores = [x["score"] for x in body["items"]]
    assert len(scores) == 5 and scores == sorted(scores, reverse=True)
    assert client.get("/recommend", params={"user_id": 10_000}).status_code == 404
    assert "total_ms" in client.get("/metrics").json()
