"""reelfunnel command line.

    python -m reelfunnel.cli synth --out data/synth
    python -m reelfunnel.cli train --data data/synth --out artifacts
    python -m reelfunnel.cli bench --artifacts artifacts --requests 500
    uvicorn reelfunnel.serve.app:app --port 8000
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main(argv=None):
    p = argparse.ArgumentParser(prog="reelfunnel")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    s.add_argument("--users", type=int, default=2000)
    s.add_argument("--videos", type=int, default=5000)
    s.add_argument("--days", type=int, default=21)
    s.add_argument("--seed", type=int, default=7)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", default="artifacts")
    t.add_argument("--tt-epochs", type=int, default=3)
    t.add_argument("--rank-epochs", type=int, default=2)
    t.add_argument("--rank-max-rows", type=int, default=1_500_000)
    t.add_argument("--eval-users", type=int, default=3000)
    t.add_argument("--content-emb", default=None, help="CoView export (.npz with video_id, emb)")

    b = sub.add_parser("bench", help="in-process latency benchmark of the full request path")
    b.add_argument("--artifacts", default="artifacts")
    b.add_argument("--requests", type=int, default=500)
    b.add_argument("--candidates", type=int, default=500)

    a = p.parse_args(argv)
    if a.cmd == "synth":
        from .synth import SynthConfig, write_kuairand_layout
        write_kuairand_layout(a.out, SynthConfig(n_users=a.users, n_videos=a.videos, days=a.days, seed=a.seed))
        print(f"wrote {a.out}")
    elif a.cmd == "train":
        from .pipeline import TrainConfig, run
        m = run(TrainConfig(data=a.data, out=a.out, tt_epochs=a.tt_epochs, rank_epochs=a.rank_epochs,
                            rank_max_rows=a.rank_max_rows, eval_users=a.eval_users, content_emb=a.content_emb))
        print(json.dumps(m, indent=2))
    elif a.cmd == "bench":
        import numpy as np

        from .serve.app import Recommender
        rec = Recommender(a.artifacts, a.candidates)
        users = list(rec.users)
        rng = np.random.default_rng(0)
        t0 = time.time()
        for u in rng.choice(users, a.requests):
            rec.recommend(int(u), 10)
        wall = time.time() - t0
        out = {"requests": a.requests, "candidates_per_request": a.candidates,
               "throughput_rps_single_thread": round(a.requests / wall, 1), **rec.latency_summary()}
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
