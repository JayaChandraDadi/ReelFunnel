# ReelFunnel

A two-stage short-video recommender: two-tower embedding retrieval over an HNSW index, a DCN v2 heavy ranker, and a FastAPI service that runs the whole request path in under 10 ms. Built on the KuaiRand short-video dataset (or a synthetic copy with the same schema).

```
user ──► user embedding (offline, from history) ──► HNSW top-500 ──► counter features ──► DCN v2 ──► top-k
          two-tower, PyTorch                         FAISS              (WindowForge names)   TensorFlow/Keras
```

## What is in it

**Retrieval (PyTorch).** User tower: user id, activity level, and mean-pooled embeddings of the last 20 positive videos. Item tower: video id, author, tag, duration bucket, and optionally a CoView content embedding. Both end in an MLP and a 64-d L2-normalised vector. Trained with in-batch negatives, sampled-softmax loss with logQ popularity correction, accidental-hit masking, temperature 0.05, warm-up and cosine decay.

**ANN serving (FAISS).** HNSW (M=32, efConstruction=200) over item vectors with inner product, checked against an exact index on every run.

**Ranking (TensorFlow/Keras).** DCN v2: embeddings for user, video, author, tag, duration, activity and hour; 20 dense features (WindowForge-style windowed counters, smoothed rates, video age and the two-tower similarity) through log1p and a learned normalisation layer; 3 full-rank cross layers in parallel with a 256-128 MLP. Label: long view.

**Serving (FastAPI).** `/recommend` looks up the precomputed user vector, queries HNSW, hydrates features from in-memory tables (what the online store holds), scores all candidates in one traced TensorFlow call, sorts. `/metrics` reports p50/p95 per stage.

**Evaluation, done the way production evaluates.**
- Everything is split by time. User histories and counter features only see data from before each event.
- Retrieval: Recall@K and NDCG over the full corpus for test-period engagements, excluding videos the user already engaged with, against popularity and item-CF baselines.
- Ranking: AUC and GAUC on the test period's randomly exposed impressions (KuaiRand's unbiased slice), plus per-user NDCG@10 when re-ordering each user's actual test impressions.

## Results (synthetic KuaiRand-schema data: 945K impressions, 1,500 users, 4,000 videos, 21 days)

Retrieval, 1,500 test users, 500 candidates:

| Source | Recall@50 | Recall@200 | Recall@500 | NDCG@50 |
| --- | --- | --- | --- | --- |
| Popularity | 0.051 | 0.149 | 0.299 | 0.045 |
| Item-CF | 0.085 | 0.196 | 0.349 | 0.076 |
| **Two-tower + HNSW** | **0.174** | **0.400** | **0.608** | **0.150** |

HNSW returns 100% of the exact top-100 at 0.38 ms per query.

Ranking, test period:

| Scorer | AUC (random exposure) | Per-user NDCG@10 on real impressions |
| --- | --- | --- |
| Random | | 0.486 |
| Video long-view rate | | 0.819 |
| Two-tower similarity | 0.778 | 0.880 |
| **DCN v2 ranker** | **0.851** (GAUC 0.860) | **0.933** (+6.1%) |

Serving, 500 candidates per request, one thread:

| Stage | p50 | p95 |
| --- | --- | --- |
| Retrieve (HNSW) | 1.1 ms | 1.4 ms |
| Feature hydration | 2.0 ms | 3.3 ms |
| Rank (DCN v2) | 2.8 ms | 4.0 ms |
| **Total** | **6.0 ms** | **8.2 ms** |

About 158 requests per second on one thread. Full training run: about 2.5 minutes on 2 CPU cores.

These are synthetic-data numbers that prove the pipeline end to end. Train on real KuaiRand for numbers you can put on a resume; `artifacts/metrics.json` records every figure above.

## Run it

```bash
pip install -r requirements.txt

# Data: synthetic, or KuaiRand-Pure from https://kuairand.com unzipped into data/kuairand
python -m reelfunnel.cli synth --out data/synth --users 1500 --videos 4000

python -m reelfunnel.cli train --data data/synth --out artifacts      # trains, evaluates, exports
python -m reelfunnel.cli bench --artifacts artifacts --requests 500    # latency per stage

uvicorn reelfunnel.serve.app:app --port 8000
curl 'localhost:8000/recommend?user_id=42&k=10'

pytest -q
```

With CoView embeddings in the item tower: `--content-emb ../coview/artifacts/item_embeddings.npz` (the ids must match the dataset's video ids).

On Apple silicon, `pip install tensorflow` works natively; PyTorch uses CPU here because the towers are small.

## Layout

```
reelfunnel/data.py                 KuaiRand loader, vocab, time-correct history matrix
reelfunnel/retrieval/two_tower.py  towers, logQ-corrected in-batch softmax, training loop
reelfunnel/retrieval/index.py      FAISS HNSW + exact index, ANN recall
reelfunnel/retrieval/baselines.py  popularity, item-CF
reelfunnel/ranking/features.py     point-in-time windowed counters (WindowForge semantics)
reelfunnel/ranking/dcn.py          DCN v2 in Keras, traced predictor
reelfunnel/pipeline.py             train → evaluate → export
reelfunnel/serve/app.py            FastAPI service
tests/                             history leakage, counters, two-tower, HNSW, API
```

## Design notes

- **Why judge the ranker on impressions, not on retrieved candidates.** Logged engagements exist only for videos the old policy chose to show. Scoring a ranker by whether it surfaces those videos rewards imitating the old policy's exposure, not predicting engagement. Per-user ranking of real impressions, plus AUC on randomly exposed traffic, measures what the ranker is for.
- **User embeddings are computed offline per day** from history before the cutoff, the same way serving loads them. The ranker's two-tower-similarity feature uses the same daily vectors, so there is no train/serve skew in that feature.
- **logQ correction** stops in-batch negatives from over-penalising popular videos, which otherwise drags recall on popular items down.
- **The HNSW search beam widens to k automatically**: an HNSW search cannot return more good neighbours than its `efSearch`.
