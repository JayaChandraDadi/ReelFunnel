"""Two-tower retrieval model in PyTorch.

User tower: user id, activity level and the mean-pooled embeddings of recent
positives. Item tower: video id, author, tag and duration bucket, plus an optional
content embedding (CoView). Both end in an MLP and an L2-normalised 64-d vector,
so retrieval is a dot product and can be served from an ANN index.

Training uses in-batch negatives with a sampled-softmax loss and logQ correction:
popular items appear as negatives more often, so their logits are discounted by
log(sampling probability). Accidental hits (the same video twice in a batch) are
masked. Temperature scaling, warm-up and cosine decay are standard for embedding-based retrieval
(Yi et al. 2019, "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations").
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TowerConfig:
    dim: int = 64
    emb: int = 48
    hidden: int = 256
    temperature: float = 0.05
    content_dim: int = 0


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.LayerNorm(hidden),
                         nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, d_out))


class TwoTower(nn.Module):
    def __init__(self, sizes: dict, n_active: int, item_table: dict, cfg: TowerConfig = TowerConfig(),
                 content: np.ndarray | None = None):
        super().__init__()
        self.cfg = cfg
        e = cfg.emb
        self.video = nn.Embedding(sizes["videos"], e, padding_idx=0)
        self.author = nn.Embedding(sizes["authors"], e // 2, padding_idx=0)
        self.tag = nn.Embedding(sizes["tags"], e // 4, padding_idx=0)
        self.dur = nn.Embedding(16, 8)
        self.user = nn.Embedding(sizes["users"], e, padding_idx=0)
        self.active = nn.Embedding(n_active + 1, 8)
        for name in ("author", "tag", "dur"):
            self.register_buffer(f"t_{name}", torch.as_tensor(item_table[name], dtype=torch.long), persistent=True)
        item_in = e + e // 2 + e // 4 + 8
        if cfg.content_dim:
            self.register_buffer("t_content", torch.as_tensor(content, dtype=torch.float32))
            self.content_proj = nn.Linear(cfg.content_dim, 32)
            item_in += 32
        self.item_mlp = _mlp(item_in, cfg.hidden, cfg.dim)
        self.user_mlp = _mlp(e + e + 8, cfg.hidden, cfg.dim)

    def item_vec(self, vid: torch.Tensor) -> torch.Tensor:
        parts = [self.video(vid), self.author(self.t_author[vid]), self.tag(self.t_tag[vid]), self.dur(self.t_dur[vid])]
        if self.cfg.content_dim:
            parts.append(self.content_proj(self.t_content[vid]))
        return F.normalize(self.item_mlp(torch.cat(parts, -1)), dim=-1)

    def user_vec(self, uid: torch.Tensor, active: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        mask = (hist > 0).float().unsqueeze(-1)
        pooled = (self.video(hist) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        x = torch.cat([self.user(uid), pooled, self.active(active)], -1)
        return F.normalize(self.user_mlp(x), dim=-1)

    def loss(self, uid, active, hist, vid, log_q: torch.Tensor) -> torch.Tensor:
        u = self.user_vec(uid, active, hist)
        v = self.item_vec(vid)
        logits = u @ v.T / self.cfg.temperature - log_q[vid].unsqueeze(0)
        same = vid.unsqueeze(0) == vid.unsqueeze(1)
        logits = logits.masked_fill(same & ~torch.eye(len(vid), dtype=torch.bool, device=vid.device), -1e9)
        labels = torch.arange(len(vid), device=vid.device)
        return F.cross_entropy(logits, labels)


def train_two_tower(model: TwoTower, batches, log_q: np.ndarray, epochs: int = 3, lr: float = 2e-3,
                    warmup_frac: float = 0.05, device: str = "cpu", log_every: int = 200):
    """`batches` is a callable returning an iterable of (uid, active, hist, vid) numpy batches."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lq = torch.as_tensor(log_q, dtype=torch.float32, device=device)
    steps_per_epoch = sum(1 for _ in batches())
    total = steps_per_epoch * epochs
    warm = max(1, int(total * warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    history, step = [], 0
    for ep in range(epochs):
        for uid, act, hist, vid in batches():
            t = [torch.as_tensor(x, device=device) for x in (uid, act, hist, vid)]
            loss = model.loss(*t, log_q=lq)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % log_every == 0:
                history.append((step, float(loss)))
                print(f"  epoch {ep + 1} step {step}/{total} loss {float(loss):.4f}", flush=True)
    model.eval()
    return history


@torch.no_grad()
def embed_items(model: TwoTower, n_items: int, batch: int = 8192) -> np.ndarray:
    out = []
    for s in range(0, n_items, batch):
        out.append(model.item_vec(torch.arange(s, min(s + batch, n_items))).numpy())
    emb = np.concatenate(out).astype(np.float32)
    emb[0] = 0.0  # padding row never retrieved
    return emb


@torch.no_grad()
def embed_users(model: TwoTower, uid: np.ndarray, active: np.ndarray, hist: np.ndarray, batch: int = 8192) -> np.ndarray:
    out = []
    for s in range(0, len(uid), batch):
        sl = slice(s, s + batch)
        out.append(model.user_vec(torch.as_tensor(uid[sl]), torch.as_tensor(active[sl]),
                                  torch.as_tensor(hist[sl])).numpy())
    return np.concatenate(out).astype(np.float32)
