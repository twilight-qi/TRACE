import torch
from typing import List, Sequence, Tuple, Dict


def _recall_at_k(targets: torch.Tensor, preds: torch.Tensor, k: int) -> torch.Tensor:
    preds_k = preds[:, :k]
    matches = (preds_k == targets.unsqueeze(1)).any(dim=1).float()
    return matches.mean()


def _ndcg_at_k(targets: torch.Tensor, preds: torch.Tensor, k: int) -> torch.Tensor:
    preds_k = preds[:, :k]
    # match mask (N, k)
    mask = preds_k == targets.unsqueeze(1)
    found = mask.any(dim=1)
    # position: first True index per row (0-based)
    # use argmax on int mask, but need to guard missing
    idx = mask.float().argmax(dim=1)  # 0 if none
    idx = idx + 1  # make 1-based
    idx = idx.float()
    idx[~found] = 0.0
    dcg = torch.zeros_like(idx)
    found_idx = found.nonzero(as_tuple=False).squeeze(1)
    if found_idx.numel() > 0:
        pos = idx[found_idx]
        dcg_vals = 1.0 / torch.log2(pos + 1.0)
        dcg[found_idx] = dcg_vals
    return dcg.mean()


def _map_at_k(targets: torch.Tensor, preds: torch.Tensor, k: int) -> torch.Tensor:
    preds_k = preds[:, :k]
    mask = preds_k == targets.unsqueeze(1)
    found = mask.any(dim=1)
    idx = mask.float().argmax(dim=1) + 1.0
    idx = idx.float()
    idx[~found] = 0.0
    map_vals = torch.zeros_like(idx)
    if found.any():
        map_vals[found] = 1.0 / idx[found]
    return map_vals.mean()


def _mrr(targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    mask = preds == targets.unsqueeze(1)
    found = mask.any(dim=1)
    idx = mask.float().argmax(dim=1) + 1.0
    idx = idx.float()
    idx[~found] = 0.0
    mrr_vals = torch.zeros_like(idx)
    if found.any():
        mrr_vals[found] = 1.0 / idx[found]
    return mrr_vals.mean()


class RankingMetrics:
    """A simple CPU-buffered ranking metrics accumulator.

    - Call `update(targets, preds)` each batch. `preds` is expected to be
      long tensor of shape (B, R) containing ranked item indices, and
      `targets` is shape (B,) containing the ground-truth index per sample.
    - `update` immediately moves tensors to CPU and detaches to avoid GPU memory growth.
    - `compute()` returns a dict of metrics (recall@k, ndcg@k, map@k, mrr).
    - `reset()` clears internal buffers.

    This mimics a torchmetrics-like interface so you can swap it with minimal code changes.
    """

    def __init__(self, ks: Sequence[int] = (1, 5, 10)) -> None:
        self.ks = tuple(sorted(int(k) for k in ks))
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    def update(self, targets: torch.Tensor, preds: torch.Tensor) -> None:
        """Buffer a batch. Moves tensors to CPU and detaches to free GPU memory."""
        if targets is None or preds is None:
            return
        t = targets.detach().cpu().clone()
        p = preds.detach().cpu().clone()
        # ensure shapes
        if t.dim() == 2 and t.size(1) == 1:
            t = t.squeeze(1)
        self._targets.append(t)
        self._preds.append(p)

    def compute(self, prefix: str = "") -> Dict[str, float]:
        """Compute metrics and return a dict with optionally prefixed keys.

        Keys produced:
          - `{prefix}ACC@{k}` <-- recall@k
          - `{prefix}NDCG@{k}`
          - `{prefix}MAP@{k}`
          - `{prefix}MRR`
        """
        if not self._targets:
            results = {}
            for k in self.ks:
                results[f"{prefix}ACC@{k}"] = 0.0
                results[f"{prefix}NDCG@{k}"] = 0.0
                results[f"{prefix}MAP@{k}"] = 0.0
            results[f"{prefix}MRR"] = 0.0
            return results

        targets = torch.cat(self._targets, dim=0)
        preds = torch.cat(self._preds, dim=0)
        results: Dict[str, float] = {}
        for k in self.ks:
            results[f"{prefix}ACC@{k}"] = float(_recall_at_k(targets, preds, k).item())
            results[f"{prefix}NDCG@{k}"] = float(_ndcg_at_k(targets, preds, k).item())
            results[f"{prefix}MAP@{k}"] = float(_map_at_k(targets, preds, k).item())
        results[f"{prefix}MRR"] = float(_mrr(targets, preds).item())
        return results

    def reset(self) -> None:
        self._preds.clear()
        self._targets.clear()


__all__ = ["RankingMetrics"]
