"""Visual-similarity features against a historical corpus.

For each query card we look up its nearest visual neighbours by cosine
similarity, ALWAYS excluding cards from the query's own set - cards in one
set share artists, layout and print style, so same-set neighbours would leak
set identity into a model that is supposed to generalise to unseen sets.

Features per query card:
- top{k}_sim_mean_return   : similarity-weighted mean 90d forward log return
                             of the k nearest neighbours (k = 10, 25, 50);
- top{k}_sim_mean_premium  : same, for the art_premium residual;
- similarity_to_top_decile : mean cosine similarity to the corpus' top 10%
                             of cards by historical art_premium;
- neighbour_dispersion     : std of the k=25 neighbours' returns (confidence);
- novelty_score            : cosine distance to the single nearest neighbour.
                             High novelty = the model is extrapolating.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norm, 1e-12, None)


def build_neighbor_features(
    query_emb: np.ndarray,
    query_set_ids: pd.Series,
    corpus_emb: np.ndarray,
    corpus_set_ids: pd.Series,
    corpus_returns: np.ndarray | None,
    corpus_premiums: np.ndarray,
    ks: list[int] = (10, 25, 50),
    top_decile_pct: float = 0.10,
    dispersion_k: int = 25,
    chunk_size: int = 512,
) -> pd.DataFrame:
    """Compute the feature block. corpus_* must all be aligned row-wise.

    corpus_returns may be None (premium-only mode, when no snapshot history
    exists yet): the return-based features are skipped and neighbour
    dispersion is computed from art premiums instead.
    """
    have_returns = corpus_returns is not None
    q = _normalize(np.asarray(query_emb, dtype=np.float64))
    c = _normalize(np.asarray(corpus_emb, dtype=np.float64))
    corpus_sets = np.asarray(corpus_set_ids)
    query_sets = np.asarray(query_set_ids)
    if have_returns:
        corpus_returns = np.asarray(corpus_returns, dtype=np.float64)
    corpus_premiums = np.asarray(corpus_premiums, dtype=np.float64)
    dispersion_src = corpus_returns if have_returns else corpus_premiums

    n_top = max(int(np.ceil(top_decile_pct * len(c))), 1)
    top_decile_mask = corpus_premiums >= np.sort(corpus_premiums)[-n_top]

    max_k = max(max(ks), dispersion_k)
    rows = []
    for start in range(0, len(q), chunk_size):
        q_chunk = q[start:start + chunk_size]
        sims = q_chunk @ c.T                       # (chunk, corpus)
        for i in range(len(q_chunk)):
            sim = sims[i].copy()
            sim[corpus_sets == query_sets[start + i]] = -np.inf  # no same-set
            valid = np.isfinite(sim)
            n_valid = int(valid.sum())
            feat: dict = {}
            if n_valid == 0:
                for k in ks:
                    if have_returns:
                        feat[f"top{k}_sim_mean_return"] = np.nan
                    feat[f"top{k}_sim_mean_premium"] = np.nan
                feat["similarity_to_top_decile"] = np.nan
                feat["neighbour_dispersion"] = np.nan
                feat["novelty_score"] = np.nan
                rows.append(feat)
                continue

            k_eff = min(max_k, n_valid)
            top_idx = np.argpartition(-sim, k_eff - 1)[:k_eff]
            top_idx = top_idx[np.argsort(-sim[top_idx])]   # sorted by sim desc
            top_sim = sim[top_idx]

            for k in ks:
                idx = top_idx[:min(k, k_eff)]
                w = np.clip(sim[idx], 0.0, None)
                w = w / w.sum() if w.sum() > 0 else np.full(len(idx), 1 / len(idx))
                if have_returns:
                    feat[f"top{k}_sim_mean_return"] = float(
                        np.nansum(w * corpus_returns[idx]))
                feat[f"top{k}_sim_mean_premium"] = float(
                    np.nansum(w * corpus_premiums[idx]))

            decile_sims = sim[valid & top_decile_mask]
            feat["similarity_to_top_decile"] = (
                float(decile_sims.mean()) if len(decile_sims) else np.nan)

            disp_idx = top_idx[:min(dispersion_k, k_eff)]
            feat["neighbour_dispersion"] = float(
                np.nanstd(dispersion_src[disp_idx]))
            feat["novelty_score"] = float(1.0 - top_sim[0])
            rows.append(feat)

    return pd.DataFrame(rows, index=pd.RangeIndex(len(q)))


def nearest_neighbors(
    query_emb: np.ndarray,
    query_set_id: str,
    corpus_emb: np.ndarray,
    corpus_set_ids: pd.Series,
    n: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, similarities) of the n nearest cross-set neighbours
    of a single query embedding - used by the dashboard's New Set Scanner."""
    q = _normalize(np.asarray(query_emb, dtype=np.float64).reshape(1, -1))
    c = _normalize(np.asarray(corpus_emb, dtype=np.float64))
    sim = (q @ c.T)[0]
    sim[np.asarray(corpus_set_ids) == query_set_id] = -np.inf
    order = np.argsort(-sim)[:n]
    order = order[np.isfinite(sim[order])]
    return order, sim[order]
