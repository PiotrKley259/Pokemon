import numpy as np
import pandas as pd
import pytest

from src.features.embeddings import EmbeddingResidualiser, rarity_probe_accuracy
from src.features.neighbors import build_neighbor_features, nearest_neighbors


@pytest.fixture
def corpus():
    """8 corpus cards in 2D embedding space, two sets."""
    emb = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9],
                    [1, 0.05], [0.05, 1], [0.7, 0.7], [0.6, 0.8]], float)
    sets = pd.Series(["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"])
    returns = np.array([0.5, 0.4, -0.5, -0.4, 0.45, -0.45, 0.0, 0.1])
    premiums = np.array([1.0, 0.8, -1.0, -0.8, 0.9, -0.9, 0.0, 0.1])
    return emb, sets, returns, premiums


def _features(query_emb, query_sets, corpus, ks=(2,)):
    emb, sets, returns, premiums = corpus
    return build_neighbor_features(
        query_emb, pd.Series(query_sets), emb, sets, returns, premiums,
        ks=list(ks), top_decile_pct=0.25, dispersion_k=3)


def test_same_set_neighbours_are_excluded(corpus):
    # Query identical to corpus card 0 but tagged set s1: its perfect matches
    # (0, 1) are excluded, so the neighbour mean must come from s2's
    # x-direction cards (returns 0.45, 0.0, 0.1 region), never 0.5.
    feats = _features(np.array([[1.0, 0.0]]), ["s1"], corpus, ks=(2,))
    assert feats.loc[0, "top2_sim_mean_return"] < 0.5
    assert feats.loc[0, "top2_sim_mean_return"] == pytest.approx(0.45, abs=0.25)

    # Same query from an unrelated set may use every corpus card.
    feats_free = _features(np.array([[1.0, 0.0]]), ["zzz"], corpus, ks=(2,))
    assert feats_free.loc[0, "top2_sim_mean_return"] > \
        feats.loc[0, "top2_sim_mean_return"]


def test_neighbour_means_track_direction(corpus):
    feats = _features(np.array([[1.0, 0.0], [0.0, 1.0]]), ["zzz", "zzz"],
                      corpus, ks=(2,))
    # x-direction queries sit near positive-return cards, y-direction near
    # negative-return cards.
    assert feats.loc[0, "top2_sim_mean_return"] > 0.3
    assert feats.loc[1, "top2_sim_mean_return"] < -0.3
    assert feats.loc[0, "top2_sim_mean_premium"] > 0
    assert feats.loc[1, "top2_sim_mean_premium"] < 0


def test_novelty_score(corpus):
    exact = _features(np.array([[1.0, 0.0]]), ["zzz"], corpus)
    far = _features(np.array([[-1.0, -1.0]]), ["zzz"], corpus)
    assert exact.loc[0, "novelty_score"] == pytest.approx(0.0, abs=1e-9)
    assert far.loc[0, "novelty_score"] > 1.0  # negative cosine -> novelty > 1


def test_dispersion_reflects_neighbour_disagreement(corpus):
    # A diagonal query sits between the +return and -return clusters, so its
    # top-3 neighbours disagree more than an x-aligned query's do.
    mixed = _features(np.array([[0.7, 0.7]]), ["zzz"], corpus)
    pure = _features(np.array([[1.0, 0.0]]), ["zzz"], corpus)
    assert mixed.loc[0, "neighbour_dispersion"] > \
        pure.loc[0, "neighbour_dispersion"]


def test_similarity_to_top_decile(corpus):
    # Top 25% by premium = cards 0 and 4 (both x-direction), so an x query
    # scores higher than a y query.
    fx = _features(np.array([[1.0, 0.0]]), ["zzz"], corpus)
    fy = _features(np.array([[0.0, 1.0]]), ["zzz"], corpus)
    assert fx.loc[0, "similarity_to_top_decile"] > \
        fy.loc[0, "similarity_to_top_decile"]


def test_nearest_neighbors_helper(corpus):
    emb, sets, _, _ = corpus
    idx, sim = nearest_neighbors(np.array([1.0, 0.0]), "s1", emb, sets, n=3)
    assert len(idx) == 3
    assert all(sets.iloc[i] != "s1" for i in idx)   # cross-set only
    assert idx[0] == 4                              # closest s2 card
    assert np.all(np.diff(sim) <= 1e-12)            # sorted descending


def test_premium_only_mode_without_returns(corpus):
    """corpus_returns=None (single-snapshot history): return features are
    absent, premium features unchanged, dispersion falls back to premiums."""
    emb, sets, returns, premiums = corpus
    feats = build_neighbor_features(
        np.array([[1.0, 0.0]]), pd.Series(["zzz"]), emb, sets,
        None, premiums, ks=[2], top_decile_pct=0.25, dispersion_k=3)
    assert not any(c.endswith("_mean_return") for c in feats.columns)

    with_ret = build_neighbor_features(
        np.array([[1.0, 0.0]]), pd.Series(["zzz"]), emb, sets,
        returns, premiums, ks=[2], top_decile_pct=0.25, dispersion_k=3)
    assert feats.loc[0, "top2_sim_mean_premium"] == \
        with_ret.loc[0, "top2_sim_mean_premium"]
    assert feats.loc[0, "novelty_score"] == with_ret.loc[0, "novelty_score"]
    assert np.isfinite(feats.loc[0, "neighbour_dispersion"])


def test_residualiser_removes_rarity_signal():
    """Embeddings built to encode rarity directly: after residualising on
    rarity, a linear probe should drop to near-chance accuracy."""
    rng = np.random.default_rng(0)
    n = 300
    rarity = pd.Series(rng.choice(["Common", "Rare", "Secret"], n))
    offset = rarity.map({"Common": -3.0, "Rare": 0.0, "Secret": 3.0}).to_numpy()
    emb = rng.normal(size=(n, 8))
    emb[:, 0] += offset
    emb[:, 1] += offset
    meta = pd.DataFrame({"rarity": rarity, "variant": "holofoil",
                         "artist": "a", "series": "S"})

    acc_raw = rarity_probe_accuracy(emb, rarity, seed=0)
    resid = EmbeddingResidualiser().fit_transform(emb, meta)
    acc_res = rarity_probe_accuracy(resid, rarity, seed=0)
    assert acc_raw > 0.85
    assert acc_res < acc_raw - 0.3
    assert acc_res < 0.55
