"""Cold-start scorer for newly released sets: art + static metadata only.

Pipeline (see README "Cold-start visual model"):
1. decompose log price into character_equity + rarity_set_baseline +
   art_premium (fitted on training sets only);
2. load frozen image embeddings, PCA-reduce (fitted on train), report the
   rarity-probe accuracy, then residualise embeddings on
   (rarity, variant, artist, era) to isolate "style";
3. build visual-neighbour features against the training corpus (same-set
   neighbours excluded);
4. hold out ENTIRE recent sets (released after T) and report a rank-IC
   ablation on realised 90d forward returns, against the character-equity-only
   baseline ("buy Charizard because Charizard");
5. save an art_premium model bundle for the dashboard's New Set Scanner.

Usage:
    python -m src.models.train_coldstart
    python -m src.models.train_coldstart --compare-crops   # art vs full crop
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor

from src.config import load_config, resolve_path, set_seeds
from src.features.build_features import (
    add_forward_return_target,
    add_static_features,
)
from src.features.embeddings import (
    EmbeddingResidualiser,
    fit_pca,
    load_embeddings,
    rarity_probe_accuracy,
)
from src.features.neighbors import build_neighbor_features
from src.features.price_decomposition import PriceDecomposer

log = logging.getLogger("coldstart")

TABULAR_CAT = ["rarity", "series", "artist", "supertype", "subtype",
               "variant", "energy_type"]
TABULAR_NUM = ["hp", "retreat_cost", "attack_count", "card_number",
               "set_size", "card_number_ratio", "pokedex_number",
               "is_full_art", "is_alt_art", "is_secret", "is_promo",
               "is_starter", "is_legendary", "is_first_gen"]

# Fixed, mildly regularised params: the ablation compares FEATURE SETS, so
# every cell gets the same model capacity.
XGB_PARAMS = dict(n_estimators=600, max_depth=5, learning_rate=0.05,
                  min_child_weight=5.0, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, objective="reg:squarederror",
                  tree_method="hist", enable_categorical=True, n_jobs=-1)


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------
def build_card_table(cfg: dict, df_raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (card_id, variant): the FIRST snapshot that has a valid
    90d forward return (release-time scoring simulation) plus, as fallback
    for cards too young to have one, the first priced snapshot with a NaN
    target (usable for the premium model, excluded from return evaluation)."""
    h = cfg["coldstart"]["horizon_days"]
    tol = cfg["targets"]["horizon_tolerance_days"]
    df = add_static_features(df_raw)
    df = add_forward_return_target(df, h, tol)
    df = df[df["price_market"] > 0].copy()
    df["log_price"] = np.log(df["price_market"])
    df["has_target"] = df[f"fwd_ret_{h}d"].notna()
    first = (df.sort_values(["has_target", "snapshot_date"],
                            ascending=[False, True])
             .groupby(["card_id", "variant"], sort=False)
             .head(1))
    return first.rename(columns={f"fwd_ret_{h}d": "fwd_return"}) \
                .reset_index(drop=True)


def split_by_set_release(cfg: dict, cards: pd.DataFrame
                         ) -> tuple[pd.Series, pd.Timestamp]:
    """Boolean holdout mask: entire sets released after T are held out."""
    holdout_after = cfg["coldstart"]["holdout_after"]
    releases = (cards.dropna(subset=["release_date"])
                .groupby("set_id")["release_date"].first().sort_values())
    if holdout_after:
        cutoff = pd.Timestamp(holdout_after)
    else:
        cutoff = releases.iloc[int(np.floor(0.8 * (len(releases) - 1)))]
    holdout_sets = set(releases[releases > cutoff].index)
    if not holdout_sets:
        raise RuntimeError(f"No sets released after {cutoff.date()} to hold out")
    log.info("Holdout: %d sets released after %s (train: %d sets)",
             len(holdout_sets), cutoff.date(), len(releases) - len(holdout_sets))
    return cards["set_id"].isin(holdout_sets), cutoff


def attach_embeddings(cfg: dict, cards: pd.DataFrame, crop: str, encoder: str
                      ) -> tuple[pd.DataFrame, np.ndarray]:
    emb, ids = load_embeddings(cfg, crop, encoder)
    idx = {cid: i for i, cid in enumerate(ids)}
    keep = cards["card_id"].map(idx).notna()
    n_before = len(cards)
    cards = cards[keep].reset_index(drop=True)
    rows = cards["card_id"].map(idx).astype(int).to_numpy()
    log.info("Embeddings (%s/%s): %d of %d cards covered",
             encoder, crop, len(cards), n_before)
    return cards, emb[rows]


# --------------------------------------------------------------------------
# Model fitting
# --------------------------------------------------------------------------
def _cast_cats(train: pd.DataFrame, test: pd.DataFrame, cats: list[str]):
    train, test = train.copy(), test.copy()
    for c in cats:
        cat_type = pd.CategoricalDtype(train[c].dropna().unique())
        train[c] = train[c].astype(cat_type)
        test[c] = test[c].astype(cat_type)
    return train, test


def fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                cat_cols: list[str], num_cols: list[str],
                target: str, seed: int) -> np.ndarray:
    """Fit XGB on train (early stopping on a set-grouped inner split),
    predict test."""
    rng = np.random.default_rng(seed)
    sets = train["set_id"].unique()
    val_sets = set(rng.choice(sets, size=max(1, len(sets) // 6), replace=False))
    inner_val = train["set_id"].isin(val_sets)
    cols = cat_cols + num_cols
    tr, va = train[~inner_val], train[inner_val]
    tr_x, va_x = _cast_cats(tr[cols], va[cols], cat_cols)
    _, te_x = _cast_cats(tr[cols], test[cols], cat_cols)
    model = XGBRegressor(random_state=seed, early_stopping_rounds=50,
                         **XGB_PARAMS)
    model.fit(tr_x, tr[target], eval_set=[(va_x, va[target])], verbose=False)
    return model.predict(te_x)


def rank_ic(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 3:
        return float("nan")
    ic, _ = spearmanr(y[mask], pred[mask])
    return float(ic)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.isfinite(pred)
    err = pred[mask] - y[mask]
    return {"rank_ic": rank_ic(y, pred),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "n": int(mask.sum())}


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------
def run_experiment(cfg: dict, crop: str) -> dict:
    seed = cfg["seed"]
    cs = cfg["coldstart"]
    encoder = cs["encoder"]

    df_raw = pd.read_parquet(resolve_path(cfg, "dataset"))
    cards = build_card_table(cfg, df_raw)
    cards, emb = attach_embeddings(cfg, cards, crop, encoder)
    holdout_mask, cutoff = split_by_set_release(cfg, cards)

    train = cards[~holdout_mask].reset_index(drop=True)
    test = cards[holdout_mask].reset_index(drop=True)
    emb_train, emb_test = emb[(~holdout_mask).to_numpy()], emb[holdout_mask.to_numpy()]

    # 1. price decomposition, fitted on train sets only
    decomp = PriceDecomposer(shrinkage_k=cs["shrinkage_k"]).fit(train)
    for part in (train, test):
        comp = decomp.transform(part)
        for c in comp.columns:
            part[c] = comp[c].to_numpy()

    # 2. PCA + leakage probes (train-fitted)
    pca, var_kept = fit_pca(emb_train, cs["pca_dims"], seed)
    z_train, z_test = pca.transform(emb_train), pca.transform(emb_test)
    probe_raw = rarity_probe_accuracy(z_train, train["rarity"], seed)

    train_meta = train.assign(era=train["series"])
    test_meta = test.assign(era=test["series"])
    resid = EmbeddingResidualiser().fit(z_train, train_meta)
    s_train = resid.transform(z_train, train_meta)
    s_test = resid.transform(z_test, test_meta)
    probe_resid = rarity_probe_accuracy(s_train, train["rarity"], seed)
    log.info("[%s] PCA variance kept %.1f%% | rarity probe raw %.3f -> "
             "residualised %.3f", crop, 100 * var_kept, probe_raw, probe_resid)

    # Mode: with enough realised forward returns, evaluate on them; with a
    # short price history (e.g. a single snapshot) fall back to the
    # cross-sectional art premium as the held-out-set target. The premium
    # needs no future prices, so the cold-start question - "can the art
    # predict which cards of an unseen set carry a premium?" - is answerable
    # from day one.
    have_returns = (train["fwd_return"].notna().sum() >= 50
                    and test["fwd_return"].notna().sum() >= 20)
    mode = "forward_return" if have_returns else "art_premium_cross_sectional"
    if not have_returns:
        log.info("[%s] too few realised %dd returns - running in premium-only "
                 "mode (target: art_premium of held-out sets)",
                 crop, cs["horizon_days"])

    # 3. neighbour features vs the TRAIN corpus (labelled rows only)
    corpus_ok = (train["fwd_return"].notna() if have_returns
                 else train["art_premium"].notna()).to_numpy()
    nb_kwargs = dict(
        corpus_emb=emb_train[corpus_ok],
        corpus_set_ids=train.loc[corpus_ok, "set_id"],
        corpus_returns=(train.loc[corpus_ok, "fwd_return"].to_numpy()
                        if have_returns else None),
        corpus_premiums=train.loc[corpus_ok, "art_premium"].to_numpy(),
        ks=list(cs["knn_ks"]), top_decile_pct=cs["top_decile_pct"],
    )
    nb_train = build_neighbor_features(emb_train, train["set_id"], **nb_kwargs)
    nb_test = build_neighbor_features(emb_test, test["set_id"], **nb_kwargs)
    nb_cols = list(nb_train.columns)
    for col in nb_cols:
        train[col] = nb_train[col].to_numpy()
        test[col] = nb_test[col].to_numpy()

    emb_cols_raw, emb_cols_style = [], []
    for j in range(z_train.shape[1]):
        train[f"emb_{j}"] = z_train[:, j]
        test[f"emb_{j}"] = z_test[:, j]
        train[f"style_{j}"] = s_train[:, j]
        test[f"style_{j}"] = s_test[:, j]
        emb_cols_raw.append(f"emb_{j}")
        emb_cols_style.append(f"style_{j}")

    # 4. ablation on the held-out sets: realised returns when available,
    # otherwise today's cross-sectional art premium
    target = "fwd_return" if have_returns else "art_premium"
    tr_lab = train[train[target].notna()].reset_index(drop=True)
    te_lab = test[test[target].notna()].reset_index(drop=True)
    equity_cols = ["character_equity", "rarity_set_baseline"]
    ablations = {
        "tabular_only": (TABULAR_CAT, TABULAR_NUM),
        "tabular+equity": (TABULAR_CAT, TABULAR_NUM + equity_cols),
        "tabular+equity+raw_emb": (TABULAR_CAT,
                                   TABULAR_NUM + equity_cols + emb_cols_raw),
        "tabular+equity+style": (TABULAR_CAT,
                                 TABULAR_NUM + equity_cols + emb_cols_style),
        "tabular+equity+style+neighbours": (
            TABULAR_CAT, TABULAR_NUM + equity_cols + emb_cols_style + nb_cols),
    }
    # Baseline: character equity alone ("buy Charizard because Charizard").
    # In premium mode it is a sanity floor: the premium is the residual left
    # AFTER removing equity, so equity should carry little rank information -
    # any ablation cell has to clear it comfortably.
    results = {"character_equity_baseline":
               _metrics(te_lab[target].to_numpy(),
                        te_lab["character_equity"].to_numpy())}
    for name, (cats, nums) in ablations.items():
        pred = fit_predict(tr_lab, te_lab, cats, nums, target, seed)
        results[name] = _metrics(te_lab[target].to_numpy(), pred)
        log.info("[%s] %-32s rank IC %+.3f (n=%d)", crop, name,
                 results[name]["rank_ic"], results[name]["n"])

    return {
        "crop": crop, "encoder": encoder,
        "mode": mode, "ablation_target": target,
        "holdout_cutoff": str(cutoff.date()),
        "n_train_cards": len(train), "n_holdout_cards": len(test),
        "pca_variance_retained": var_kept,
        "rarity_probe_accuracy_raw": probe_raw,
        "rarity_probe_accuracy_residualised": probe_resid,
        "ablation": results,
        "_artifacts": {"train": train, "test": test, "decomp": decomp,
                       "pca": pca, "resid": resid,
                       "emb_train": emb_train, "corpus_ok": corpus_ok,
                       "nb_cols": nb_cols, "style_cols": emb_cols_style},
    }


def save_premium_model(cfg: dict, res: dict, crop: str) -> None:
    """Fit the art_premium model on ALL training cards and bundle everything
    the dashboard needs to score a genuinely new set."""
    seed = cfg["seed"]
    a = res["_artifacts"]
    train = a["train"]
    cats, nums = TABULAR_CAT, TABULAR_NUM + a["style_cols"] + a["nb_cols"]

    rng = np.random.default_rng(seed)
    sets = train["set_id"].unique()
    val_sets = set(rng.choice(sets, size=max(1, len(sets) // 6), replace=False))
    inner_val = train["set_id"].isin(val_sets)
    cols = cats + nums
    tr, va = train[~inner_val], train[inner_val]
    tr_x, va_x = _cast_cats(tr[cols], va[cols], cats)
    model = XGBRegressor(random_state=seed, early_stopping_rounds=50,
                         **XGB_PARAMS)
    model.fit(tr_x, tr["art_premium"], eval_set=[(va_x, va["art_premium"])],
              verbose=False)

    corpus_ok = a["corpus_ok"]
    bundle = {
        "model": model, "cat_cols": cats, "num_cols": nums,
        "cat_dtypes": {c: pd.CategoricalDtype(tr[c].dropna().unique())
                       for c in cats},
        "decomposer": a["decomp"], "pca": a["pca"], "residualiser": a["resid"],
        "crop": crop, "encoder": res["encoder"], "seed": seed,
        "mode": res["mode"],
        "knn_ks": list(cfg["coldstart"]["knn_ks"]),
        "top_decile_pct": cfg["coldstart"]["top_decile_pct"],
        "corpus": {
            "emb": a["emb_train"][corpus_ok],
            "card_id": train.loc[corpus_ok, "card_id"].to_numpy(),
            "set_id": train.loc[corpus_ok, "set_id"].to_numpy(),
            "name": train.loc[corpus_ok, "name"].to_numpy(),
            "image_url": train.loc[corpus_ok, "image_url"].to_numpy(),
            "fwd_return": train.loc[corpus_ok, "fwd_return"].to_numpy(),
            "art_premium": train.loc[corpus_ok, "art_premium"].to_numpy(),
        },
    }
    out = resolve_path(cfg, "models_dir") / "coldstart_premium.joblib"
    joblib.dump(bundle, out)
    log.info("Cold-start premium bundle saved to %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-crops", action="store_true",
                        help="Run the ablation for both art and full crops")
    args = parser.parse_args()
    cfg = load_config()
    set_seeds(cfg["seed"])
    resolve_path(cfg, "models_dir").mkdir(parents=True, exist_ok=True)
    resolve_path(cfg, "reports_dir").mkdir(parents=True, exist_ok=True)

    crops = (["art", "full"] if args.compare_crops
             else [cfg["coldstart"]["crop"]])
    all_results = {}
    for crop in crops:
        res = run_experiment(cfg, crop)
        if crop == crops[0]:
            save_premium_model(cfg, res, crop)
        res.pop("_artifacts")
        all_results[crop] = res

    out = resolve_path(cfg, "reports_dir") / "coldstart_metrics.json"
    out.write_text(json.dumps(all_results, indent=2))
    log.info("Cold-start metrics written to %s", out)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
