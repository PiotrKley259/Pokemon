"""Train both models.

Task A - cross-sectional valuation: y = log(market_price) at the latest
snapshot, validated with GroupKFold grouped by set_id so a fold never sees
its own set at train time.

Task B - forward return: y = log(P_{t+h}/P_t) for h in {30, 90} days,
validated with purged + embargoed expanding-window time splits.

Every fitted transform (category mapping, one-hot, imputer, scaler) lives in
an sklearn Pipeline fitted inside each fold - nothing is fitted on the full
dataset. Optuna searches hyperparameters inside the CV loop.

Usage:
    python -m src.models.train --task a
    python -m src.models.train --task b --horizon 30
    python -m src.models.train --task all
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.config import load_config, resolve_path, set_seeds
from src.features.build_features import (
    build_task_a_frame,
    build_task_b_frame,
    feature_columns,
)
from src.models.splits import PurgedExpandingTimeSplit

log = logging.getLogger("train")


# --------------------------------------------------------------------------
# Per-fold transformers
# --------------------------------------------------------------------------
class CategoricalCaster(BaseEstimator, TransformerMixin):
    """Cast categorical columns to pandas 'category' dtype using categories
    learned from the *training fold only*. Unseen categories at transform
    time become NaN, which XGBoost handles natively."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.categories_ = {
            c: pd.Index(X[c].dropna().unique()) for c in self.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.columns:
            X[c] = pd.Categorical(X[c], categories=self.categories_[c])
        return X


def make_xgb_pipeline(cat_cols: list[str], params: dict, seed: int) -> Pipeline:
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    return Pipeline([
        ("cats", CategoricalCaster(cat_cols)),
        ("xgb", model),
    ])


def make_ridge_pipeline(cat_cols: list[str], num_cols: list[str], seed: int) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]), cat_cols),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
    ])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=1.0, random_state=seed))])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def log_space_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mdape = float(np.median(np.abs(np.exp(y_pred) - np.exp(y_true))
                            / np.exp(y_true)))
    return {
        "mae_log": float(np.mean(np.abs(err))),
        "rmse_log": float(np.sqrt(np.mean(err ** 2))),
        "mdape": mdape,
    }


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ic, _ = spearmanr(y_true, y_pred)
    return float(ic) if np.isfinite(ic) else 0.0


def decile_spread(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Backtest: mean realised return of the top vs bottom predicted decile."""
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    try:
        df["decile"] = pd.qcut(df["pred"], 10, labels=False, duplicates="drop")
    except ValueError:
        return {"top_decile_ret": float("nan"), "bottom_decile_ret": float("nan"),
                "spread": float("nan"), "by_decile": {}}
    by_dec = df.groupby("decile")["true"].mean()
    top, bottom = float(by_dec.iloc[-1]), float(by_dec.iloc[0])
    return {
        "top_decile_ret": top,
        "bottom_decile_ret": bottom,
        "spread": top - bottom,
        "by_decile": {int(k): float(v) for k, v in by_dec.items()},
    }


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def median_baseline(train: pd.DataFrame, val: pd.DataFrame, target: str) -> np.ndarray:
    """Set+rarity median of the target, fitted on the training fold only,
    with rarity-median then global-median fallbacks."""
    by_both = train.groupby(["set_id", "rarity"], observed=True)[target].median()
    by_rarity = train.groupby("rarity", observed=True)[target].median()
    global_med = train[target].median()

    keys = pd.MultiIndex.from_frame(val[["set_id", "rarity"]])
    pred = pd.Series(by_both.reindex(keys).to_numpy(), index=val.index)
    pred = pred.fillna(pd.Series(
        by_rarity.reindex(val["rarity"]).to_numpy(), index=val.index))
    return pred.fillna(global_med).to_numpy()


# --------------------------------------------------------------------------
# Optuna objective shared by both tasks
# --------------------------------------------------------------------------
def sample_params(trial: optuna.Trial, space: dict, n_estimators: int) -> dict:
    return {
        "n_estimators": n_estimators,
        "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *space["learning_rate"], log=True),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", *space["min_child_weight"], log=True),
        "subsample": trial.suggest_float("subsample", *space["subsample"]),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *space["colsample_bytree"]),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", *space["reg_lambda"], log=True),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", *space["reg_alpha"], log=True),
    }


def fit_xgb_fold(
    df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
    cat_cols: list[str], num_cols: list[str], target: str,
    params: dict, seed: int, early_stopping_rounds: int,
) -> tuple[Pipeline, np.ndarray]:
    """Fit one fold with early stopping on the validation fold."""
    cols = cat_cols + num_cols
    tr, va = df.iloc[train_idx], df.iloc[val_idx]
    pipe = make_xgb_pipeline(cat_cols, params, seed)
    caster = pipe.named_steps["cats"].fit(tr[cols])
    X_tr, X_va = caster.transform(tr[cols]), caster.transform(va[cols])
    xgb = pipe.named_steps["xgb"]
    xgb.set_params(early_stopping_rounds=early_stopping_rounds)
    xgb.fit(
        X_tr, tr[target],
        sample_weight=tr["sample_weight"],
        eval_set=[(X_va, va[target])],
        verbose=False,
    )
    return pipe, xgb.predict(X_va)


def run_cv(
    df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]],
    cat_cols: list[str], num_cols: list[str], target: str, cfg: dict,
    task_label: str,
) -> dict:
    """Optuna search inside the CV loop, then out-of-fold evaluation of
    XGBoost against the baselines with the best parameters."""
    seed = cfg["seed"]
    space = cfg["model"]["search_space"]
    n_estimators = cfg["model"]["n_estimators"]
    esr = cfg["model"]["early_stopping_rounds"]
    cols = cat_cols + num_cols

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, space, n_estimators)
        rmses = []
        for train_idx, val_idx in folds:
            _, pred = fit_xgb_fold(df, train_idx, val_idx, cat_cols, num_cols,
                                   target, params, seed, esr)
            y_va = df.iloc[val_idx][target].to_numpy()
            rmses.append(np.sqrt(np.mean((pred - y_va) ** 2)))
        return float(np.mean(rmses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=cfg["model"]["optuna_trials"],
                   show_progress_bar=False)
    best_params = sample_params(
        optuna.trial.FixedTrial(study.best_params), space, n_estimators)
    log.info("[%s] best params: %s (cv rmse %.4f)",
             task_label, study.best_params, study.best_value)

    # Out-of-fold predictions with the best params, plus baselines.
    oof = df.copy()
    oof["pred_xgb"] = np.nan
    oof["pred_ridge"] = np.nan
    oof["pred_median"] = np.nan
    oof["fold"] = -1
    best_iters = []
    for i, (train_idx, val_idx) in enumerate(folds):
        pipe, pred = fit_xgb_fold(df, train_idx, val_idx, cat_cols, num_cols,
                                  target, best_params, seed, esr)
        best_iters.append(pipe.named_steps["xgb"].best_iteration or n_estimators)
        tr, va = df.iloc[train_idx], df.iloc[val_idx]
        ridge = make_ridge_pipeline(cat_cols, num_cols, seed)
        ridge.fit(tr[cols], tr[target], ridge__sample_weight=tr["sample_weight"])
        oof.iloc[val_idx, oof.columns.get_loc("pred_xgb")] = pred
        oof.iloc[val_idx, oof.columns.get_loc("pred_ridge")] = ridge.predict(va[cols])
        oof.iloc[val_idx, oof.columns.get_loc("pred_median")] = median_baseline(
            tr, va, target)
        oof.iloc[val_idx, oof.columns.get_loc("fold")] = i

    scored = oof[oof["fold"] >= 0]
    y = scored[target].to_numpy()
    metrics = {
        "xgboost": log_space_metrics(y, scored["pred_xgb"].to_numpy()),
        "ridge": log_space_metrics(y, scored["pred_ridge"].to_numpy()),
        "set_rarity_median": log_space_metrics(y, scored["pred_median"].to_numpy()),
    }
    if task_label.startswith("task_b"):
        metrics["zero_return"] = log_space_metrics(y, np.zeros_like(y))
        for name, col in [("xgboost", "pred_xgb"), ("ridge", "pred_ridge")]:
            metrics[name]["rank_ic"] = rank_ic(y, scored[col].to_numpy())
            metrics[name]["decile_backtest"] = decile_spread(
                y, scored[col].to_numpy())
        # mdape is meaningless for returns (exp of a return is not a price gap)
        for m in metrics.values():
            m.pop("mdape", None)

    return {
        "best_params": best_params,
        "best_iteration_mean": int(np.mean(best_iters)),
        "metrics": metrics,
        "oof": scored,
    }


# --------------------------------------------------------------------------
# Task drivers
# --------------------------------------------------------------------------
def train_task_a(cfg: dict, df_raw: pd.DataFrame) -> dict:
    frame = build_task_a_frame(df_raw)
    cat_cols, num_cols = feature_columns(cfg, "a")
    target = "log_price"

    gkf = GroupKFold(n_splits=cfg["validation"]["task_a"]["n_splits"])
    folds = list(gkf.split(frame, groups=frame["set_id"]))
    result = run_cv(frame, folds, cat_cols, num_cols, target, cfg, "task_a")

    # Final model: refit on ALL data (no early-stopping fold left, so use the
    # mean best iteration found in CV).
    params = dict(result["best_params"])
    params["n_estimators"] = max(result["best_iteration_mean"], 50)
    pipe = make_xgb_pipeline(cat_cols, params, cfg["seed"])
    cols = cat_cols + num_cols
    pipe.fit(frame[cols], frame[target],
             xgb__sample_weight=frame["sample_weight"])

    models_dir = resolve_path(cfg, "models_dir")
    joblib.dump(
        {"pipeline": pipe, "cat_cols": cat_cols, "num_cols": num_cols,
         "target": target, "params": params, "seed": cfg["seed"]},
        models_dir / "task_a.joblib",
    )
    result["oof"].to_parquet(
        resolve_path(cfg, "reports_dir") / "oof_task_a.parquet", index=False)
    return {"task_a": {k: result[k] for k in
                       ("best_params", "best_iteration_mean", "metrics")}}


def train_task_b(cfg: dict, df_raw: pd.DataFrame, horizon: int) -> dict:
    frame = build_task_b_frame(df_raw, cfg, horizon)
    cat_cols, num_cols = feature_columns(cfg, "b")
    target = f"fwd_ret_{horizon}d"
    label = f"task_b_h{horizon}"

    splitter = PurgedExpandingTimeSplit(
        n_splits=cfg["validation"]["task_b"]["n_splits"],
        horizon_days=horizon,
        embargo_days=cfg["validation"]["task_b"]["embargo_days"],
        min_train_days=cfg["validation"]["task_b"]["min_train_days"],
    )
    folds = list(splitter.split(frame["snapshot_date"]))
    if not folds:
        raise RuntimeError(
            f"Not enough price history for h={horizon}: collect more "
            "snapshots over time before training Task B."
        )
    result = run_cv(frame, folds, cat_cols, num_cols, target, cfg, label)

    params = dict(result["best_params"])
    params["n_estimators"] = max(result["best_iteration_mean"], 50)
    pipe = make_xgb_pipeline(cat_cols, params, cfg["seed"])
    cols = cat_cols + num_cols
    pipe.fit(frame[cols], frame[target],
             xgb__sample_weight=frame["sample_weight"])

    models_dir = resolve_path(cfg, "models_dir")
    joblib.dump(
        {"pipeline": pipe, "cat_cols": cat_cols, "num_cols": num_cols,
         "target": target, "params": params, "seed": cfg["seed"],
         "horizon": horizon},
        models_dir / f"{label}.joblib",
    )
    result["oof"].to_parquet(
        resolve_path(cfg, "reports_dir") / f"oof_{label}.parquet", index=False)
    return {label: {k: result[k] for k in
                    ("best_params", "best_iteration_mean", "metrics")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["a", "b", "all"], default="all")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Task B horizon in days (default: all configured)")
    args = parser.parse_args()

    cfg = load_config()
    set_seeds(cfg["seed"])
    df_raw = pd.read_parquet(resolve_path(cfg, "dataset"))
    resolve_path(cfg, "models_dir").mkdir(parents=True, exist_ok=True)
    resolve_path(cfg, "reports_dir").mkdir(parents=True, exist_ok=True)

    all_metrics: dict = {}
    if args.task in ("a", "all"):
        all_metrics.update(train_task_a(cfg, df_raw))
    if args.task in ("b", "all"):
        horizons = ([args.horizon] if args.horizon
                    else cfg["targets"]["task_b_horizons"])
        for h in horizons:
            try:
                all_metrics.update(train_task_b(cfg, df_raw, h))
            except RuntimeError as exc:
                log.warning("Skipping task B h=%d: %s", h, exc)
                all_metrics[f"task_b_h{h}"] = {"skipped": str(exc)}

    out = resolve_path(cfg, "reports_dir") / "metrics.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(all_metrics)
    out.write_text(json.dumps(existing, indent=2))
    log.info("Metrics written to %s", out)
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
