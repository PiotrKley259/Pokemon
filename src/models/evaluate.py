"""Interpretability and residual analysis for a trained model.

Produces, under reports/figures/:
- shap_summary_<task>.png            SHAP beeswarm over all features
- shap_dependence_<task>_<feat>.png  dependence plots for the top-5 features
- pred_vs_actual_<task>.png          out-of-fold predicted vs actual
- residuals_by_rarity_<task>.png     which rarities are hardest
- residuals_by_set_<task>.png        which sets are hardest (top 20 by |resid|)

Usage:
    python -m src.models.evaluate --task a
    python -m src.models.evaluate --task b --horizon 30
"""
from __future__ import annotations

import argparse
import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import load_config, resolve_path, set_seeds

log = logging.getLogger("evaluate")


def _load(cfg: dict, label: str):
    bundle = joblib.load(resolve_path(cfg, "models_dir") / f"{label}.joblib")
    oof = pd.read_parquet(resolve_path(cfg, "reports_dir") / f"oof_{label}.parquet")
    return bundle, oof


def shap_plots(cfg: dict, bundle: dict, oof: pd.DataFrame, label: str,
               max_rows: int = 5000) -> None:
    fig_dir = resolve_path(cfg, "figures_dir")
    cols = bundle["cat_cols"] + bundle["num_cols"]
    pipe = bundle["pipeline"]
    sample = oof.sample(min(max_rows, len(oof)), random_state=cfg["seed"])
    X = pipe.named_steps["cats"].transform(sample[cols])

    explainer = shap.TreeExplainer(pipe.named_steps["xgb"])
    sv = explainer(X)

    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(fig_dir / f"shap_summary_{label}.png", dpi=150)
    plt.close("all")

    # dependence_plot sorts raw feature values; categorical columns with NaN
    # become mixed str/float arrays it cannot sort, so plot a stringified copy
    # (SHAP label-encodes strings itself).
    X_plot = X.copy()
    for c in bundle["cat_cols"]:
        # astype(str) on a Categorical keeps NaN as float NaN, so fill first.
        X_plot[c] = X_plot[c].astype(object).fillna("missing").astype(str)

    top5 = np.argsort(np.abs(sv.values).mean(axis=0))[::-1][:5]
    for idx in top5:
        feat = X.columns[idx]
        plt.figure()
        shap.dependence_plot(feat, sv.values, X_plot, show=False,
                             interaction_index=None)
        plt.tight_layout()
        plt.savefig(fig_dir / f"shap_dependence_{label}_{feat}.png", dpi=150)
        plt.close("all")
    log.info("SHAP plots for %s written to %s", label, fig_dir)


def residual_plots(cfg: dict, bundle: dict, oof: pd.DataFrame, label: str) -> None:
    fig_dir = resolve_path(cfg, "figures_dir")
    target = bundle["target"]
    oof = oof.copy()
    oof["residual"] = oof["pred_xgb"] - oof[target]

    plt.figure(figsize=(6, 6))
    plt.scatter(oof[target], oof["pred_xgb"], s=4, alpha=0.3)
    lims = [oof[target].min(), oof[target].max()]
    plt.plot(lims, lims, color="red", linewidth=1)
    plt.xlabel(f"actual {target}")
    plt.ylabel("predicted (out-of-fold)")
    plt.title(f"Predicted vs actual - {label}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"pred_vs_actual_{label}.png", dpi=150)
    plt.close("all")

    by_rarity = (oof.groupby("rarity", observed=True)["residual"]
                 .agg(["median", lambda s: s.abs().mean(), "count"])
                 .rename(columns={"<lambda_0>": "mae"})
                 .query("count >= 20").sort_values("mae", ascending=False))
    plt.figure(figsize=(8, max(4, 0.3 * len(by_rarity))))
    plt.barh(by_rarity.index.astype(str), by_rarity["mae"])
    plt.xlabel("mean |residual| (log space)")
    plt.title(f"Hardest rarities - {label}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"residuals_by_rarity_{label}.png", dpi=150)
    plt.close("all")

    by_set = (oof.groupby("set_name", observed=True)["residual"]
              .agg(mae=lambda s: s.abs().mean(), count="count")
              .query("count >= 20").sort_values("mae", ascending=False).head(20))
    plt.figure(figsize=(8, 7))
    plt.barh(by_set.index.astype(str), by_set["mae"])
    plt.xlabel("mean |residual| (log space)")
    plt.title(f"Hardest sets (top 20) - {label}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"residuals_by_set_{label}.png", dpi=150)
    plt.close("all")
    log.info("Residual plots for %s written to %s", label, fig_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["a", "b"], default="a")
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config()
    set_seeds(cfg["seed"])
    label = "task_a" if args.task == "a" else f"task_b_h{args.horizon}"
    bundle, oof = _load(cfg, label)
    resolve_path(cfg, "figures_dir").mkdir(parents=True, exist_ok=True)
    shap_plots(cfg, bundle, oof, label)
    residual_plots(cfg, bundle, oof, label)


if __name__ == "__main__":
    main()
