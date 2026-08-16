"""Predicted future price path for a card.

Projects a card's price forward using the Task B forward-return models:

    P_pred(t + h) = P(t) * exp(predicted_log_return_h)

for each trained horizon (30d, 90d), then interpolates a daily path in log
space between today and the horizons. We forecast the RETURN and re-anchor
on the current price rather than predicting future price levels directly -
gradient-boosted trees cannot extrapolate beyond the price range they were
trained on, but a return forecast composes with any current price.

The horizon predictions are point estimates from a model whose honest
validation error is reported in reports/metrics.json - treat the path as a
scenario, not a promise.

Usage:
    python -m src.models.forecast --card-id xy7-54 --variant holofoil
"""
from __future__ import annotations

import argparse
import logging
import sys

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.features.build_features import (
    add_history_features,
    add_liquidity_weight,
    add_static_features,
)

log = logging.getLogger("forecast")


def available_horizon_models(cfg: dict) -> dict[int, dict]:
    """Load whichever task_b bundles exist, keyed by horizon days."""
    models = {}
    for h in cfg["targets"]["task_b_horizons"]:
        path = resolve_path(cfg, "models_dir") / f"task_b_h{h}.joblib"
        if path.exists():
            models[h] = joblib.load(path)
    return models


def latest_feature_row(cfg: dict, df_raw: pd.DataFrame, card_id: str,
                       variant: str) -> pd.DataFrame:
    """Build the Task-B feature row for a card's most recent snapshot."""
    hist = df_raw[(df_raw["card_id"] == card_id)
                  & (df_raw["variant"] == variant)]
    if hist.empty or hist["price_market"].dropna().empty:
        raise KeyError(f"No priced history for {card_id!r} / {variant!r}")
    # History features need the card's own past plus same-day set peers for
    # the set-median feature; restrict to the involved sets to stay fast.
    set_id = hist["set_id"].iloc[0]
    ctx = df_raw[df_raw["set_id"] == set_id]
    ctx = add_static_features(ctx)
    ctx = add_liquidity_weight(ctx)
    ctx = add_history_features(ctx, cfg)
    row = (ctx[(ctx["card_id"] == card_id) & (ctx["variant"] == variant)]
           .sort_values("snapshot_date").tail(1))
    return row


def forecast_card(cfg: dict, card_id: str, variant: str) -> pd.DataFrame:
    """Return a daily predicted price path (log-space interpolation between
    today's price and each horizon's predicted level)."""
    models = available_horizon_models(cfg)
    if not models:
        raise FileNotFoundError(
            "No Task B models trained yet - forward-return models need "
            "snapshots spanning the horizon. Keep running `make collect` "
            "daily, then `make train`."
        )
    df_raw = pd.read_parquet(resolve_path(cfg, "dataset"))
    row = latest_feature_row(cfg, df_raw, card_id, variant)
    p0 = float(row["price_market"].iloc[0])
    t0 = row["snapshot_date"].iloc[0]

    anchors: list[tuple[int, float]] = [(0, np.log(p0))]
    for h, bundle in sorted(models.items()):
        cols = bundle["cat_cols"] + bundle["num_cols"]
        r_hat = float(bundle["pipeline"].predict(row[cols])[0])
        anchors.append((h, np.log(p0) + r_hat))

    days = np.arange(0, max(models) + 1)
    log_path = np.interp(days, [a[0] for a in anchors],
                         [a[1] for a in anchors])
    return pd.DataFrame({
        "date": t0 + pd.to_timedelta(days, unit="D"),
        "predicted_price": np.exp(log_path),
        "is_anchor": np.isin(days, [a[0] for a in anchors]),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card-id", required=True)
    parser.add_argument("--variant", default="holofoil")
    args = parser.parse_args()
    cfg = load_config()
    try:
        path = forecast_card(cfg, args.card_id, args.variant)
    except (KeyError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    p0 = path["predicted_price"].iloc[0]
    print(f"{args.card_id} [{args.variant}] - current ${p0:,.2f}")
    for _, a in path[path["is_anchor"]].iloc[1:].iterrows():
        chg = 100 * (a["predicted_price"] / p0 - 1)
        print(f"  {a['date'].date()}  predicted ${a['predicted_price']:,.2f} "
              f"({chg:+.1f}%)")
    print("Point estimates from the forward-return models; see "
          "reports/metrics.json for their validated error. Not financial "
          "advice.")


if __name__ == "__main__":
    main()
