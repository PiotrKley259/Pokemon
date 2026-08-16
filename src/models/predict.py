"""Predict a card's fair value (Task A model) from the command line.

Usage:
    python -m src.models.predict --card-id xy7-54 --variant holofoil
"""
from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.features.build_features import build_task_a_frame


def predict_card(cfg: dict, card_id: str, variant: str) -> dict:
    bundle = joblib.load(resolve_path(cfg, "models_dir") / "task_a.joblib")
    df_raw = pd.read_parquet(resolve_path(cfg, "dataset"))

    frame = build_task_a_frame(df_raw)
    row = frame[(frame["card_id"] == card_id) & (frame["variant"] == variant)]
    if row.empty:
        available = frame.loc[frame["card_id"] == card_id, "variant"].tolist()
        raise KeyError(
            f"No priced row for card_id={card_id!r} variant={variant!r}. "
            + (f"Variants available for this card: {available}" if available
               else "Card not found in the dataset.")
        )

    cols = bundle["cat_cols"] + bundle["num_cols"]
    log_pred = float(bundle["pipeline"].predict(row[cols])[0])
    fair_value = float(np.exp(log_pred))
    actual = float(row["price_market"].iloc[0])
    return {
        "card_id": card_id,
        "name": row["name"].iloc[0],
        "set": row["set_name"].iloc[0],
        "rarity": row["rarity"].iloc[0],
        "variant": variant,
        "snapshot_date": str(row["snapshot_date"].iloc[0].date()),
        "predicted_fair_value": fair_value,
        "actual_market_price": actual,
        "gap_pct": 100.0 * (fair_value - actual) / actual,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card-id", required=True, help="e.g. xy7-54")
    parser.add_argument("--variant", default="holofoil",
                        help="normal | holofoil | reverseHolofoil | ...")
    args = parser.parse_args()

    cfg = load_config()
    try:
        res = predict_card(cfg, args.card_id, args.variant)
    except (KeyError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"{res['name']} ({res['set']}, {res['rarity']}) "
          f"[{res['variant']}] @ {res['snapshot_date']}")
    print(f"  predicted fair value : ${res['predicted_fair_value']:.2f}")
    print(f"  actual market price  : ${res['actual_market_price']:.2f}")
    verdict = "undervalued" if res["gap_pct"] > 0 else "overvalued"
    print(f"  gap                  : {res['gap_pct']:+.1f}%  ({verdict} "
          "per the model; educational estimate, not financial advice)")


if __name__ == "__main__":
    main()
