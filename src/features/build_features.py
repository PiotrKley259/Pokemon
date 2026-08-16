"""Feature engineering.

All functions are pure DataFrame -> DataFrame transforms. Nothing here fits
statistics on the data (no encoders, no scalers) - anything that must be
*fitted* lives inside the per-fold sklearn Pipeline in src/models/train.py,
so no information can leak across CV folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Pokedex numbers for popularity proxies -------------------------------------

# Starter families (base + both evolutions) for gens 1-9, plus Pikachu/Eevee.
_STARTER_BASES = [1, 4, 7, 152, 155, 158, 252, 255, 258, 387, 390, 393,
                  495, 498, 501, 650, 653, 656, 722, 725, 728, 810, 813, 816,
                  906, 909, 912]
STARTER_DEX = set()
for base in _STARTER_BASES:
    STARTER_DEX.update({base, base + 1, base + 2})
STARTER_DEX.update({25, 26, 133})  # Pikachu line + Eevee

# Legendaries and mythicals, gens 1-9.
LEGENDARY_DEX = set()
for rng in [(144, 146), (150, 151), (243, 245), (249, 251), (377, 386),
            (480, 494), (638, 649), (716, 721), (785, 809), (888, 898),
            (905, 905), (1001, 1010)]:
    LEGENDARY_DEX.update(range(rng[0], rng[1] + 1))

FIRST_GEN_MAX_DEX = 151


def parse_rarity_flags(rarity: pd.Series) -> pd.DataFrame:
    """Parse full-art / alt-art / secret / promo flags from the rarity string."""
    r = rarity.fillna("").str.lower()
    return pd.DataFrame({
        "is_full_art": r.str.contains("full art|ultra|vmax|vstar").astype(int),
        "is_alt_art": r.str.contains(
            "alt|illustration rare|special illustration|character"
        ).astype(int),
        "is_secret": r.str.contains(
            "secret|rainbow|gold|hyper rare"
        ).astype(int),
        "is_promo": r.str.contains("promo").astype(int),
    }, index=rarity.index)


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Card-level features valid for both tasks (no price history involved)."""
    out = df.copy()

    flags = parse_rarity_flags(out["rarity"])
    for col in flags.columns:
        out[col] = flags[col]

    dex = pd.to_numeric(out["pokedex_number"], errors="coerce")
    out["is_starter"] = dex.isin(STARTER_DEX).astype(int)
    out["is_legendary"] = dex.isin(LEGENDARY_DEX).astype(int)
    out["is_first_gen"] = ((dex >= 1) & (dex <= FIRST_GEN_MAX_DEX)).astype(int)

    set_size = pd.to_numeric(out["set_size"], errors="coerce")
    card_number = pd.to_numeric(out["card_number"], errors="coerce")
    out["card_number_ratio"] = card_number / set_size.where(set_size > 0)

    out["days_since_release"] = (
        out["snapshot_date"] - out["release_date"]
    ).dt.days.astype("float64")

    for col in ["hp", "retreat_cost", "attack_count", "card_number",
                "set_size", "pokedex_number"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def add_liquidity_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Reliability proxy: how many snapshots have a non-null market price.

    Used as an XGBoost sample weight so thinly traded cards (one noisy
    snapshot) don't dominate the loss. Weight = log1p(count), normalised
    to mean 1 so the effective loss scale is unchanged.
    """
    out = df.copy()
    counts = (
        out["price_market"].notna()
        .groupby([out["card_id"], out["variant"]])
        .transform("sum")
    )
    out["n_price_snapshots"] = counts
    w = np.log1p(counts.astype(float))
    out["sample_weight"] = w / max(w.mean(), 1e-12)
    return out


def _trailing_return(g: pd.DataFrame, window: int, tol: int) -> pd.Series:
    """log(P_t / P_{t-window}) using the closest past snapshot within tol days."""
    past = pd.merge_asof(
        g[["snapshot_date"]].assign(
            lookup=g["snapshot_date"] - pd.Timedelta(days=window)
        ).sort_values("lookup"),
        g[["snapshot_date", "price_market"]].rename(
            columns={"snapshot_date": "past_date", "price_market": "past_price"}
        ).sort_values("past_date"),
        left_on="lookup", right_on="past_date",
        direction="backward", tolerance=pd.Timedelta(days=tol),
    ).set_index(g.index)
    # Only accept a genuinely earlier snapshot as the reference price.
    valid = past["past_date"] < g["snapshot_date"]
    return np.log(g["price_market"] / past["past_price"].where(valid))


def add_history_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Price-history features for Task B. Uses only past data at each row."""
    out = df.sort_values(["card_id", "variant", "snapshot_date"]).copy()
    windows = cfg["features"]["history"]["return_windows"]
    vol_window = cfg["features"]["history"]["vol_window"]
    tol = cfg["targets"]["horizon_tolerance_days"]

    grouped = out.groupby(["card_id", "variant"], group_keys=False, sort=False)

    for w in windows:
        ret = grouped.apply(
            lambda g, w=w: _trailing_return(g, w, tol), include_groups=False
        )
        if isinstance(ret.index, pd.MultiIndex):
            ret = ret.reset_index(level=list(range(ret.index.nlevels - 1)), drop=True)
        out[f"ret_{w}d"] = ret

    # Rolling volatility of snapshot-over-snapshot log returns (past only).
    step_ret = grouped["price_market"].transform(lambda s: np.log(s / s.shift(1)))
    out["step_log_ret"] = step_ret
    out[f"vol_{vol_window}d"] = (
        out.groupby(["card_id", "variant"], sort=False)["step_log_ret"]
        .transform(lambda s: s.rolling(window=5, min_periods=2).std())
    )
    out = out.drop(columns=["step_log_ret"])

    # Price relative to the same-day median of the card's set (cross-sectional,
    # uses only time-t information).
    set_median = out.groupby(["set_id", "snapshot_date"], sort=False)[
        "price_market"
    ].transform("median")
    out["price_rel_set_median"] = np.log(
        out["price_market"] / set_median.where(set_median > 0)
    )
    return out


def add_forward_return_target(
    df: pd.DataFrame, horizon_days: int, tol: int
) -> pd.DataFrame:
    """Task B target: log(P_{t+h} / P_t).

    We predict the *return*, never the raw future price level: gradient-
    boosted trees predict a constant (the training-leaf mean) outside the
    range of targets seen in training, so a tree trained on price levels
    cannot extrapolate an upward-trending market - every future price above
    the historical maximum would be clipped to it. Returns are roughly
    stationary, so trees can model them.
    """
    out = df.sort_values(["card_id", "variant", "snapshot_date"]).copy()

    def _future(g: pd.DataFrame) -> pd.Series:
        fut = pd.merge_asof(
            g[["snapshot_date"]].assign(
                lookup=g["snapshot_date"] + pd.Timedelta(days=horizon_days)
            ).sort_values("lookup"),
            g[["snapshot_date", "price_market"]].rename(
                columns={"snapshot_date": "fut_date", "price_market": "fut_price"}
            ).sort_values("fut_date"),
            left_on="lookup", right_on="fut_date",
            direction="nearest", tolerance=pd.Timedelta(days=tol),
        ).set_index(g.index)
        # Only accept genuinely future snapshots.
        valid = fut["fut_date"] > g["snapshot_date"]
        return fut["fut_price"].where(valid)

    fut_price = (
        out.groupby(["card_id", "variant"], group_keys=False, sort=False)
        .apply(_future, include_groups=False)
    )
    if isinstance(fut_price.index, pd.MultiIndex):
        fut_price = fut_price.reset_index(
            level=list(range(fut_price.index.nlevels - 1)), drop=True
        )
    out[f"fwd_ret_{horizon_days}d"] = np.log(fut_price / out["price_market"])
    return out


def build_task_a_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Latest snapshot per (card_id, variant) with log(market_price) target."""
    out = add_static_features(df)
    out = add_liquidity_weight(out)
    latest = (
        out.sort_values("snapshot_date")
        .groupby(["card_id", "variant"], sort=False)
        .tail(1)
    )
    latest = latest[latest["price_market"] > 0].copy()
    latest["log_price"] = np.log(latest["price_market"])
    return latest.reset_index(drop=True)


def build_task_b_frame(df: pd.DataFrame, cfg: dict, horizon_days: int) -> pd.DataFrame:
    """All snapshots with history features and the forward-return target."""
    out = add_static_features(df)
    out = add_liquidity_weight(out)
    out = add_history_features(out, cfg)
    out = add_forward_return_target(
        out, horizon_days, cfg["targets"]["horizon_tolerance_days"]
    )
    out = out[out["price_market"] > 0].copy()
    return out.dropna(subset=[f"fwd_ret_{horizon_days}d"]).reset_index(drop=True)


def feature_columns(cfg: dict, task: str, horizon_days: int | None = None) -> tuple[list[str], list[str]]:
    """Return (categorical, numeric) feature column names for a task."""
    cat = list(cfg["features"]["categorical"])
    num = list(cfg["features"]["numeric"])
    if task == "b":
        windows = cfg["features"]["history"]["return_windows"]
        vol_window = cfg["features"]["history"]["vol_window"]
        num += [f"ret_{w}d" for w in windows]
        num += [f"vol_{vol_window}d", "price_rel_set_median", "n_price_snapshots"]
    return cat, num
