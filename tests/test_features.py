import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    add_forward_return_target,
    add_history_features,
    add_liquidity_weight,
    add_static_features,
    build_task_a_frame,
    parse_rarity_flags,
)


def test_parse_rarity_flags():
    rarities = pd.Series([
        "Rare Secret", "Rare Rainbow", "Illustration Rare",
        "Rare Ultra", "Full Art", "Promo", "Common", None,
    ])
    flags = parse_rarity_flags(rarities)
    assert flags.loc[0, "is_secret"] == 1
    assert flags.loc[1, "is_secret"] == 1
    assert flags.loc[2, "is_alt_art"] == 1
    assert flags.loc[3, "is_full_art"] == 1
    assert flags.loc[4, "is_full_art"] == 1
    assert flags.loc[5, "is_promo"] == 1
    assert flags.loc[6].sum() == 0
    assert flags.loc[7].sum() == 0  # NaN rarity -> all zero, no crash


def test_static_features(toy_panel):
    out = add_static_features(toy_panel)
    row = out[out["card_id"] == "a-1"].iloc[0]
    assert row["is_starter"] == 1        # Charmander line, dex 6
    assert row["is_first_gen"] == 1
    assert row["is_legendary"] == 0
    assert row["card_number_ratio"] == pytest.approx(4 / 100)
    assert row["hp"] == 120.0            # string "120" coerced to float
    expected_days = (row["snapshot_date"] - row["release_date"]).days
    assert row["days_since_release"] == expected_days


def test_liquidity_weight(toy_panel):
    df = toy_panel.copy()
    # Knock out most of a-2's prices -> lower reliability weight than a-1.
    df.loc[(df["card_id"] == "a-2") & (df.index % 3 != 0), "price_market"] = np.nan
    out = add_liquidity_weight(df)
    w1 = out.loc[out["card_id"] == "a-1", "sample_weight"].iloc[0]
    w2 = out.loc[out["card_id"] == "a-2", "sample_weight"].iloc[0]
    assert w1 > w2 > 0
    assert out["sample_weight"].mean() == pytest.approx(1.0)


def test_trailing_returns_use_past_only(toy_panel, toy_cfg):
    out = add_history_features(add_static_features(toy_panel), toy_cfg)
    a1 = out[out["card_id"] == "a-1"].sort_values("snapshot_date")
    # First snapshot has no 7d history.
    assert np.isnan(a1["ret_7d"].iloc[0])
    # a-1 grows by 2**(1/25) per week -> weekly log return = log(2)/25.
    assert a1["ret_7d"].iloc[5] == pytest.approx(np.log(2) / 25, rel=1e-6)
    # Flat card has ~zero returns and zero volatility.
    a2 = out[out["card_id"] == "a-2"]
    assert np.nanmax(np.abs(a2["ret_30d"])) == pytest.approx(0.0, abs=1e-12)
    assert np.nanmax(a2["vol_30d"]) == pytest.approx(0.0, abs=1e-12)


def test_price_rel_set_median(toy_panel, toy_cfg):
    out = add_history_features(add_static_features(toy_panel), toy_cfg)
    day1 = out[out["snapshot_date"] == out["snapshot_date"].min()]
    # Both cards cost 10 on day 1 -> both sit at the set median -> log ratio 0.
    assert day1["price_rel_set_median"].abs().max() == pytest.approx(0.0)


def test_forward_return_target(toy_panel, toy_cfg):
    out = add_forward_return_target(toy_panel, 30, 7)
    a1 = out[out["card_id"] == "a-1"].sort_values("snapshot_date")
    # 30 days ~ 4-5 weekly steps; growth log(2)/25 per step. Snapshot at
    # t+28d is within the 7-day tolerance.
    expected = 4 * np.log(2) / 25
    assert a1["fwd_ret_30d"].iloc[0] == pytest.approx(expected, rel=1e-6)
    # Last snapshots have no future price within tolerance -> NaN target.
    assert np.isnan(a1["fwd_ret_30d"].iloc[-1])
    # Flat card: forward return exactly zero.
    a2 = out[out["card_id"] == "a-2"]
    assert np.nanmax(np.abs(a2["fwd_ret_30d"])) == pytest.approx(0.0, abs=1e-12)


def test_forward_return_never_uses_past_price(toy_panel):
    # A card with a single snapshot must get a NaN target, not a "return"
    # computed against itself or a past snapshot.
    single = toy_panel[toy_panel["snapshot_date"]
                       == toy_panel["snapshot_date"].min()]
    out = add_forward_return_target(single, 30, 7)
    assert out["fwd_ret_30d"].isna().all()


def test_task_a_frame_latest_snapshot_only(toy_panel):
    frame = build_task_a_frame(toy_panel)
    # One row per (card_id, variant), taken at the latest date.
    assert len(frame) == 2
    assert (frame["snapshot_date"] == toy_panel["snapshot_date"].max()).all()
    a1 = frame[frame["card_id"] == "a-1"].iloc[0]
    assert a1["log_price"] == pytest.approx(np.log(a1["price_market"]))
