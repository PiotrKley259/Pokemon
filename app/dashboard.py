"""Streamlit dashboard: search a card, compare predicted vs actual price,
see the SHAP waterfall for that card, and browse the most undervalued cards.

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.config import load_config, resolve_path
from src.features.build_features import build_task_a_frame

st.set_page_config(page_title="Pokemon TCG fair value", layout="wide")


@st.cache_resource
def load_bundle():
    cfg = load_config()
    bundle = joblib.load(resolve_path(cfg, "models_dir") / "task_a.joblib")
    return cfg, bundle


@st.cache_data
def load_frame():
    cfg = load_config()
    df_raw = pd.read_parquet(resolve_path(cfg, "dataset"))
    return build_task_a_frame(df_raw)


@st.cache_data
def score_frame():
    cfg, bundle = load_bundle()
    frame = load_frame()
    cols = bundle["cat_cols"] + bundle["num_cols"]
    frame = frame.copy()
    frame["pred_log"] = bundle["pipeline"].predict(frame[cols])
    frame["fair_value"] = np.exp(frame["pred_log"])
    frame["gap_pct"] = 100 * (frame["fair_value"] - frame["price_market"]) \
        / frame["price_market"]
    return frame


def shap_waterfall(bundle, row: pd.DataFrame):
    cols = bundle["cat_cols"] + bundle["num_cols"]
    X = bundle["pipeline"].named_steps["cats"].transform(row[cols])
    explainer = shap.TreeExplainer(bundle["pipeline"].named_steps["xgb"])
    sv = explainer(X)
    fig = plt.figure()
    shap.plots.waterfall(sv[0], max_display=14, show=False)
    plt.tight_layout()
    return fig


def main():
    st.title("Pokemon TCG card fair-value explorer")
    st.caption(
        "Educational project - model estimates from public TCGPlayer prices. "
        "Not financial advice."
    )

    try:
        cfg, bundle = load_bundle()
        scored = score_frame()
    except FileNotFoundError:
        st.error(
            "Model or dataset missing. Run `make data` then `make train` first."
        )
        return

    tab_card, tab_ranked = st.tabs(["Card lookup", "Most undervalued"])

    with tab_card:
        scored["label"] = (
            scored["name"] + " - " + scored["set_name"].fillna("?")
            + " (" + scored["card_id"] + ", " + scored["variant"] + ")"
        )
        query = st.text_input("Search by card name", "Charizard")
        matches = scored[scored["name"].str.contains(query, case=False, na=False)]
        if matches.empty:
            st.info("No card matches that search.")
        else:
            choice = st.selectbox("Card / variant", matches["label"].tolist())
            row = matches[matches["label"] == choice].head(1)

            c1, c2 = st.columns([1, 2])
            with c1:
                if pd.notna(row["image_url"].iloc[0]):
                    st.image(row["image_url"].iloc[0], width=260)
            with c2:
                st.metric("Actual market price",
                          f"${row['price_market'].iloc[0]:,.2f}")
                st.metric("Predicted fair value",
                          f"${row['fair_value'].iloc[0]:,.2f}",
                          delta=f"{row['gap_pct'].iloc[0]:+.1f}% vs market")
                st.write(
                    f"**{row['rarity'].iloc[0]}** · {row['series'].iloc[0]} · "
                    f"snapshot {row['snapshot_date'].iloc[0].date()}"
                )
                st.subheader("Why? (SHAP waterfall, log-price space)")
                st.pyplot(shap_waterfall(bundle, row))

    with tab_ranked:
        st.subheader("Ranked by model-implied undervaluation")
        min_price = st.slider("Minimum market price ($)", 0.0, 50.0, 1.0)
        min_liq = st.slider("Minimum price snapshots (liquidity)", 1, 20, 2)
        table = scored[
            (scored["price_market"] >= min_price)
            & (scored["n_price_snapshots"] >= min_liq)
        ].nlargest(50, "gap_pct")[
            ["card_id", "name", "set_name", "rarity", "variant",
             "price_market", "fair_value", "gap_pct", "n_price_snapshots"]
        ]
        st.dataframe(
            table.style.format({"price_market": "${:.2f}",
                                "fair_value": "${:.2f}",
                                "gap_pct": "{:+.1f}%"}),
            use_container_width=True, height=600,
        )


if __name__ == "__main__":
    main()
