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
from src.features.build_features import add_static_features, build_task_a_frame
from src.features.neighbors import build_neighbor_features, nearest_neighbors
from src.features.price_decomposition import PriceDecomposer  # noqa: F401 (unpickling)

st.set_page_config(page_title="Pokemon TCG fair value", layout="wide")


@st.cache_resource
def load_bundle():
    cfg = load_config()
    bundle = joblib.load(resolve_path(cfg, "models_dir") / "task_a.joblib")
    return cfg, bundle


@st.cache_data
def load_raw():
    cfg = load_config()
    return pd.read_parquet(resolve_path(cfg, "dataset"))


@st.cache_data
def load_frame():
    return build_task_a_frame(load_raw())


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
    # Hype-outlier flag: price sits far above the set x rarity median, i.e.
    # the market prices something (a collab, scarcity event, cultural status)
    # that no feature describes. The model's gap for these cards measures the
    # hype itself, not mispricing. 3 log-units = ~20x the comparable median.
    peer_median = frame.groupby(["set_id", "rarity"], observed=True)[
        "price_market"].transform("median")
    frame["hype_dev"] = np.log(frame["price_market"]) - \
        np.log(peer_median.where(peer_median > 0))
    frame["hype_outlier"] = frame["hype_dev"] > 3.0
    return frame


def price_history_chart(cfg, card_id: str, variant: str):
    """Observed snapshot history plus the Task-B forecast path (dashed)."""
    from src.models.forecast import forecast_card

    hist = load_raw()
    hist = hist[(hist["card_id"] == card_id) & (hist["variant"] == variant)]
    hist = hist.dropna(subset=["price_market"]).sort_values("snapshot_date")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(hist["snapshot_date"], hist["price_market"], marker="o",
            markersize=3, linewidth=1.2, label="observed market price")
    forecast_err = None
    try:
        path = forecast_card(cfg, card_id, variant)
        ax.plot(path["date"], path["predicted_price"], linestyle="--",
                linewidth=1.2, color="darkorange", label="predicted path")
        anchors = path[path["is_anchor"]]
        ax.scatter(anchors["date"], anchors["predicted_price"],
                   color="darkorange", s=25, zorder=3)
    except (FileNotFoundError, KeyError) as exc:
        forecast_err = str(exc)
    ax.set_ylabel("price ($)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig, forecast_err


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

    tab_card, tab_ranked, tab_scanner = st.tabs(
        ["Card lookup", "Most undervalued", "New Set Scanner"])

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
                if bool(row["hype_outlier"].iloc[0]):
                    st.warning(
                        "🔥 **Hype outlier** — this card trades at "
                        f"~{np.exp(row['hype_dev'].iloc[0]):.0f}× the median "
                        "of comparable cards in its set and rarity. Its price "
                        "is driven by factors outside the model's features "
                        "(collabs, scarcity events, cultural status), so the "
                        "gap above measures the hype itself, not mispricing."
                    )
                st.write(
                    f"**{row['rarity'].iloc[0]}** · {row['series'].iloc[0]} · "
                    f"snapshot {row['snapshot_date'].iloc[0].date()}"
                )
                st.subheader("Price history & predicted path")
                fig, forecast_err = price_history_chart(
                    cfg, row["card_id"].iloc[0], row["variant"].iloc[0])
                st.pyplot(fig)
                if forecast_err:
                    st.info(f"No forecast yet: {forecast_err}")
                else:
                    st.caption(
                        "Dashed path: current price compounded with the "
                        "Task B forward-return predictions (30/90d point "
                        "estimates, interpolated in log space). See "
                        "reports/metrics.json for the models' validated "
                        "error. Not financial advice."
                    )
                st.subheader("Why? (SHAP waterfall, log-price space)")
                st.pyplot(shap_waterfall(bundle, row))

    with tab_ranked:
        st.subheader("Ranked by model-implied undervaluation")
        min_price = st.slider("Minimum market price ($)", 0.0, 50.0, 1.0)
        min_liq = st.slider("Minimum price snapshots (liquidity)", 1, 20, 2)
        hide_hype = st.checkbox(
            "Exclude hype outliers (price > ~20× comparable median — the "
            "model cannot judge these)", value=True)
        pool = scored[
            (scored["price_market"] >= min_price)
            & (scored["n_price_snapshots"] >= min_liq)
        ]
        if hide_hype:
            pool = pool[~pool["hype_outlier"]]
        table = pool.nlargest(50, "gap_pct")[
            ["card_id", "name", "set_name", "rarity", "variant",
             "price_market", "fair_value", "gap_pct", "hype_outlier",
             "n_price_snapshots"]
        ]
        table["hype_outlier"] = table["hype_outlier"].map({True: "🔥", False: ""})
        st.dataframe(
            table.style.format({"price_market": "${:.2f}",
                                "fair_value": "${:.2f}",
                                "gap_pct": "{:+.1f}%"}),
            use_container_width=True, height=600,
        )

    with tab_scanner:
        render_new_set_scanner()


# --------------------------------------------------------------------------
# New Set Scanner (cold-start visual model)
# --------------------------------------------------------------------------
@st.cache_resource
def load_coldstart():
    cfg = load_config()
    return cfg, joblib.load(
        resolve_path(cfg, "models_dir") / "coldstart_premium.joblib")


def _confidence(novelty: float, dispersion: float) -> tuple[str, str]:
    """Badge from novelty (extrapolation risk) + neighbour dispersion."""
    if not np.isfinite(novelty) or not np.isfinite(dispersion):
        return "no data", "⚪"
    if novelty > 0.35 or dispersion > 0.5:
        return "low confidence", "🔴"
    if novelty > 0.2 or dispersion > 0.25:
        return "medium confidence", "🟡"
    return "high confidence", "🟢"


def score_new_set(cfg, bundle, set_id: str) -> pd.DataFrame | None:
    from src.features.embeddings import load_embeddings

    df_raw = pd.read_parquet(resolve_path(cfg, "dataset"))
    rows = df_raw[df_raw["set_id"] == set_id]
    if rows.empty:
        return None
    latest = (add_static_features(rows).sort_values("snapshot_date")
              .groupby(["card_id", "variant"], sort=False).tail(1)
              .reset_index(drop=True))

    emb_all, ids = load_embeddings(cfg, bundle["crop"], bundle["encoder"])
    idx = {cid: i for i, cid in enumerate(ids)}
    latest = latest[latest["card_id"].isin(idx)].reset_index(drop=True)
    if latest.empty:
        return None
    emb = emb_all[[idx[c] for c in latest["card_id"]]]

    comp = bundle["decomposer"].transform(latest)
    latest["character_equity"] = comp["character_equity"].to_numpy()
    latest["rarity_set_baseline"] = comp["rarity_set_baseline"].to_numpy()

    z = bundle["pca"].transform(emb)
    style = bundle["residualiser"].transform(z, latest.assign(era=latest["series"]))
    for j in range(style.shape[1]):
        latest[f"style_{j}"] = style[:, j]

    corpus = bundle["corpus"]
    have_returns = np.isfinite(corpus["fwd_return"]).any()
    nb = build_neighbor_features(
        emb, latest["set_id"], corpus_emb=corpus["emb"],
        corpus_set_ids=pd.Series(corpus["set_id"]),
        corpus_returns=corpus["fwd_return"] if have_returns else None,
        corpus_premiums=corpus["art_premium"],
        ks=bundle["knn_ks"], top_decile_pct=bundle["top_decile_pct"])
    for col in nb.columns:
        latest[col] = nb[col].to_numpy()

    cols = bundle["cat_cols"] + bundle["num_cols"]
    X = latest[cols].copy()
    for c, dtype in bundle["cat_dtypes"].items():
        X[c] = X[c].astype(dtype)
    latest["pred_art_premium"] = bundle["model"].predict(X)
    latest["_emb_row"] = range(len(latest))
    latest.attrs["emb"] = emb
    return latest


def render_new_set_scanner():
    st.subheader("New Set Scanner — cold-start visual scoring")
    st.caption(
        "Scores cards from art + static metadata only (no price history). "
        "The predicted **art premium** is the log-price residual after "
        "removing character equity and the rarity/era baseline."
    )
    try:
        cfg, bundle = load_coldstart()
    except FileNotFoundError:
        st.warning("Cold-start model missing. Run `make images`, "
                   "`make embed`, then `make coldstart` first.")
        return

    df_raw = pd.read_parquet(resolve_path(cfg, "dataset"))
    sets = (df_raw.dropna(subset=["release_date"])
            .groupby("set_id").agg(set_name=("set_name", "first"),
                                   release=("release_date", "max"))
            .sort_values("release", ascending=False))
    label_map = {f"{r.set_name} ({sid}, {r.release.date()})": sid
                 for sid, r in sets.iterrows()}
    choice = st.selectbox("Set (newest first)", list(label_map))
    scored = score_new_set(cfg, bundle, label_map[choice])
    if scored is None:
        st.info("No cards with cached embeddings for this set — run "
                "`make images` and `make embed` after collecting it.")
        return

    scored = scored.sort_values("pred_art_premium", ascending=False)
    corpus = bundle["corpus"]
    emb = scored.attrs["emb"]
    top_n = st.slider("Cards to show", 5, 50, 15)

    for _, card in scored.head(top_n).iterrows():
        conf_label, conf_icon = _confidence(card["novelty_score"],
                                            card["neighbour_dispersion"])
        with st.expander(
            f"{conf_icon} {card['name']} ({card['variant']}) — "
            f"predicted art premium {card['pred_art_premium']:+.2f}",
            expanded=False,
        ):
            c1, c2 = st.columns([1, 3])
            with c1:
                if pd.notna(card["image_url"]):
                    st.image(card["image_url"], width=180)
                st.write(f"**{card['rarity']}**")
                st.write(f"{conf_icon} {conf_label}")
                st.write(f"novelty {card['novelty_score']:.2f} · "
                         f"dispersion {card['neighbour_dispersion']:.2f}")
            with c2:
                st.write("**5 nearest visual neighbours** "
                         "(historical 90d return / art premium)")
                nn_idx, nn_sim = nearest_neighbors(
                    emb[int(card["_emb_row"])], card["set_id"],
                    corpus["emb"], pd.Series(corpus["set_id"]), n=5)
                thumb_cols = st.columns(max(len(nn_idx), 1))
                for col, j, s in zip(thumb_cols, nn_idx, nn_sim):
                    with col:
                        if pd.notna(corpus["image_url"][j]):
                            st.image(corpus["image_url"][j], width=110)
                        ret = corpus["fwd_return"][j]
                        ret_txt = (f"ret {ret:+.1%} · " if np.isfinite(ret)
                                   else "")
                        st.caption(
                            f"{corpus['name'][j]}\n"
                            f"sim {s:.2f} · {ret_txt}prem "
                            f"{corpus['art_premium'][j]:+.2f}")


if __name__ == "__main__":
    main()
