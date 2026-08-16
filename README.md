# Pokémon TCG Card Price Prediction

Predicts Pokémon TCG card prices with XGBoost, on top of price snapshots from
the [Pokémon TCG API](https://pokemontcg.io) (TCGPlayer market prices, plus
Cardmarket averages where available).

Inspired by
[nayelsdk/Quantitative_Portfolio_Pokemon_Card](https://github.com/nayelsdk/Quantitative_Portfolio_Pokemon_Card),
which does portfolio optimisation over card prices. This project adds the
predictive-modelling layer that repo doesn't have: two supervised tasks with
leakage-safe validation, baselines, and interpretability.

> **Disclaimer** — This is an educational project for learning quantitative
> methods and machine learning. Nothing here is financial or investment
> advice. Card prices are volatile, thinly traded, and driven by factors no
> model captures. Do not make purchase decisions based on this code.

## The two tasks

| | Task A — cross-sectional valuation | Task B — forward return |
|---|---|---|
| Target | `log(market_price)` at the latest snapshot | `log(P_{t+h} / P_t)` for h = 30, 90 days |
| Question | "Is this card cheap or expensive vs comparable cards?" | "Which cards will outperform over the next h days?" |
| Validation | GroupKFold grouped by `set_id` (a fold never trains on its own set) | Purged + embargoed expanding-window time splits |
| Baselines | set×rarity median, ridge regression | set×rarity median, ridge, **zero return** |

Task B deliberately predicts the *return*, never the raw future price: a
gradient-boosted tree predicts the mean of its training leaves and therefore
outputs a constant outside the target range it saw — it cannot extrapolate a
trending price level. Returns are roughly stationary, so trees can model them.

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
make install

# optional but recommended: free key from https://dev.pokemontcg.io
export POKEMONTCG_API_KEY=your-key

make data        # pull today's snapshot (resumable, cached, rate-limited)
                 # -> data/raw/snapshots/<date>/, data/processed/cards.parquet
make train       # Task A immediately; Task B once you have weeks of snapshots
make evaluate    # SHAP + residual figures -> reports/figures/
make test        # pytest: feature builder + time-split logic

python -m src.models.predict --card-id xy7-54 --variant holofoil
make dashboard   # Streamlit app
```

Run `make collect` on a schedule (e.g. daily cron) — each run adds one price
snapshot, and Task B becomes trainable once the history spans
`min_train_days` + one horizon.

## Project layout

```
config.yaml                  all paths & hyperparameters
src/data/collect.py          API collector: cached raw JSON, resumable, rate-limited
src/data/build_dataset.py    -> data/processed/cards.parquet, one row per
                             (card_id, variant, snapshot_date)
src/features/build_features.py  pure feature transforms + targets (no fitting!)
src/models/splits.py         PurgedExpandingTimeSplit (purge + embargo)
src/models/train.py          CV + Optuna + baselines + final model
src/models/evaluate.py       SHAP summary/dependence, residual analysis
src/models/predict.py        CLI fair-value lookup
app/dashboard.py             Streamlit dashboard
notebooks/                   01_eda, 02_modelling
tests/                       pytest for features and split logic
```

## Validation design (the part that matters)

- **No global fitting.** Category mappings, one-hot encoders, imputers and
  scalers live inside sklearn Pipelines fitted per fold. Feature engineering
  in `build_features.py` is purely row-local or uses only past/same-day data.
- **Task A** groups folds by `set_id`, so the model can't memorise a set's
  price level and must generalise from rarity/series/card attributes.
- **Task B** trains only on snapshots dated at least `h + embargo` days before
  the validation window starts. The purge removes training rows whose label
  window `[t, t+h]` overlaps validation; the embargo (5 days) adds slack for
  snapshot jitter and serial correlation. Windows expand, never shuffle.
- **Sample weights**: `log1p(#non-null price snapshots)` so one noisy print of
  a thinly traded card doesn't dominate the loss.
- **Optuna** (50 trials, TPE, seeded) searches `max_depth, learning_rate,
  min_child_weight, subsample, colsample_bytree, reg_lambda, reg_alpha`
  *inside* the CV loop, with early stopping on each validation fold.

## Metrics

Numbers depend on your collected snapshots — regenerate with `make train`,
which writes `reports/metrics.json`. Fill this table from that file:

| Task | Model | MAE (log) | RMSE (log) | MdAPE | Rank IC | Decile spread |
|---|---|---|---|---|---|---|
| A | set×rarity median | . | . | . | — | — |
| A | ridge | . | . | . | — | — |
| A | **XGBoost** | . | . | . | — | — |
| B (30d) | zero return | . | . | — | — | — |
| B (30d) | ridge | . | . | — | . | . |
| B (30d) | **XGBoost** | . | . | — | . | . |

If XGBoost does not beat the baselines on your data, the honest conclusion is
that it does not beat them — with few snapshots Task B especially is mostly
noise, and the zero-return baseline is genuinely hard to beat.

Out-of-fold predicted-vs-actual and SHAP plots: `reports/figures/` after
`make evaluate` (e.g. `pred_vs_actual_task_a.png`, `shap_summary_task_a.png`).

## Cold-start visual model (new sets, no price history)

A newly released set has no price history, so the return model has nothing to
work with. The cold-start scorer (`src/models/train_coldstart.py`) works from
card art plus static metadata only.

**1. Price decomposition** (fitted on training sets only):

```
log_price = rarity_set_baseline + character_equity + art_premium + noise
```

`rarity_set_baseline` is the median log price by (rarity, variant, era).
`character_equity` is a per-species effect across all historical printings,
shrunk toward zero with empirical-Bayes weight `n/(n+k)` — the "Charizard is
popular" term, robust for species with few printings. `art_premium` is the
residual, and it is the **only** target the image model predicts: predicting
raw price from pixels would let the encoder cheat by reading rarity and foil
layout off the card frame.

**2. Embeddings.** Card art is cached to `data/images/` in two 336px crops
(full card, and an artwork-only crop that removes the frame and text box;
`--compare-crops` reports both). A frozen CLIP ViT-L/14 (open_clip) or DINOv2
encodes them; embeddings are cached to `data/embeddings/` and PCA-reduced to
64 dims **fitted per split**, with retained variance reported.

**3. Leakage control.** A linear probe predicts rarity from the embeddings
alone — its accuracy is reported below; high accuracy means the "visual"
signal is mostly a rarity detector. Each embedding dimension is then
residualised on (rarity, variant, artist, era), and results are reported for
raw and residualised ("style") embeddings side by side.

**4. Neighbour features** against the historical corpus, always excluding
same-set cards: similarity-weighted mean 90d return and art premium of the
k = 10/25/50 nearest visual neighbours, similarity to the top premium decile,
neighbour dispersion (confidence), and a novelty score (distance to the
nearest neighbour — high novelty = extrapolation, flagged low-confidence).

**5. Validation** holds out **entire recent sets** (released after a cutoff;
default: newest 20% of sets), simulating the real use case of scoring a new
set at release. With enough snapshot history the target is the realised 90d
forward return; with a short history (e.g. a single snapshot) the trainer
automatically falls back to **premium-only mode**, where the held-out-set
target is today's cross-sectional art premium instead - answerable from day
one, and clearly labelled via `mode` in `coldstart_metrics.json`. The
return-based evaluation switches on automatically once snapshots span the
horizon. Fill this table from `reports/coldstart_metrics.json`:

| Feature set | Rank IC (held-out sets, 90d return) |
|---|---|
| character_equity alone ("buy Charizard") | . |
| tabular only | . |
| tabular + character equity | . |
| + raw image embeddings | . |
| + residualised style features | . |
| + neighbour features | . |

Rarity-probe accuracy: raw `.` → residualised `.` · PCA variance retained: `.`

Beating the character-equity baseline is the whole point of the visual model.
If the image features do not improve held-out-set performance on your data,
that is the result — report it plainly.

**Dashboard:** the *New Set Scanner* tab scores any set, showing each card's
predicted art premium, its 5 nearest visual neighbours as thumbnails with
their historical returns, and a confidence badge driven by novelty and
neighbour dispersion — the model's reasoning is visually inspectable.

```bash
make images     # download + crop card art
make embed      # CLIP embeddings for both crops (torch; slow on CPU)
make coldstart  # decomposition, probes, ablation, premium model
```

## Limitations

- **Survivorship / coverage bias.** The API only lists cards it tracks and
  prices only where TCGPlayer has a market — dead or delisted products vanish,
  inflating apparent returns.
- **Thin liquidity.** Many "market prices" come from a handful of sales; the
  liquidity weight mitigates but does not fix this.
- **Grading not captured.** PSA/BGS-graded cards trade in a different market
  at very different prices; we model raw near-mint prices only.
- **Hype regime shifts.** Set rotations, influencer attention, and reprint
  announcements move prices in ways no historical feature anticipates. A model
  validated on a calm period will underestimate risk in a mania.
- **Short price history.** Task B is only as good as the snapshot history you
  collect; with weeks of data the confidence intervals dwarf the signal.
- **Visual leakage is only mitigated, not eliminated.** Residualising on
  rarity/variant/artist/era removes the linear part of what the encoder reads
  off the card frame; nonlinear traces (foil texture, border style) can
  remain. Treat the ablation deltas, not the absolute IC, as the evidence.
- **Novelty means extrapolation.** For genuinely new art styles the nearest
  neighbours are far away and the neighbour features are noise — that is what
  the novelty flag is for.
