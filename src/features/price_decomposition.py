"""Additive log-price decomposition for the cold-start model.

    log_price = rarity_set_baseline + character_equity + art_premium + noise

- rarity_set_baseline: median log price by (rarity, variant, era), where era
  is the set's series. Fallbacks: (rarity, variant) -> rarity -> global.
- character_equity: per-species mean of the baseline residual across ALL of
  that species' historical printings, shrunk toward zero (the global residual
  mean is ~0 by construction) with empirical-Bayes / James-Stein weight
  n / (n + k). One noisy printing of an obscure Pokemon gets pulled to zero;
  Charizard, with dozens of printings, keeps its full premium.
- art_premium: whatever is left. This residual is the ONLY target the image
  model is allowed to predict - predicting raw price from pixels would let
  the encoder cheat by reading rarity and foil layout off the card frame.

Everything is fitted on training rows only (`fit`) and applied to any rows
(`transform`), so held-out sets never leak into the tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def species_key(df: pd.DataFrame) -> pd.Series:
    """Stable per-character key: pokedex number when present (ties every
    printing of a species together), otherwise the card name (trainers etc.)."""
    dex = pd.to_numeric(df["pokedex_number"], errors="coerce")
    return dex.astype("Int64").astype(str).where(dex.notna(), df["name"])


@dataclass
class PriceDecomposer:
    shrinkage_k: float = 5.0

    baseline_full_: pd.Series = field(default=None, repr=False)
    baseline_rv_: pd.Series = field(default=None, repr=False)
    baseline_rarity_: pd.Series = field(default=None, repr=False)
    baseline_global_: float = field(default=None, repr=False)
    equity_: pd.Series = field(default=None, repr=False)

    def fit(self, df: pd.DataFrame) -> "PriceDecomposer":
        """`df` needs log_price, rarity, variant, series, pokedex_number, name."""
        d = df.copy()
        d["era"] = d["series"].fillna("unknown")
        d["rarity"] = d["rarity"].fillna("unknown")

        self.baseline_full_ = d.groupby(["rarity", "variant", "era"],
                                        observed=True)["log_price"].median()
        self.baseline_rv_ = d.groupby(["rarity", "variant"],
                                      observed=True)["log_price"].median()
        self.baseline_rarity_ = d.groupby("rarity",
                                          observed=True)["log_price"].median()
        self.baseline_global_ = float(d["log_price"].median())

        resid = d["log_price"] - self._baseline(d)
        key = species_key(d)
        grp = resid.groupby(key)
        n = grp.size().astype(float)
        raw_mean = grp.mean()
        # James-Stein style shrinkage toward the global residual mean (~0):
        # equity = mean * n / (n + k). Few printings -> heavy shrinkage.
        self.equity_ = raw_mean * n / (n + self.shrinkage_k)
        return self

    def _baseline(self, df: pd.DataFrame) -> pd.Series:
        d = df.copy()
        d["era"] = d["series"].fillna("unknown")
        d["rarity"] = d["rarity"].fillna("unknown")
        full_keys = pd.MultiIndex.from_frame(d[["rarity", "variant", "era"]])
        rv_keys = pd.MultiIndex.from_frame(d[["rarity", "variant"]])
        base = pd.Series(self.baseline_full_.reindex(full_keys).to_numpy(),
                         index=d.index)
        base = base.fillna(pd.Series(
            self.baseline_rv_.reindex(rv_keys).to_numpy(), index=d.index))
        base = base.fillna(pd.Series(
            self.baseline_rarity_.reindex(d["rarity"]).to_numpy(), index=d.index))
        return base.fillna(self.baseline_global_)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the three components (and art_premium where log_price exists).

        Unseen species get zero character equity - exactly the honest prior
        for a brand-new character."""
        if self.equity_ is None:
            raise RuntimeError("PriceDecomposer must be fitted first")
        out = pd.DataFrame(index=df.index)
        out["rarity_set_baseline"] = self._baseline(df)
        out["character_equity"] = (
            self.equity_.reindex(species_key(df)).fillna(0.0).to_numpy()
        )
        if "log_price" in df.columns:
            out["art_premium"] = (df["log_price"] - out["rarity_set_baseline"]
                                  - out["character_equity"])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
