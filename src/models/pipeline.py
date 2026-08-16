"""Shared per-fold pipeline components.

CategoricalCaster lives here (not in train.py) so that pickled model bundles
reference a stable module path: a class defined in a script run via
`python -m src.models.train` pickles as `__main__.CategoricalCaster` and can
then only be unpickled from that same entry point.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
            # Mask unseen values to NaN before casting: pandas 4 deprecates
            # (and will forbid) constructing a Categorical from values
            # outside the given categories.
            masked = X[c].where(X[c].isin(self.categories_[c]))
            X[c] = pd.Categorical(masked, categories=self.categories_[c])
        return X
