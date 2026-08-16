"""Cross-validation splitters.

Task A uses sklearn's GroupKFold grouped by set_id (imported directly in
train.py). Task B needs strictly time-ordered splits with a purge and an
embargo, implemented here.

Why purge + embargo: the Task B target at snapshot date t is
log(P_{t+h} / P_t), i.e. the *label* of a training row dated t reaches h days
into the future. If validation starts at date v, any training row with
t + h >= v has a label that overlaps the validation period - that is look-ahead
leakage. So we purge every training row within h days of the validation start,
and add an extra embargo on top to absorb snapshot-date jitter and serial
correlation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class PurgedExpandingTimeSplit:
    """Expanding-window time series split with purge and embargo.

    Parameters
    ----------
    n_splits : number of folds.
    horizon_days : label horizon h; training rows within h days of the
        validation start are purged.
    embargo_days : extra gap on top of the purge.
    min_train_days : minimum span of the first training window.
    """

    n_splits: int
    horizon_days: int
    embargo_days: int = 0
    min_train_days: int = 0

    def split(
        self, dates: pd.Series
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, val_idx) positional index arrays.

        `dates` is the snapshot date of every row (unsorted is fine).
        Validation windows are consecutive, non-overlapping blocks of unique
        dates at the end of the timeline; training always uses only dates
        strictly before (val_start - horizon - embargo).
        """
        dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        unique_dates = np.sort(dates.unique())
        if len(unique_dates) < self.n_splits + 1:
            raise ValueError(
                f"Need more unique dates ({len(unique_dates)}) than "
                f"splits ({self.n_splits})"
            )

        first_allowed = unique_dates[0] + pd.Timedelta(days=self.min_train_days)
        eligible_val_dates = unique_dates[unique_dates > first_allowed]
        if len(eligible_val_dates) < self.n_splits:
            raise ValueError(
                "min_train_days leaves too few dates for validation"
            )

        folds = np.array_split(eligible_val_dates, self.n_splits)
        gap = pd.Timedelta(days=self.horizon_days + self.embargo_days)
        for fold_dates in folds:
            if len(fold_dates) == 0:
                continue
            val_start = pd.Timestamp(fold_dates[0])
            val_end = pd.Timestamp(fold_dates[-1])
            train_mask = dates < (val_start - gap)
            val_mask = (dates >= val_start) & (dates <= val_end)
            train_idx = np.flatnonzero(train_mask.to_numpy())
            val_idx = np.flatnonzero(val_mask.to_numpy())
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            yield train_idx, val_idx
