import numpy as np
import pandas as pd
import pytest

from src.models.splits import PurgedExpandingTimeSplit


@pytest.fixture
def daily_dates():
    """300 daily snapshot dates, 3 rows per date (like 3 cards)."""
    dates = pd.date_range("2025-01-01", periods=300, freq="D")
    return pd.Series(np.repeat(dates, 3))


def _folds(dates, **kwargs):
    defaults = dict(n_splits=4, horizon_days=30, embargo_days=5,
                    min_train_days=60)
    defaults.update(kwargs)
    return list(PurgedExpandingTimeSplit(**defaults).split(dates)), defaults


def test_no_index_overlap(daily_dates):
    folds, _ = _folds(daily_dates)
    assert len(folds) > 0
    for train_idx, val_idx in folds:
        assert len(np.intersect1d(train_idx, val_idx)) == 0


def test_train_strictly_before_validation(daily_dates):
    folds, _ = _folds(daily_dates)
    for train_idx, val_idx in folds:
        assert daily_dates.iloc[train_idx].max() < daily_dates.iloc[val_idx].min()


def test_purge_and_embargo_gap(daily_dates):
    folds, params = _folds(daily_dates)
    gap = pd.Timedelta(days=params["horizon_days"] + params["embargo_days"])
    for train_idx, val_idx in folds:
        train_end = daily_dates.iloc[train_idx].max()
        val_start = daily_dates.iloc[val_idx].min()
        # Every training label (t + horizon) must end before validation
        # starts, with the embargo on top: train_end < val_start - gap.
        assert train_end < val_start - gap + pd.Timedelta(days=1)
        assert (val_start - train_end).days >= params["horizon_days"] \
            + params["embargo_days"]


def test_expanding_window(daily_dates):
    folds, _ = _folds(daily_dates)
    train_sizes = [len(tr) for tr, _ in folds]
    val_starts = [daily_dates.iloc[va].min() for _, va in folds]
    assert train_sizes == sorted(train_sizes)          # training only grows
    assert val_starts == sorted(val_starts)            # folds move forward
    # Validation blocks must not overlap each other either.
    for (_, va1), (_, va2) in zip(folds, folds[1:]):
        assert daily_dates.iloc[va1].max() < daily_dates.iloc[va2].min()


def test_min_train_days_respected(daily_dates):
    folds, params = _folds(daily_dates)
    first_date = daily_dates.min()
    for _, val_idx in folds:
        val_start = daily_dates.iloc[val_idx].min()
        assert (val_start - first_date).days > params["min_train_days"]


def test_all_rows_of_a_date_stay_together(daily_dates):
    folds, _ = _folds(daily_dates)
    for train_idx, val_idx in folds:
        train_dates = set(daily_dates.iloc[train_idx])
        val_dates = set(daily_dates.iloc[val_idx])
        assert train_dates.isdisjoint(val_dates)


def test_unsorted_input_supported(daily_dates):
    shuffled = daily_dates.sample(frac=1.0, random_state=0).reset_index(drop=True)
    folds, _ = _folds(shuffled)
    for train_idx, val_idx in folds:
        assert shuffled.iloc[train_idx].max() < shuffled.iloc[val_idx].min()


def test_too_few_dates_raises():
    dates = pd.Series(pd.date_range("2025-01-01", periods=3, freq="D"))
    with pytest.raises(ValueError):
        list(PurgedExpandingTimeSplit(n_splits=4, horizon_days=30).split(dates))
