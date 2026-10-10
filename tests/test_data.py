"""Tests for the train/dev/test split and the preprocessing."""

import numpy as np

from scratchnet.data import (
    TARGET,
    load_raw,
    make_preprocessor,
    merge_unknown_codes,
    split,
    to_xy,
)


def test_split_keeps_everyone_once_and_keeps_balance():
    """Nobody is lost, nobody is in two sets, and every set keeps about 22% defaulters."""
    df = merge_unknown_codes(load_raw())
    train, dev, test = split(df)

    assert len(train) + len(dev) + len(test) == len(df)

    train_ids, dev_ids, test_ids = set(train.index), set(dev.index), set(test.index)
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids.isdisjoint(test_ids)
    assert dev_ids.isdisjoint(test_ids)

    overall = df[TARGET].mean()
    for part in (train, dev, test):
        assert abs(part[TARGET].mean() - overall) < 0.005


def test_preprocessor_is_fitted_on_train_only():
    """Train's scaled columns end up with mean 0 and spread 1, and every set gets 29 columns."""
    train, dev, _ = split(merge_unknown_codes(load_raw()))
    pre = make_preprocessor(train)
    X_train, _ = to_xy(pre, train)
    X_dev, _ = to_xy(pre, dev)

    assert X_train.shape[1] == X_dev.shape[1] == 29

    scaled = X_train[:, 9:]
    assert np.allclose(scaled.mean(axis=0), 0)
    assert np.allclose(scaled.std(axis=0), 1)
