"""Tests for the train/dev/test split."""

from scratchnet.data import TARGET, load_raw, merge_unknown_codes, split


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
