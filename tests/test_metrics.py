"""Known-value checks for my scoring function, using the example I did by hand."""

import numpy as np

from scratchnet.metrics import evaluate


def test_evaluate_matches_my_hand_calculation():
    """The A, B, C, D example gave ROC-AUC 0.75 and PR-AUC 5/6 when I worked it out."""
    scores = evaluate(np.array([1, 1, 0, 0]), np.array([0.8, 0.4, 0.3, 0.6]))

    assert np.isclose(scores["roc_auc"], 0.75)
    assert np.isclose(scores["pr_auc"], 5 / 6)
    assert np.isclose(scores["f1"], 0.5)
    assert np.isclose(scores["log_loss"], -np.mean(np.log([0.8, 0.4, 0.7, 0.4])))
    assert np.array_equal(scores["confusion"], [[1, 1], [1, 1]])
