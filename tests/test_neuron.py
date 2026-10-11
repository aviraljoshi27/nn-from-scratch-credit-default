"""Tests for the single neuron."""

import numpy as np
from sklearn.metrics import log_loss

from scratchnet.neuron import backward, compute_cost, forward, initialize, train


def test_initialize_shapes():
    """w should be a (n_x, 1) column of zeros and b should be 0."""
    w, b = initialize(4)
    assert w.shape == (4, 1)
    assert np.all(w == 0) and b == 0.0


def test_forward_shape_and_values():
    """Two customers in, two probabilities out, matching values worked out by hand."""
    w = np.array([[0.5], [1.0]])
    X = np.array([[2.0, 1.0], [-1.0, 1.0]])
    A = forward(w, 0.1, X)
    assert A.shape == (1, 2)
    assert np.allclose(A, [[0.524979, 0.832018]])


def test_cost_matches_hand_value_and_sklearn():
    """Cost for two customers, checked by hand and against sklearn's log_loss."""
    A = np.array([[0.8, 0.4]])
    Y = np.array([[1, 0]])
    J = compute_cost(A, Y)
    assert np.isclose(J, 0.366985)
    assert np.isclose(J, log_loss(Y.ravel(), A.ravel()))


def test_cost_stays_finite_at_0_and_1():
    """Predictions of exactly 0 or 1 shouldn't give inf or nan."""
    J = compute_cost(np.array([[1.0, 0.0]]), np.array([[0, 1]]))
    assert np.isfinite(J)


def test_backward_known_values():
    """Same 2-feature, 3-customer example I worked through by hand."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    A = np.array([[0.1, 0.8, 0.3]])
    Y = np.array([[0, 1, 0]])
    dw, db = backward(X, A, Y)
    assert dw.shape == (2, 1)
    assert np.allclose(dw, [[0.2], [0.4]])
    assert np.isclose(db, 0.2 / 3)


def test_train_lowers_cost_and_fits_tiny_data():
    """On 4 easy customers the cost should drop and every prediction should end up right."""
    X = np.array([[-2.0, -1.0, 1.0, 2.0]])
    Y = np.array([[0, 0, 1, 1]])
    w, b, costs = train(X, Y, learning_rate=1.0, num_iterations=500)
    assert costs[-1] < costs[0]
    predictions = (forward(w, b, X) > 0.5).astype(int)
    assert np.array_equal(predictions, Y)
