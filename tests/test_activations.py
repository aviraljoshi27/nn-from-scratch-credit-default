"""Tests for the activation functions."""

import numpy as np

from scratchnet.activations import sigmoid


def test_sigmoid_known_values():
    """sigmoid(0) should be 0.5, and the output should keep the input's shape."""
    z = np.array([[0.0, 2.0, -2.0]])
    a = sigmoid(z)
    assert a.shape == (1, 3)
    assert np.allclose(a, [[0.5, 0.880797, 0.119203]])


def test_sigmoid_no_overflow():
    """Huge inputs shouldn't overflow, just give 0 and 1."""
    with np.errstate(over="raise"):
        a = sigmoid(np.array([-1000.0, 1000.0]))
    assert np.allclose(a, [0.0, 1.0])
