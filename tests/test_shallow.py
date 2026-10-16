import numpy as np

from scratchnet.shallow import forward, initialize


def test_initialize_shapes_and_zero_biases():
    params = initialize(n_x=4, n_h=5)
    assert params["W1"].shape == (5, 4)
    assert params["b1"].shape == (5, 1)
    assert params["W2"].shape == (1, 5)
    assert params["b2"].shape == (1, 1)
    assert np.all(params["b1"] == 0) and np.all(params["b2"] == 0)


def test_initialize_rows_differ_and_seed_repeats():
    W1 = initialize(n_x=4, n_h=5, seed=7)["W1"]
    assert len(np.unique(W1, axis=0)) == 5
    assert np.array_equal(W1, initialize(n_x=4, n_h=5, seed=7)["W1"])


def test_forward_shapes():
    A2, cache = forward(initialize(n_x=4, n_h=5), np.ones((4, 7)))
    assert A2.shape == (1, 7)
    assert cache["Z1"].shape == (5, 7) and cache["A1"].shape == (5, 7)
    assert cache["Z2"].shape == (1, 7)


def test_forward_matches_hand_calculation():
    params = {
        "W1": np.array([[0.2, 0.4], [-0.5, 0.1], [0.3, -0.2]]),
        "b1": np.array([[0.1], [0.0], [-0.1]]),
        "W2": np.array([[0.6, -0.4, 0.8]]),
        "b2": np.array([[0.05]]),
    }
    A2, cache = forward(params, np.array([[0.5], [-1.0]]))
    assert np.allclose(cache["Z1"], [[-0.2], [-0.35], [0.25]])
    assert np.allclose(A2, [[0.565143]])
