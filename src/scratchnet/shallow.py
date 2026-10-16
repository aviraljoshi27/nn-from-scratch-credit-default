"""My shallow network: one hidden layer (tanh) and one output neuron (sigmoid).

I kept this separate from neuron.py so the single neuron stays exactly how it
was when it matched sklearn.
"""

import numpy as np

from scratchnet.activations import sigmoid


def initialize(n_x, n_h, seed=42):
    """Makes the starting weights and biases and gives them back in one dict.

    The weights start as small random numbers so the hidden neurons don't all
    end up learning the same thing. The biases can just start at zero.
    Same seed means the same starting numbers every run.
    """
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((n_h, n_x)) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = rng.standard_normal((1, n_h)) * 0.01
    b2 = np.zeros((1, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(params, X):
    """Runs every customer through the network and gives back the predictions.

    X has one column per customer. I keep the in-between results in a cache
    because the backward pass needs A1 and A2 and I don't want to work them
    out twice.
    """
    Z1 = params["W1"] @ X + params["b1"]
    A1 = np.tanh(Z1)
    Z2 = params["W2"] @ A1 + params["b2"]
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def backward(params, cache, X, Y):
    """Works out how much each weight and bias should change to lower the cost.

    It walks back from the output: first how wrong we were, then layer 2's
    gradients, then the blame passed back through W2 and tanh, then layer 1's.
    Every gradient comes out the same shape as the thing it belongs to.
    """
    m = X.shape[1]
    A1, A2 = cache["A1"], cache["A2"]
    dZ2 = A2 - Y
    dW2 = dZ2 @ A1.T / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = (params["W2"].T @ dZ2) * (1 - np.power(A1, 2))
    dW1 = dZ1 @ X.T / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
