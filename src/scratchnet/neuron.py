"""A single neuron (logistic regression), written in Andrew Ng's notation.

Examples go in columns: X has shape (n_x, m).
"""

import numpy as np

from scratchnet.activations import sigmoid


def initialize(n_x):
    """Starts w as a column of zeros and b at 0. Zeros are fine with one neuron."""
    w = np.zeros((n_x, 1))
    b = 0.0
    return w, b


def forward(w, b, X):
    """Turns every customer (each column of X) into a default probability."""
    Z = w.T @ X + b
    A = sigmoid(Z)
    return A


def compute_cost(A, Y):
    """Average log loss over all customers. Lower means better predictions.

    I keep A a tiny bit away from 0 and 1 so np.log() never sees a 0.
    """
    m = Y.shape[1]
    A = np.clip(A, 1e-15, 1 - 1e-15)
    J = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    return float(J)


def backward(X, A, Y):
    """Works out how the cost changes with each weight and with the bias.

    dZ is just each customer's error (prediction minus truth). The weight
    gradients are the average of feature times error, all done with one @.
    """
    m = X.shape[1]
    dZ = A - Y
    dw = (X @ dZ.T) / m
    db = float(np.sum(dZ) / m)
    return dw, db


def train(X, Y, learning_rate=0.1, num_iterations=1000, record_every=10):
    """Runs gradient descent and gives back the learned w, b and the cost history.

    Each loop: predict, measure the cost, work out the gradients, take a step.
    I save the cost every few steps so I can plot whether it's going down.
    """
    w, b = initialize(X.shape[0])
    costs = []
    for i in range(num_iterations):
        A = forward(w, b, X)
        dw, db = backward(X, A, Y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % record_every == 0:
            costs.append(compute_cost(A, Y))
    return w, b, costs
