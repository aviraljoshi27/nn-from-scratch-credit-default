"""Activation functions used by the neuron and, later, the network layers."""

import numpy as np


def sigmoid(z):
    """Squashes every value in z into the range 0 to 1.

    I clip z to between -500 and 500 first, because np.exp() overflows
    on really big numbers (e^710 is already too big for a float64).
    The answer barely changes, since sigmoid is basically 0 or 1 out there.
    """
    z = np.clip(z, -700, 700)
    return 1 / (1 + np.exp(-z))
