"""Checks that TensorFlow can find my GPU and really uses it for maths.
I run this first in every new setup, before writing any model code."""

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow version:", tf.__version__)
print("GPUs found:", gpus)
if not gpus:
    raise SystemExit("No GPU found, so TensorFlow would run everything on the CPU.")

a = tf.random.normal((1000, 1000))
b = tf.random.normal((1000, 1000))
c = a @ b

print("Result shape:", c.shape)
print("Result dtype:", c.dtype)
print("Computed on:", c.device)
