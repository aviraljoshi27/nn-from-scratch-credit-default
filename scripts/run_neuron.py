"""Trains my NumPy neuron on the training set and scores it on the dev set."""

import json
from pathlib import Path

from scratchnet.data import (
    load_raw,
    make_preprocessor,
    merge_unknown_codes,
    split,
    to_xy,
)
from scratchnet.metrics import evaluate
from scratchnet.neuron import forward, train

LEARNING_RATE = 0.5
NUM_ITERATIONS = 2000

train_df, dev_df, _ = split(merge_unknown_codes(load_raw()))
pre = make_preprocessor(train_df)
X_train, y_train = to_xy(pre, train_df)
X_dev, y_dev = to_xy(pre, dev_df)

w, b, costs = train(X_train.T, y_train.reshape(1, -1), LEARNING_RATE, NUM_ITERATIONS)
p_dev = forward(w, b, X_dev.T).ravel()
scores = evaluate(y_dev, p_dev)
print(scores)
print("first cost:", costs[0], "last cost:", costs[-1])

scores["confusion"] = scores["confusion"].tolist()
scores.update(learning_rate=LEARNING_RATE, num_iterations=NUM_ITERATIONS, costs=costs)
out = Path(__file__).resolve().parents[1] / "results" / "neuron_dev.json"
out.write_text(json.dumps(scores, indent=2))
