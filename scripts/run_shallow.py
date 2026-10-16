"""Trains my shallow network (one hidden layer) on the training set and scores
it on the dev set. Same data steps as run_neuron.py so the two are fair to
compare."""

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
from scratchnet.shallow import forward, train

N_HIDDEN = 8
LEARNING_RATE = 0.5
NUM_ITERATIONS = 2000
SEED = 42

train_df, dev_df, _ = split(merge_unknown_codes(load_raw()))
pre = make_preprocessor(train_df)
X_train, y_train = to_xy(pre, train_df)
X_dev, y_dev = to_xy(pre, dev_df)

params, costs = train(
    X_train.T,
    y_train.reshape(1, -1),
    N_HIDDEN,
    learning_rate=LEARNING_RATE,
    num_iterations=NUM_ITERATIONS,
    seed=SEED,
)
A2_dev, _ = forward(params, X_dev.T)
p_dev = A2_dev.ravel()
scores = evaluate(y_dev, p_dev)
print(scores)
print("first cost:", costs[0], "last cost:", costs[-1])

scores["confusion"] = scores["confusion"].tolist()
scores.update(
    n_hidden=N_HIDDEN,
    learning_rate=LEARNING_RATE,
    num_iterations=NUM_ITERATIONS,
    seed=SEED,
    costs=costs,
)
out = Path(__file__).resolve().parents[1] / "results" / "shallow_dev.json"
out.write_text(json.dumps(scores, indent=2))
