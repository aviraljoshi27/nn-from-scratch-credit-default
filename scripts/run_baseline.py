"""Trains sklearn's LogisticRegression as my baseline and scores it on the dev set."""

import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from scratchnet.data import (
    load_raw,
    make_preprocessor,
    merge_unknown_codes,
    split,
    to_xy,
)
from scratchnet.metrics import evaluate

train, dev, _ = split(merge_unknown_codes(load_raw()))
pre = make_preprocessor(train)
X_train, y_train = to_xy(pre, train)
X_dev, y_dev = to_xy(pre, dev)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
scores = evaluate(y_dev, model.predict_proba(X_dev)[:, 1])
print(scores)

scores["confusion"] = scores["confusion"].tolist()
out = Path(__file__).resolve().parents[1] / "results" / "baseline_dev.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(scores, indent=2))
