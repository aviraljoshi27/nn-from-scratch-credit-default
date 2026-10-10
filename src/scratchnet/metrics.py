"""Scores I use for every model, so they're all judged the same way."""

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


def evaluate(y_true, p, threshold=0.5):
    """Gives back all my scores for one model's predicted probabilities.

    ROC-AUC is the main one. F1 and the confusion matrix need yes/no
    decisions, so I flag everyone whose number is at or above the threshold.
    """
    flagged = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "f1": float(f1_score(y_true, flagged)),
        "log_loss": float(log_loss(y_true, p)),
        "confusion": confusion_matrix(y_true, flagged),
    }
