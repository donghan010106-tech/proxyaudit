"""TRUST of predictions (the 'trust' of the Fair-Faith-Trust triad).

Trust is operationalised as:
  * Calibration error (ECE) and Brier score -- can the probabilities be believed?
  * Decision stability -- of the candidates whose outcome is *legitimately*
        unchanged (i.e. not in the protected-disadvantaged cell), how many keep
        their decision after the fairness intervention? A trustworthy fix should
        flip mostly the cases that were unfairly decided, not churn everyone.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true, p, n_bins=10):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        m = (p >= bins[b]) & (p < bins[b + 1] if b < n_bins - 1 else p <= bins[b + 1])
        if m.any():
            conf = p[m].mean()
            acc = y_true[m].mean()
            ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def trust_report(y_true, p, p_before=None, pred_before=None, pred_after=None):
    rep = {
        "ECE": expected_calibration_error(y_true, p),
        "Brier": float(brier_score_loss(y_true, p)),
    }
    if pred_before is not None and pred_after is not None:
        flip = float(np.mean(pred_before != pred_after))
        rep["decision_flip_rate"] = flip
        rep["decision_stability"] = 1.0 - flip
    if p_before is not None:
        rep["mean_abs_prob_shift"] = float(np.mean(np.abs(np.asarray(p) - np.asarray(p_before))))
    return rep
