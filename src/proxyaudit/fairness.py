"""Group-fairness metrics, aligned with the FairCVtest and FairJob baselines."""
from __future__ import annotations
import numpy as np


def positive_rate(y_pred, A, a):
    m = (A == a)
    return float(y_pred[m].mean()) if m.any() else 0.0


def demographic_parity_gap(y_pred, A):
    """|P(Yhat=1|A=1) - P(Yhat=1|A=0)|  (FairJob DP, FairCVtest DP gap)."""
    return abs(positive_rate(y_pred, A, 1) - positive_rate(y_pred, A, 0))


def disparate_impact(y_pred, A):
    """min/max positive-rate ratio (the EEOC four-fifths rule; <0.8 = violation)."""
    r0 = positive_rate(y_pred, A, 0)
    r1 = positive_rate(y_pred, A, 1)
    lo, hi = min(r0, r1), max(r0, r1)
    return float(lo / hi) if hi > 0 else 1.0


def equal_opportunity_gap(y_true, y_pred, A):
    """|TPR(A=1) - TPR(A=0)|."""
    def tpr(a):
        m = (A == a) & (y_true == 1)
        return float(y_pred[m].mean()) if m.any() else 0.0
    return abs(tpr(1) - tpr(0))


def fairness_report(y_true, y_pred, A):
    return {
        "DP_gap": demographic_parity_gap(y_pred, A),
        "EOO_gap": equal_opportunity_gap(y_true, y_pred, A),
        "DI": disparate_impact(y_pred, A),
        "pos_rate_A0": positive_rate(y_pred, A, 0),
        "pos_rate_A1": positive_rate(y_pred, A, 1),
    }


def conditional_dp_gap(y_pred, A, cond_mask):
    """DP gap restricted to a subgroup (e.g. senior==1 in FairJob)."""
    yp, AA = y_pred[cond_mask], A[cond_mask]
    return demographic_parity_gap(yp, AA)
