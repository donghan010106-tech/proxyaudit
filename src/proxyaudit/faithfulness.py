"""Explanation FAITHFULNESS (the 'faith' of the Fair-Faith-Trust triad).

We use the standard erasure-based faithfulness metrics (DeYoung et al., 2020):

  * Comprehensiveness: how much the score drops when the top-k most important
        features (by |phi|) are removed. High = the explanation captures the
        features the model truly relies on.
  * Sufficiency: how much the score is retained when ONLY the top-k features
        are kept. Low residual gap = the top features are sufficient.

A trustworthy fairness intervention should *preserve* faithfulness on the
legitimate (non-proxy) features: the model should keep explaining itself through
competencies, not through laundered proxies.
"""
from __future__ import annotations
import numpy as np


def _score(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(np.asarray(X))
    p = model.predict_proba(np.asarray(X))[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def comprehensiveness(model, X, phi, k_frac=0.3, background_mean=None):
    """Mean absolute score change when removing the top-k|phi| features."""
    X = np.asarray(X, dtype=float)
    phi = np.asarray(phi)
    n, d = X.shape
    k = max(1, int(round(k_frac * d)))
    if background_mean is None:
        background_mean = X.mean(axis=0)
    base = _score(model, X)
    Xr = X.copy()
    for i in range(n):
        top = np.argsort(-np.abs(phi[i]))[:k]
        Xr[i, top] = background_mean[top]
    pert = _score(model, Xr)
    return float(np.mean(np.abs(base - pert)))


def sufficiency(model, X, phi, k_frac=0.3, background_mean=None):
    """Mean absolute score change when keeping ONLY the top-k|phi| features."""
    X = np.asarray(X, dtype=float)
    phi = np.asarray(phi)
    n, d = X.shape
    k = max(1, int(round(k_frac * d)))
    if background_mean is None:
        background_mean = X.mean(axis=0)
    base = _score(model, X)
    Xs = np.tile(background_mean, (n, 1)).astype(float)
    for i in range(n):
        top = np.argsort(-np.abs(phi[i]))[:k]
        Xs[i, top] = X[i, top]
    pert = _score(model, Xs)
    return float(np.mean(np.abs(base - pert)))


def faithfulness_report(model, X, phi, k_frac=0.3, background_mean=None):
    comp = comprehensiveness(model, X, phi, k_frac, background_mean)
    suff = sufficiency(model, X, phi, k_frac, background_mean)
    # normalised faithfulness in [0,1]: high comprehensiveness, high sufficiency
    denom = comp + 1e-9
    return {"comprehensiveness": comp, "sufficiency": suff,
            "suff_ratio": float(suff / denom)}
