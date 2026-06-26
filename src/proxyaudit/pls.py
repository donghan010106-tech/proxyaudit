"""Proxy-Channel Localization: the core novelty of ProxyAudit.

Proposition 1 (exact demographic-parity decomposition in score space).
  Let f be any model whose score (logit) admits an additive SHAP explanation
  with local accuracy:  f(x) = phi_0 + sum_j phi_j(x).  Taking expectations
  conditional on the protected attribute A,

      E[f | A=1] - E[f | A=0] = sum_j ( E[phi_j | A=1] - E[phi_j | A=0] )
                              = sum_j  Delta_j .

  The left-hand side is the demographic-parity gap of the *score*; therefore
  Delta_j is the EXACT contribution of feature j to that gap. Features with
  large |Delta_j| are the channels through which A reaches Yhat.

This turns the qualitative claim of FairCVtest (CVPRW 2020) and FairJob
(NeurIPS 2024) -- "the model learns A -> X -> Yhat" -- into a quantitative,
per-feature attribution.

Proxy Leakage Score (PLS).
  A feature is a *proxy* if it (a) carries information about A (reconstructs A)
  AND (b) drives the DP gap. We combine:
      R_j  = reconstructability   = 2*|AUC(x_j -> A) - 0.5|        in [0,1]
      D_j  = normalised DP contribution = |Delta_j| / sum_k|Delta_k|
  and define
      PLS_j = sqrt( R_j * D_j )      (geometric mean; high only if BOTH high)
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score


def dp_score_decomposition(phi, A):
    """Delta_j = E[phi_j|A=1] - E[phi_j|A=0] for each feature.

    Returns (delta, dp_score_gap) where sum(delta) == dp_score_gap exactly
    (up to the SHAP estimator's local-accuracy error)."""
    phi = np.asarray(phi)
    m1, m0 = (A == 1), (A == 0)
    delta = phi[m1].mean(axis=0) - phi[m0].mean(axis=0)
    return delta, float(delta.sum())


def reconstructability(X, A):
    """R_j = 2*|AUC(x_j -> A) - 0.5| per feature (univariate separability of A)."""
    X = np.asarray(X, dtype=float)
    R = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        try:
            auc = roc_auc_score(A, X[:, j])
        except Exception:
            auc = 0.5
        R[j] = 2 * abs(auc - 0.5)
    return R


def proxy_leakage_score(X, A, phi, feature_names=None, eps=1e-12,
                        X_recon=None, A_recon=None):
    """Compute PLS_j and the supporting quantities.

    Reconstructability R_j is a property of the marginal x_j -> A and is
    independent of the SHAP explanation, so it may be estimated on a larger
    sample (`X_recon`, `A_recon`) than the explanation subset used for the
    Shapley-based DP contribution Delta_j. When `X_recon` is None it falls back
    to the explanation subset `X`.

    Returns a dict with arrays aligned to feature order plus a ranked table.
    """
    R = reconstructability(X if X_recon is None else X_recon,
                           A if A_recon is None else A_recon)
    delta, dp_gap = dp_score_decomposition(phi, A)
    absdelta = np.abs(delta)
    D = absdelta / (absdelta.sum() + eps)            # normalised DP contribution
    PLS = np.sqrt(np.clip(R, 0, 1) * np.clip(D, 0, 1))

    order = np.argsort(-PLS)
    names = (list(feature_names) if feature_names is not None
             else [f"f{j}" for j in range(len(PLS))])
    table = [{
        "feature": names[j], "PLS": float(PLS[j]),
        "reconstructability": float(R[j]),
        "dp_contribution": float(delta[j]),
        "dp_contribution_abs": float(absdelta[j]),
        "dp_share": float(D[j]),
        "rank": int(r),
    } for r, j in enumerate(order)]
    return {
        "PLS": PLS, "R": R, "delta": delta, "D": D,
        "dp_score_gap": dp_gap, "order": order,
        "names": names, "table": table,
    }


def _knee_index(s):
    """Robust knee of a descending curve `s` via maximum distance to the chord.

    The chord joins (0, s[0]) and (n-1, s[n-1]); the knee is the index of the
    point lying farthest *below* that chord. Unlike a max-consecutive-gap rule,
    this looks at the global shape of the curve, so a dip *within* the leaking
    block (some proxy channels leak more than others) does not trigger an early
    cut. Returns the number of leading points to keep (>=1).
    """
    s = np.asarray(s, dtype=float)
    n = len(s)
    if n < 3 or s[0] <= 0:
        return 1
    x = np.arange(n, dtype=float)
    x0, y0, x1, y1 = 0.0, s[0], float(n - 1), s[-1]
    denom = np.hypot(y1 - y0, x1 - x0) + 1e-12
    # signed perpendicular distance of each point to the chord
    dist = ((y1 - y0) * x - (x1 - x0) * s + x1 * y0 - y1 * x0) / denom
    knee = int(np.argmax(dist))           # point of maximum curvature
    return max(1, knee + 1)


def select_proxies(pls_result, k=None, tau=None, recon_gate=0.10):
    """Choose the leaking features to neutralise.

    * if k given: top-k by PLS.
    * if tau given: all features with PLS >= tau.
    * default: robust distance-to-chord knee on the sorted PLS curve, then a
      light reconstructability gate (R_j >= recon_gate) so that features which
      do not actually carry information about A are never neutralised.
    """
    PLS = pls_result["PLS"]
    order = pls_result["order"]
    names = pls_result["names"]
    R = pls_result["R"]
    if k is not None:
        chosen = order[:k]
    elif tau is not None:
        chosen = np.where(PLS >= tau)[0]
    else:
        keep = _knee_index(PLS[order])
        cand = order[:keep]
        # gate: must carry some information about A to be a genuine proxy
        cand = [j for j in cand if R[j] >= recon_gate]
        chosen = np.asarray(cand if cand else order[:1], dtype=int)
    return [names[j] for j in chosen], list(np.asarray(chosen, dtype=int))


def localization_quality(pls_result, proxy_truth):
    """If ground-truth proxy indices are known, score the localisation.

    Returns precision@k (k = #true proxies), AUC of PLS vs truth, and the
    average-precision. Used on the FairCVtest-recipe testbed.
    """
    names = pls_result["names"]
    truth = np.array([1 if proxy_truth.get(n, False) else 0 for n in names])
    if truth.sum() == 0:
        return {}
    PLS = pls_result["PLS"]
    k = int(truth.sum())
    topk = pls_result["order"][:k]
    prec_at_k = float(truth[topk].mean())
    try:
        auc = float(roc_auc_score(truth, PLS))
    except Exception:
        auc = float("nan")
    # average precision
    order = pls_result["order"]
    tp = 0
    aps = []
    for i, j in enumerate(order, 1):
        if truth[j]:
            tp += 1
            aps.append(tp / i)
    ap = float(np.mean(aps)) if aps else 0.0
    return {"precision_at_k": prec_at_k, "pls_auc": auc,
            "average_precision": ap, "n_true_proxies": k}
