"""Mediator-group decomposition of the parity gap (NDE / NIE), and a contrast
between interventional and conditional Shapley.

The withheld attribute A reaches the score only through the observed features
(mediators) X. We split X into the localized proxy block P (illegitimate
mediators) and the rest R = X\\P (legitimate mediators). By Proposition 1 the
score-level parity gap decomposes as

    DP^f = sum_j Delta_j = (sum_{j in P} Delta_j) + (sum_{j in R} Delta_j)
         =        NIE_proxy            +            NDE_rest,

a Shapley reading of the natural indirect effect carried by the proxy path
(NIE_proxy) versus the effect along the remaining paths (NDE_rest). This is an
SCM-free, mediator-group attribution: it does not identify a full structural
model, but it isolates how much of the disparity flows through the audited
channel.

The estimator matters. *Interventional* Shapley attributes to each feature only
its own mechanistic contribution, so when a proxy is correlated with a
legitimate feature the proxy keeps its credit and the NIE/NDE split is faithful.
*Conditional* (observational) Shapley redistributes credit along correlations, so
proxy credit smears onto correlated legitimate features, biasing the split. We
quantify this contrast.
"""
from __future__ import annotations
import numpy as np
from .models import linear_part, transform_for_explainer
from .shapley import LinearSHAP, SamplingSHAP, GaussianConditionalSHAP
from .pls import dp_score_decomposition


def _attributions(model, Xexp, background, mode, seed=0, n_perm=48):
    """Return per-feature SHAP attributions (n_explain x d) in score space."""
    lin = linear_part(model)
    if mode == "interventional":
        if lin is not None:
            bg = transform_for_explainer(model, background)
            Xt = transform_for_explainer(model, Xexp)
            return LinearSHAP(lin, bg).shap_values(Xt)
        return SamplingSHAP(model, np.asarray(background), n_perm=n_perm, seed=seed
                            ).shap_values(np.asarray(Xexp))
    elif mode == "conditional":
        Xt = transform_for_explainer(model, Xexp) if lin is not None else np.asarray(Xexp)
        bg = transform_for_explainer(model, background) if lin is not None else np.asarray(background)
        return GaussianConditionalSHAP(model, bg, n_perm=n_perm, seed=seed).shap_values(Xt)
    raise ValueError(f"unknown mode {mode}")


def mediation_decomposition(model, Xexp, A_exp, background, proxies,
                            feature_names, mode="interventional", seed=0, n_perm=48):
    """NDE/NIE-style decomposition of the parity gap under a chosen Shapley mode."""
    phi = _attributions(model, Xexp, background, mode, seed=seed, n_perm=n_perm)
    delta, dp_gap = dp_score_decomposition(phi, A_exp)
    names = list(feature_names)
    pset = set(proxies)
    pidx = [i for i, nm in enumerate(names) if nm in pset]
    ridx = [i for i in range(len(names)) if i not in pidx]
    nie = float(np.sum(delta[pidx])) if pidx else 0.0
    nde = float(np.sum(delta[ridx])) if ridx else 0.0
    return {
        "mode": mode, "delta": delta, "dp_gap": dp_gap,
        "NIE_proxy": nie, "NDE_rest": nde,
        "NIE_share": float(nie / dp_gap) if abs(dp_gap) > 1e-9 else float("nan"),
        "names": names, "proxy_idx": pidx,
        "per_feature": {names[i]: float(delta[i]) for i in range(len(names))},
    }


def compare_modes(model, Xexp, A_exp, background, proxies, feature_names,
                  truth_proxies=None, seed=0, n_perm=48):
    """Run both estimators and quantify proxy-credit smearing.

    Returns each mode's NIE/NDE plus, if the ground-truth proxy set is known, the
    fraction of the gap each estimator concentrates on the *true* proxies and the
    leakage onto the top non-proxy (legitimate) feature.
    """
    res = {}
    for mode in ["interventional", "conditional"]:
        res[mode] = mediation_decomposition(
            model, Xexp, A_exp, background, proxies, feature_names,
            mode=mode, seed=seed, n_perm=n_perm)
    if truth_proxies is not None:
        names = res["interventional"]["names"]
        tset = set(truth_proxies)
        tidx = [i for i, nm in enumerate(names) if nm in tset]
        nidx = [i for i in range(len(names)) if i not in tidx]
        for mode in res:
            d = res[mode]["delta"]; total = np.sum(np.abs(d)) + 1e-12
            res[mode]["true_proxy_share"] = float(np.sum(np.abs(d[tidx])) / total)
            # biggest credit leaked onto a legitimate (non-proxy) feature
            leg = [(names[i], abs(float(d[i]))) for i in nidx]
            leg.sort(key=lambda t: -t[1])
            res[mode]["top_legit_leak"] = leg[0] if leg else (None, 0.0)
    return res
