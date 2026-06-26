"""Proxy-Cluster Counterfactuals (PCC): necessity, sufficiency, and causation.

The population audit (Proposition 1) tells us *how much* of the group-fairness
gap flows through the localized proxy cluster $P$. PCC asks the same question one
individual at a time and turns the answer into a per-decision certificate.

For a fixed model $f$ (the unaware model under audit) and the localized cluster
$P$, two complementary directed interventions are used.

*Necessity probe* -- neutralize the cluster, hold the rest fixed:

    x^CF = x  with block P replaced by its A-orthogonalized value.

If x was rejected and x^CF is accepted, the leaked cluster was *necessary* for
the rejection ("rejected because of the proxy channel").

*Sufficiency probe* -- keep the cluster, neutralize the rest to a baseline:

    x^SUF = (population-average profile)  with block P set to x's factual value.

If x^SUF reproduces x's (biased) decision, the cluster *alone* is sufficient to
drive the outcome ("the proxy signature by itself is enough").

A cluster that is both necessary and sufficient for an individual's decision is a
genuine per-decision *cause* of the unfair outcome -- a fully explainable
fairness certificate. We compute all three rates per group, plus the consistency
corollary linking the individual effect to the Proposition-1 cluster term.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .neutralize import neutralize
from .shapley import _logit_score


def _scores(model, X):
    return _logit_score(model, X)


def _decide(model, Xvals, thr):
    p = model.predict_proba(Xvals)[:, 1]
    return p, (p >= thr).astype(int)


def proxy_cluster_counterfactual(model, Xtr, Xte, Atr, Ate, proxies,
                                 select_rate, cluster_delta=None):
    """Compute PCC necessity, sufficiency, causation, and the corollary check."""
    proxies = [c for c in proxies if c in Xte.columns]
    cols = list(Xte.columns)

    # ---- factual decisions at a fixed budget -----------------------------
    p0 = model.predict_proba(Xte.values)[:, 1]
    thr = float(np.quantile(p0, 1.0 - select_rate))
    d0 = (p0 >= thr).astype(int)

    # ---- NECESSITY probe: neutralize cluster, rest fixed -----------------
    _, fitted = neutralize(Xtr, proxies, mode="orthogonalize", A=Atr)
    Xte_cf, _ = neutralize(Xte, proxies, mode="orthogonalize", fitted=fitted)
    s0 = _scores(model, Xte.values)
    s1 = _scores(model, Xte_cf.values)
    eff = s0 - s1                                      # individual cluster-CF effect
    p1, d1 = _decide(model, Xte_cf.values, thr)
    flip = d0 != d1

    # ---- SUFFICIENCY probe: keep cluster, rest -> neutral baseline --------
    base_row = Xtr.mean(axis=0)
    Xsuf = pd.DataFrame(np.tile(base_row.values, (len(Xte), 1)), columns=cols)
    for c in proxies:
        Xsuf[c] = Xte[c].values
    p_suf, d_suf = _decide(model, Xsuf.values, thr)
    p_base = float(model.predict_proba(base_row.values.reshape(1, -1))[:, 1][0])
    d_base = int(p_base >= thr)

    A1, A0 = (Ate == 1), (Ate == 0)
    adv_rej = A1 & (d0 == 0)
    fav_acc = A0 & (d0 == 1)
    necessity_A1 = float(np.mean(d1[adv_rej] == 1)) if adv_rej.any() else float("nan")
    necessity_A0 = float(np.mean(d1[fav_acc] == 0)) if fav_acc.any() else float("nan")
    sufficiency_A1 = float(np.mean(d_suf[adv_rej] == 0)) if adv_rej.any() else float("nan")
    sufficiency_A0 = float(np.mean(d_suf[fav_acc] == 1)) if fav_acc.any() else float("nan")
    nec1 = (d1 == 1); suf1 = (d_suf == 0)
    nec0 = (d1 == 0); suf0 = (d_suf == 1)
    caused_A1 = float(np.mean(nec1[adv_rej] & suf1[adv_rej])) if adv_rej.any() else float("nan")
    caused_A0 = float(np.mean(nec0[fav_acc] & suf0[fav_acc])) if fav_acc.any() else float("nan")

    out = {
        "n_proxies": len(proxies), "select_threshold": thr,
        "flip_rate": float(flip.mean()),
        "flip_rate_A1": float(flip[A1].mean()) if A1.any() else float("nan"),
        "flip_rate_A0": float(flip[A0].mean()) if A0.any() else float("nan"),
        "gain_rate_A1": float(np.mean(flip[A1] & (d1[A1] > d0[A1]))) if A1.any() else float("nan"),
        "lose_rate_A0": float(np.mean(flip[A0] & (d1[A0] < d0[A0]))) if A0.any() else float("nan"),
        "mean_effect_A1": float(eff[A1].mean()) if A1.any() else float("nan"),
        "mean_effect_A0": float(eff[A0].mean()) if A0.any() else float("nan"),
        "effect_group_diff": float(eff[A1].mean() - eff[A0].mean()),
        "necessity_A1": necessity_A1, "necessity_A0": necessity_A0,
        "sufficiency_A1": sufficiency_A1, "sufficiency_A0": sufficiency_A0,
        "caused_A1": caused_A1, "caused_A0": caused_A0,
        "n_adv_rejected": int(adv_rej.sum()), "n_fav_accepted": int(fav_acc.sum()),
        "n_rescued": int(np.sum(d1[adv_rej] == 1)),
        "n_caused_A1": int(np.sum(nec1[adv_rej] & suf1[adv_rej])),
        "n_flipped": int(flip.sum()),
        "baseline_decision": d_base, "baseline_prob": p_base,
    }
    if cluster_delta is not None:
        out["cluster_delta_prop1"] = float(cluster_delta)
        out["corollary_abs_error"] = float(abs(out["effect_group_diff"] - cluster_delta))

    out["_arrays"] = {"effect": eff, "d0": d0, "d1": d1, "d_suf": d_suf,
                      "p0": p0, "p_suf": p_suf, "flip": flip, "Ate": Ate,
                      "necessary": (d1 != d0), "sufficient": (d_suf == d0)}
    return out


def individual_certificate(pcc, i):
    """Human-readable necessity/sufficiency certificate for test individual i."""
    a = pcc["_arrays"]
    d0 = int(a["d0"][i]); d1 = int(a["d1"][i]); dsuf = int(a["d_suf"][i])
    grp = int(a["Ate"][i]); eff = float(a["effect"][i])
    necessary = (d1 != d0)
    sufficient = (dsuf == d0)
    direction = "suppressed" if eff < 0 else "inflated"
    verdict = ("proxy channel is a CAUSE of this decision"
               if (necessary and sufficient) else
               "proxy channel necessary but not sufficient" if necessary else
               "proxy channel sufficient but not necessary" if sufficient else
               "decision not attributable to the proxy channel")
    return {
        "group": grp, "factual_decision": d0,
        "decision_if_cluster_neutralized": d1,
        "decision_from_cluster_alone": dsuf,
        "cluster_effect": eff, "effect_direction": direction,
        "necessary": bool(necessary), "sufficient": bool(sufficient),
        "verdict": verdict,
    }
