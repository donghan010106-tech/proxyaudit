"""End-to-end PCLN pipeline: localize -> neutralize -> before/after triad."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from .models import make_model, linear_part, transform_for_explainer
from .shapley import LinearSHAP, SamplingSHAP
from .pls import proxy_leakage_score, select_proxies, localization_quality
from .neutralize import neutralize
from .fairness import fairness_report, conditional_dp_gap
from .faithfulness import faithfulness_report
from .trust import trust_report
from .counterfactual import proxy_cluster_counterfactual


def _explain(model, X_raw, background_raw, n_explain=500, seed=0):
    """Return SHAP attributions (n_explain x d) in model score space, plus the
    explained X (raw, for PLS reconstructability) and the chosen row indices."""
    lin = linear_part(model)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_raw), min(n_explain, len(X_raw)), replace=False)
    Xs_raw = X_raw.iloc[idx]
    if lin is not None:
        Xt = transform_for_explainer(model, Xs_raw)
        bg = transform_for_explainer(model, background_raw)
        expl = LinearSHAP(lin, bg)
        phi = expl.shap_values(Xt)
    else:
        expl = SamplingSHAP(model, np.asarray(background_raw), n_perm=40, seed=seed)
        phi = expl.shap_values(np.asarray(Xs_raw))
    return phi, Xs_raw, idx


def perf(model, X, y, select_rate=None):
    """Score and binarise at a fixed *selection budget*.

    In a recruitment setting only a fixed fraction of candidates can be hired,
    so the natural operating point is "select the top `select_rate` by score"
    rather than a fixed probability threshold of 0.5. Using the same budget for
    the before and after models makes their demographic-parity gaps directly
    comparable (identical number selected) and removes the threshold-shift
    confound that a fixed 0.5 cut introduces when the score distribution moves
    after neutralisation. Defaults to the label base rate.
    """
    p = model.predict_proba(X)[:, 1]
    if select_rate is None:
        select_rate = float(np.mean(y))
    select_rate = float(min(max(select_rate, 1e-3), 1 - 1e-3))
    thr = np.quantile(p, 1.0 - select_rate)
    yp = (p >= thr).astype(int)
    return {
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ACC": float(accuracy_score(y, yp)),
        "F1": float(f1_score(y, yp, zero_division=0)),
        "select_rate": select_rate,
    }, p, yp


def run_pcln(ds, kind="lr", seed=0, neutralize_mode="orthogonalize",
             k=None, n_explain=500, senior_cond=False, verbose=True):
    """Run the full PCLN audit on a Dataset and return a results dict."""
    tr, te = ds.split(test_frac=0.2, seed=seed)
    Xtr, Xte = ds.X.iloc[tr].reset_index(drop=True), ds.X.iloc[te].reset_index(drop=True)
    ytr, yte = ds.y[tr], ds.y[te]
    Atr, Ate = ds.A[tr], ds.A[te]
    senior_te = None
    if senior_cond and ds.meta.get("senior") is not None:
        senior_te = np.asarray(ds.meta["senior"])[te]

    out = {"dataset": ds.name, "model": kind, "seed": seed,
           "neutralize_mode": neutralize_mode, "n_features": Xtr.shape[1]}

    # shared hiring budget (top-tau by score) for comparable before/after DP
    select_rate = float(np.mean(ytr))

    # ---- (A) BEFORE: unaware baseline (no A in features) ------------------
    m_un = make_model(kind, seed=seed)
    m_un.fit(Xtr.values, ytr)
    perf_un, p_un, yp_un = perf(m_un, Xte.values, yte, select_rate=select_rate)
    fair_un = fairness_report(yte, yp_un, Ate)
    out["before_perf"] = perf_un
    out["before_fair"] = fair_un
    if senior_te is not None:
        out["before_fair"]["DP_gap_senior"] = conditional_dp_gap(yp_un, Ate, senior_te == 1)

    # ---- (B) LOCALIZE the proxy channel via PLS ---------------------------
    phi, Xexp_raw, exp_idx = _explain(m_un, Xte, Xtr, n_explain=n_explain, seed=seed)
    A_exp = Ate[exp_idx]
    pls = proxy_leakage_score(Xexp_raw.values, A_exp,
                              phi, feature_names=list(Xte.columns),
                              X_recon=Xte.values, A_recon=Ate)
    out["pls_table"] = pls["table"]
    out["dp_score_gap"] = pls["dp_score_gap"]
    proxies, proxy_idx = select_proxies(pls, k=k)
    out["localized_proxies"] = proxies
    if ds.proxy_truth:
        out["localization"] = localization_quality(pls, ds.proxy_truth)

    # ---- Faithfulness/Trust BEFORE ---------------------------------------
    bg_mean = transform_for_explainer(m_un, Xtr).mean(axis=0) \
        if linear_part(m_un) is not None else Xtr.values.mean(axis=0)
    Xexp_model = (transform_for_explainer(m_un, Xexp_raw)
                  if linear_part(m_un) is not None else Xexp_raw.values)
    faith_un = faithfulness_report(m_un, Xexp_model, phi, k_frac=0.3,
                                   background_mean=bg_mean)
    out["before_faith"] = faith_un
    out["before_trust"] = trust_report(yte, p_un)

    # ---- (C) NEUTRALIZE + retrain (PCLN, the AFTER model) -----------------
    Xtr_n, fitted = neutralize(Xtr, proxies, mode=neutralize_mode, A=Atr)
    Xte_n, _ = neutralize(Xte, proxies, mode=neutralize_mode, fitted=fitted)
    m_pc = make_model(kind, seed=seed)
    m_pc.fit(Xtr_n.values, ytr)
    perf_pc, p_pc, yp_pc = perf(m_pc, Xte_n.values, yte, select_rate=select_rate)
    fair_pc = fairness_report(yte, yp_pc, Ate)
    out["after_perf"] = perf_pc
    out["after_fair"] = fair_pc
    if senior_te is not None:
        out["after_fair"]["DP_gap_senior"] = conditional_dp_gap(yp_pc, Ate, senior_te == 1)

    # Faithfulness/Trust AFTER
    phi_pc, Xexp_pc_raw, _ = _explain(m_pc, Xte_n, Xtr_n, n_explain=n_explain, seed=seed)
    bg_mean_pc = transform_for_explainer(m_pc, Xtr_n).mean(axis=0) \
        if linear_part(m_pc) is not None else Xtr_n.values.mean(axis=0)
    Xexp_pc_model = (transform_for_explainer(m_pc, Xexp_pc_raw)
                     if linear_part(m_pc) is not None else Xexp_pc_raw.values)
    out["after_faith"] = faithfulness_report(m_pc, Xexp_pc_model, phi_pc,
                                             k_frac=0.3, background_mean=bg_mean_pc)
    out["after_trust"] = trust_report(yte, p_pc, p_before=p_un,
                                       pred_before=yp_un, pred_after=yp_pc)

    # ---- (D') PROXY-CLUSTER COUNTERFACTUAL (individual-level certificate) --
    # Probe the fixed unaware model's reliance on the localized cluster via a
    # directed (A-orthogonalized) counterfactual; links to Prop. 1 by the
    # corollary  E[e|A=1]-E[e|A=0] = sum_{j in P} Delta_j.
    cluster_delta = float(np.sum(pls["delta"][np.asarray(proxy_idx, dtype=int)])) \
        if len(proxy_idx) else 0.0
    out["pcc"] = proxy_cluster_counterfactual(
        m_un, Xtr, Xte, Atr, Ate, proxies,
        select_rate=select_rate, cluster_delta=cluster_delta)

    # ---- (D) reference: BLANKET in-processing penalty? (drop-all proxies
    #          == targeted unawareness vs full unawareness already done) ----
    if verbose:
        _print_summary(out)
    out["_arrays"] = {"yte": yte, "Ate": Ate, "p_un": p_un, "p_pc": p_pc,
                      "yp_un": yp_un, "yp_pc": yp_pc,
                      "phi_before": phi, "Xexp_before": Xexp_raw,
                      "pls": pls, "Xte": Xte, "feature_names": list(Xte.columns)}
    return out


def _print_summary(out):
    b, a = out["before_fair"], out["after_fair"]
    bp, ap = out["before_perf"], out["after_perf"]
    print(f"\n=== {out['dataset']} | {out['model']} | mode={out['neutralize_mode']} ===")
    print(f"  localized proxies: {out['localized_proxies']}")
    if "localization" in out:
        L = out["localization"]
        print(f"  localization: P@k={L['precision_at_k']:.2f} "
              f"AUC={L['pls_auc']:.3f} AP={L['average_precision']:.3f}")
    print(f"  AUC   {bp['AUC']:.3f} -> {ap['AUC']:.3f}")
    print(f"  DP    {b['DP_gap']:.4f} -> {a['DP_gap']:.4f}  "
          f"({100*(b['DP_gap']-a['DP_gap'])/max(b['DP_gap'],1e-9):+.0f}%)")
    print(f"  DI    {b['DI']:.3f} -> {a['DI']:.3f}")
    print(f"  Faith(comp) {out['before_faith']['comprehensiveness']:.3f} -> "
          f"{out['after_faith']['comprehensiveness']:.3f}")
    print(f"  Trust(ECE)  {out['before_trust']['ECE']:.4f} -> "
          f"{out['after_trust']['ECE']:.4f}  "
          f"stability={out['after_trust'].get('decision_stability', float('nan')):.3f}")
