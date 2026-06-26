"""Mediation: both Shapley estimators satisfy the exact decomposition, and the
interventional estimator concentrates attribution on the true proxies."""
import numpy as np
from proxyaudit.data import make_faircv_recipe
from proxyaudit.models import make_model
from proxyaudit.mediation import mediation_decomposition, compare_modes


def _fit_and_explain(seed=0):
    ds = make_faircv_recipe(n=6000, seed=seed, n_proxy=6, legit_corr=0.7)
    tr, te = ds.split(test_frac=0.3, seed=seed)
    Xtr = ds.X.iloc[tr].reset_index(drop=True)
    Xte = ds.X.iloc[te].reset_index(drop=True)
    m = make_model("lr", seed=seed); m.fit(Xtr.values, ds.y[tr])
    idx = np.random.default_rng(seed).choice(len(Xte), 300, replace=False)
    truth = [c for c in ds.feature_names if ds.proxy_truth.get(c)]
    return m, Xte.iloc[idx], ds.A[te][idx], Xtr, truth, ds.feature_names


def test_efficiency_holds_for_both_estimators():
    m, Xexp, Aexp, Xtr, truth, names = _fit_and_explain()
    for mode in ["interventional", "conditional"]:
        r = mediation_decomposition(m, Xexp, Aexp, Xtr, truth, names,
                                    mode=mode, n_perm=32)
        # NIE + NDE == DP gap (Proposition 1) exactly
        assert abs((r["NIE_proxy"] + r["NDE_rest"]) - r["dp_gap"]) < 1e-6


def test_interventional_is_more_faithful_than_conditional():
    m, Xexp, Aexp, Xtr, truth, names = _fit_and_explain()
    r = compare_modes(m, Xexp, Aexp, Xtr, truth, names,
                      truth_proxies=truth, n_perm=40)
    # interventional concentrates more |Delta| on the TRUE proxies
    assert r["interventional"]["true_proxy_share"] > r["conditional"]["true_proxy_share"]
