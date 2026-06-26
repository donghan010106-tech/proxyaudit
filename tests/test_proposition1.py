"""Proposition 1: the score-level DP gap equals the sum of per-feature Delta_j."""
import numpy as np
from proxyaudit.data import make_faircv_recipe
from proxyaudit.models import make_model, linear_part, transform_for_explainer
from proxyaudit.shapley import LinearSHAP
from proxyaudit.pls import dp_score_decomposition


def test_dp_decomposition_is_exact_for_linear_model():
    ds = make_faircv_recipe(n=4000, seed=0)
    m = make_model("lr", seed=0)
    m.fit(ds.X.values, ds.y)
    lin = linear_part(m)
    Xt = transform_for_explainer(m, ds.X)
    phi = LinearSHAP(lin, transform_for_explainer(m, ds.X)).shap_values(Xt)
    delta, dp_gap = dp_score_decomposition(phi, ds.A)
    # local accuracy => sum(delta) == dp_score_gap exactly
    assert abs(float(delta.sum()) - dp_gap) < 1e-8
