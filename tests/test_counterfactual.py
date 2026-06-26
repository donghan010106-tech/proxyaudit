"""PCC: necessity, sufficiency, causation, and the consistency corollary."""
import numpy as np
from proxyaudit.data import make_faircv_recipe
from proxyaudit.pipeline import run_pcln
from proxyaudit.counterfactual import individual_certificate


def test_corollary_consistency():
    ds = make_faircv_recipe(n=8000, seed=0)
    out = run_pcln(ds, kind="lr", seed=0, neutralize_mode="orthogonalize",
                   n_explain=400, verbose=False)
    p = out["pcc"]
    # PCC group-difference averages back to the Prop-1 cluster term
    assert p["corollary_abs_error"] < 0.1


def test_necessity_sufficiency_in_unit_interval():
    ds = make_faircv_recipe(n=6000, seed=1)
    p = run_pcln(ds, kind="lr", seed=1, neutralize_mode="orthogonalize",
                 n_explain=300, verbose=False)["pcc"]
    for k in ["necessity_A1", "sufficiency_A1", "caused_A1"]:
        assert 0.0 <= p[k] <= 1.0
    # caused <= min(necessity, sufficiency)
    assert p["caused_A1"] <= min(p["necessity_A1"], p["sufficiency_A1"]) + 1e-9


def test_individual_certificate_fields():
    ds = make_faircv_recipe(n=4000, seed=0)
    out = run_pcln(ds, kind="lr", seed=0, neutralize_mode="orthogonalize",
                   n_explain=200, verbose=False)
    cert = individual_certificate(out["pcc"], 0)
    assert set(["necessary", "sufficient", "verdict"]).issubset(cert)
