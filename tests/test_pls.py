"""PLS localizes the injected ground-truth proxy cluster perfectly."""
import numpy as np
from proxyaudit.data import make_faircv_recipe
from proxyaudit.pipeline import run_pcln


def test_localization_recovers_injected_proxies():
    ds = make_faircv_recipe(n=6000, seed=0)
    out = run_pcln(ds, kind="lr", seed=0, neutralize_mode="orthogonalize",
                   n_explain=300, verbose=False)
    L = out["localization"]
    assert L["average_precision"] == 1.0
    assert L["precision_at_k"] == 1.0
    # every selected proxy is a true emb_ proxy
    assert all(p.startswith("emb_") for p in out["localized_proxies"])


def test_neutralization_reduces_dp_gap():
    ds = make_faircv_recipe(n=6000, seed=0)
    out = run_pcln(ds, kind="lr", seed=0, neutralize_mode="orthogonalize",
                   n_explain=300, verbose=False)
    assert out["after_fair"]["DP_gap"] < 0.5 * out["before_fair"]["DP_gap"]
    assert out["after_fair"]["DI"] >= 0.8  # clears four-fifths rule
