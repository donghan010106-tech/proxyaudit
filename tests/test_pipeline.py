"""End-to-end smoke test of the audit on both model families."""
import pytest
from proxyaudit.data import make_faircv_recipe
from proxyaudit.pipeline import run_pcln


@pytest.mark.parametrize("kind", ["lr", "hgb"])
def test_pipeline_runs(kind):
    ds = make_faircv_recipe(n=3000, seed=0)
    out = run_pcln(ds, kind=kind, seed=0, neutralize_mode="orthogonalize",
                   n_explain=120, verbose=False)
    for key in ["before_fair", "after_fair", "localization", "pcc"]:
        assert key in out
    assert out["after_perf"]["AUC"] > 0.5
