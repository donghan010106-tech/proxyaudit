"""Run PCLN on the REAL FairCVdb (24,000 profiles).

Point this at the official FairCVdb.npy (after `git lfs pull`), or a pre-exported CSV:

    python scripts/run_faircv_real.py /path/to/FairCVdb.npy

The 20-D face embedding is the real proxy channel. There is no per-feature
ground truth here (we did not inject the proxies), so localization quality is
reported descriptively (which features PLS flags) rather than scored against a
known answer; the before/after triad is fully measured.
"""
from __future__ import annotations
import sys, os, json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from proxyaudit.data import load_faircv_real, load_faircv_npy
from proxyaudit.pipeline import run_pcln
from proxyaudit import viz

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)


def main(path, label="biased_label_gender", kind="lr", mode="orthogonalize"):
    # Accept the official FairCVdb.npy dict directly, or a pre-exported CSV.
    if path.endswith(".npy"):
        ds = load_faircv_npy(path, config="gender")
    else:
        ds = load_faircv_real(path, protected="gender", label=label)
    print(f"loaded {ds.name}: n={ds.n}, features={len(ds.feature_names)}")
    out = run_pcln(ds, kind=kind, seed=0, neutralize_mode=mode,
                   n_explain=800, verbose=True)

    flagged = out["localized_proxies"]
    print("\nPCLN flagged proxy channel:", flagged)
    summary = {
        "dataset": ds.name, "model": kind, "mode": mode,
        "before": {"AUC": out["before_perf"]["AUC"], **out["before_fair"]},
        "after": {"AUC": out["after_perf"]["AUC"], **out["after_fair"]},
        "flagged_proxies": flagged,
    }
    json.dump(summary, open(os.path.join(RES, "faircv_real_results.json"), "w"),
              indent=2, default=float)

    # figures from the real run
    pls = out["_arrays"]["pls"]
    viz.fig_pls_ranking(out["pls_table"], ds.proxy_truth or {f: f.startswith("face_emb_") for f in ds.feature_names},
                        [c for c in ds.feature_names if not c.startswith("face_emb_")],
                        os.path.join(FIG, "faircv_real_pls.png"))
    before = {"fair": out["before_fair"], "faith": out["before_faith"], "trust": out["before_trust"]}
    after = {"fair": out["after_fair"], "faith": out["after_faith"], "trust": out["after_trust"]}
    viz.fig_triad(before, after, os.path.join(FIG, "faircv_real_triad.png"))
    print("\nSaved results/faircv_real_results.json and figures/faircv_real_*.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/run_faircv_real.py FairCVdb.csv [label]")
    label = sys.argv[2] if len(sys.argv) > 2 else "biased_label_gender"
    main(sys.argv[1], label=label)
