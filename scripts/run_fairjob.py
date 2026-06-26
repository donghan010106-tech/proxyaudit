"""Run PCLN on FairJob.

Two modes:

  (1) REAL data (requires the licensed release + `datasets`/parquet):
        python scripts/run_fairjob.py --real criteo/FairJob
        python scripts/run_fairjob.py --real /path/to/fairjob.parquet --nrows 200000
      Applies the identical localize->neutralize->triad loop. The DP gap is
      reported on the senior-ad slice, as the FairJob benchmark prescribes, and
      the displayrandom slice gives an unbiased utility estimate.

  (2) DISCLOSED protocol DEMO on a FairJob-calibrated synthetic twin (default):
        python scripts/run_fairjob.py
      Produces transfer figures so the pipeline can be inspected offline. Every
      number here is a PROTOCOL DEMONSTRATION on a synthetic twin, never a
      measurement of the real dataset. The real-world anchors are FairJob's
      PUBLISHED baselines (Vladimirova et al., NeurIPS 2024):
        unaware XGB  AUC ~0.758  DP ~0.0028
        unfair  XGB  AUC ~0.762  DP ~0.0032   (fairness ~ indistinguishable)
        dummy        AUC  0.500  DP  0.000
"""
from __future__ import annotations
import sys, os, json, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from proxyaudit.data import make_fairjob_sim, load_fairjob_real
from proxyaudit.pipeline import run_pcln
from proxyaudit import viz

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

PUBLISHED = {  # real-world anchors, cited not measured
    "unaware_xgb": {"AUC": 0.758, "DP": 0.00278},
    "unfair_xgb":  {"AUC": 0.762, "DP": 0.00323},
    "dummy":       {"AUC": 0.500, "DP": 0.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="HF id or .parquet/.csv path")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--kind", default="lr")
    ap.add_argument("--mode", default="orthogonalize")
    args = ap.parse_args()

    if args.real:
        ds = load_fairjob_real(args.real, nrows=args.nrows)
        tag = "fairjob_real"
        disclosure = "REAL FairJob measurement."
    else:
        ds = make_fairjob_sim(n=40000, seed=0)
        tag = "fairjob_sim"
        disclosure = ("DISCLOSED synthetic twin -- PROTOCOL DEMO ONLY, "
                      "not a real FairJob measurement.")
    print(f"[{tag}] {disclosure}")
    print(f"loaded {ds.name}: n={ds.n}, features={len(ds.feature_names)}, "
          f"click_rate={float(np.mean(ds.y)):.4f}")

    out = run_pcln(ds, kind=args.kind, seed=0, neutralize_mode=args.mode,
                   n_explain=600, senior_cond=True, verbose=True)

    summary = {
        "dataset": ds.name, "disclosure": disclosure,
        "model": args.kind, "mode": args.mode,
        "before": {"AUC": out["before_perf"]["AUC"], **out["before_fair"]},
        "after": {"AUC": out["after_perf"]["AUC"], **out["after_fair"]},
        "flagged_proxies": out["localized_proxies"],
        "published_real_anchors": PUBLISHED,
    }
    if ds.proxy_truth:
        summary["localization"] = out.get("localization", {})
    json.dump(summary, open(os.path.join(RES, f"{tag}_results.json"), "w"),
              indent=2, default=float)

    # transfer figures
    comp = [c for c in ds.feature_names if not c.startswith("num")]
    viz.fig_pls_ranking(out["pls_table"], ds.proxy_truth, comp,
                        os.path.join(FIG, f"{tag}_pls.png"), top=20)
    before = {"fair": out["before_fair"], "faith": out["before_faith"], "trust": out["before_trust"]}
    after = {"fair": out["after_fair"], "faith": out["after_faith"], "trust": out["after_trust"]}
    viz.fig_triad(before, after, os.path.join(FIG, f"{tag}_triad.png"))
    print(f"\nSaved results/{tag}_results.json and figures/{tag}_*.png")
    if not args.real:
        print("\nReminder: real-world claims rest on the PUBLISHED anchors above,"
              "\nnot on this synthetic twin. Use --real to audit the licensed data.")


if __name__ == "__main__":
    main()
