"""Console entry points for ProxyAudit.

`proxyaudit-synth` runs the controlled FairCVtest-recipe audit and prints the
headline before/after triad plus the PCC necessity/sufficiency certificate.
For the full figure + JSON artifacts, use `python scripts/run_synth.py`.
"""
from __future__ import annotations
import argparse
import numpy as np

from .data import make_faircv_recipe
from .pipeline import run_pcln


def main_synth(argv=None):
    ap = argparse.ArgumentParser(prog="proxyaudit-synth",
                                 description="Run the controlled PCLN audit.")
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=["lr", "hgb"], default="lr")
    ap.add_argument("--mode", choices=["drop", "suppress", "orthogonalize"],
                    default="orthogonalize")
    ap.add_argument("--rho", type=float, default=0.85)
    args = ap.parse_args(argv)

    ds = make_faircv_recipe(n=args.n, seed=args.seed, proxy_strength=args.rho)
    o = run_pcln(ds, kind=args.model, seed=args.seed,
                 neutralize_mode=args.mode, n_explain=500, verbose=False)
    b, a = o["before_fair"], o["after_fair"]
    bp, ap_ = o["before_perf"], o["after_perf"]
    L = o.get("localization", {})
    p = o["pcc"]
    print("PCLN audit  | model={} mode={} rho={}".format(args.model, args.mode, args.rho))
    print("  localization  P@k={:.2f}  AP={:.2f}".format(
        L.get("precision_at_k", float("nan")), L.get("average_precision", float("nan"))))
    print("  DP gap   {:.3f} -> {:.3f}".format(b["DP_gap"], a["DP_gap"]))
    print("  DI       {:.3f} -> {:.3f}".format(b["DI"], a["DI"]))
    print("  AUC      {:.3f} -> {:.3f}".format(bp["AUC"], ap_["AUC"]))
    print("  PCC (disadvantaged A=1 rejected):")
    print("    necessity  = {:.3f}".format(p["necessity_A1"]))
    print("    sufficiency= {:.3f}".format(p["sufficiency_A1"]))
    print("    CAUSED     = {:.3f}  ({}/{} rejections caused by the proxy cluster)".format(
        p["caused_A1"], p["n_caused_A1"], p["n_adv_rejected"]))
    print("  Corollary |E[e|A=1]-E[e|A=0] - sum Delta_P| = {:.4f}".format(
        p.get("corollary_abs_error", float("nan"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_synth())
