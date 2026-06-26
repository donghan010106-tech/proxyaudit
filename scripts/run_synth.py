"""Run the controlled FairCVtest-recipe experiment end-to-end.

Produces:
  results/synth_results.json   -- headline before/after triad (multi-seed mean+/-std)
  results/synth_pls.csv        -- PLS ranking table (seed 0)
  figures/*.png                -- all paper figures
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from proxyaudit.data import make_faircv_recipe, COMPETENCY
from proxyaudit.pipeline import run_pcln
from proxyaudit.pls import proxy_leakage_score, localization_quality
from proxyaudit import viz

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)


def agg(dicts, path):
    keys = dicts[0].keys()
    return {k: {"mean": float(np.mean([d[k] for d in dicts])),
                "std": float(np.std([d[k] for d in dicts]))} for k in keys}


def main():
    SEEDS = [0, 1, 2, 3, 4]
    MODE = "orthogonalize"
    print(f"Controlled FairCVtest-recipe experiment  | mode={MODE} | seeds={SEEDS}")

    # ---- multi-seed headline (LR, exact SHAP) ---------------------------
    runs = []
    for s in SEEDS:
        ds = make_faircv_recipe(n=12000, seed=s)
        out = run_pcln(ds, kind="lr", seed=s, neutralize_mode=MODE,
                       n_explain=600, verbose=(s == 0))
        runs.append(out)

    def collect(side):
        rows = []
        for o in runs:
            rows.append({
                "AUC": o[f"{side}_perf"]["AUC"], "F1": o[f"{side}_perf"]["F1"],
                "DP_gap": o[f"{side}_fair"]["DP_gap"], "DI": o[f"{side}_fair"]["DI"],
                "EOO_gap": o[f"{side}_fair"]["EOO_gap"],
                "comprehensiveness": o[f"{side}_faith"]["comprehensiveness"],
                "suff_ratio": o[f"{side}_faith"]["suff_ratio"],
                "ECE": o[f"{side}_trust"]["ECE"], "Brier": o[f"{side}_trust"]["Brier"],
            })
        return rows

    loc_rows = [o["localization"] for o in runs]
    pcc_keys = ["flip_rate", "flip_rate_A1", "flip_rate_A0", "gain_rate_A1",
                "lose_rate_A0", "mean_effect_A1", "mean_effect_A0",
                "effect_group_diff", "cluster_delta_prop1", "corollary_abs_error",
                "necessity_A1", "necessity_A0", "sufficiency_A1", "sufficiency_A0",
                "caused_A1", "caused_A0"]
    headline = {
        "mode": MODE, "seeds": SEEDS, "model": "lr",
        "before": agg(collect("before"), None),
        "after": agg(collect("after"), None),
        "localization": {k: {"mean": float(np.mean([r[k] for r in loc_rows])),
                             "std": float(np.std([r[k] for r in loc_rows]))}
                         for k in loc_rows[0].keys()},
        "decision_stability_mean": float(np.mean([o["after_trust"]["decision_stability"] for o in runs])),
        "pcc": {k: {"mean": float(np.mean([o["pcc"][k] for o in runs])),
                    "std": float(np.std([o["pcc"][k] for o in runs]))} for k in pcc_keys},
        "pcc_corollary_pairs": [[o["pcc"]["cluster_delta_prop1"],
                                 o["pcc"]["effect_group_diff"]] for o in runs],
        "pcc_n_rescued_seed0": runs[0]["pcc"]["n_rescued"],
        "pcc_n_adv_rejected_seed0": runs[0]["pcc"]["n_adv_rejected"],
    }

    # ---- neutraliser comparison on a MIXED-proxy scenario ----------------
    # Here proxies carry a latent legitimate competency, so `drop` discards real
    # signal while `orthogonalize` removes only the A-aligned component. This is
    # the experiment that motivates orthogonalisation as the default.
    mode_cmp = {}
    sem_before = None
    for nm in ["drop", "suppress", "orthogonalize"]:
        rws = []
        for s in SEEDS:
            ds = make_faircv_recipe(n=12000, seed=s, n_proxy=4, n_semantic=3,
                                    semantic_signal=1.0)
            o = run_pcln(ds, kind="lr", seed=s, neutralize_mode=nm,
                         n_explain=500, verbose=False)
            rws.append({"AUC": o["after_perf"]["AUC"], "DP_gap": o["after_fair"]["DP_gap"],
                        "DI": o["after_fair"]["DI"], "EOO_gap": o["after_fair"]["EOO_gap"],
                        "n_selected": len(o["localized_proxies"])})
            if nm == "drop" and s == SEEDS[0]:
                sem_before = {"AUC": o["before_perf"]["AUC"],
                              "DP_gap": o["before_fair"]["DP_gap"],
                              "localization_ap": o["localization"]["average_precision"]}
        mode_cmp[nm] = {k: {"mean": float(np.mean([r[k] for r in rws])),
                            "std": float(np.std([r[k] for r in rws]))} for k in rws[0]}
    headline["mode_comparison"] = mode_cmp
    headline["mode_comparison_before"] = sem_before

    # ---- necessity / sufficiency / causation across proxy regimes --------
    def _nsc(ds_kwargs, seeds=(0, 1, 2)):
        n, s_, c = [], [], []
        for s in seeds:
            ds = make_faircv_recipe(n=9000, seed=s, **ds_kwargs)
            o = run_pcln(ds, kind="lr", seed=s, neutralize_mode="orthogonalize",
                         n_explain=350, verbose=False)
            p = o["pcc"]
            n.append(p["necessity_A1"]); s_.append(p["sufficiency_A1"]); c.append(p["caused_A1"])
        return float(np.mean(n)), float(np.mean(s_)), float(np.mean(c))

    rho_rows = [[rho, *_nsc(dict(proxy_strength=rho))]
                for rho in [0.3, 0.5, 0.7, 0.85, 0.95]]
    mix_rows = [[k, *_nsc(dict(n_proxy=4, n_semantic=k))] for k in [0, 2, 3]]
    headline["causation_rho"] = rho_rows
    headline["causation_mix"] = mix_rows

    # ---- interventional vs conditional Shapley for NDE/NIE ----------------
    from proxyaudit.models import make_model
    from proxyaudit.mediation import compare_modes
    iv_sh, cd_sh, spur_iv, spur_cd = [], [], [], []
    cmp0 = None
    for s in SEEDS:
        dsm = make_faircv_recipe(n=7000, seed=s, n_proxy=6, legit_corr=0.7)
        trm, tem = dsm.split(test_frac=0.3, seed=s)
        Xtrm = dsm.X.iloc[trm].reset_index(drop=True)
        Xtem = dsm.X.iloc[tem].reset_index(drop=True)
        mm = make_model("lr", seed=s); mm.fit(Xtrm.values, dsm.y[trm])
        ridx = np.random.default_rng(s).choice(len(Xtem), 500, replace=False)
        truth = [c for c in dsm.feature_names if dsm.proxy_truth.get(c)]
        cmp = compare_modes(mm, Xtem.iloc[ridx], dsm.A[tem][ridx], Xtrm, truth,
                            dsm.feature_names, truth_proxies=truth, seed=s, n_perm=48)
        iv_sh.append(cmp["interventional"]["true_proxy_share"])
        cd_sh.append(cmp["conditional"]["true_proxy_share"])
        spur_iv.append(sum(abs(cmp["interventional"]["per_feature"][c])
                           for c in dsm.feature_names if c.startswith("spur_")))
        spur_cd.append(sum(abs(cmp["conditional"]["per_feature"][c])
                           for c in dsm.feature_names if c.startswith("spur_")))
        if s == SEEDS[0]:
            cmp0 = (cmp, truth)
    headline["mediation"] = {
        "true_proxy_share_interventional": {"mean": float(np.mean(iv_sh)), "std": float(np.std(iv_sh))},
        "true_proxy_share_conditional": {"mean": float(np.mean(cd_sh)), "std": float(np.std(cd_sh))},
        "spurious_attr_interventional": {"mean": float(np.mean(spur_iv)), "std": float(np.std(spur_iv))},
        "spurious_attr_conditional": {"mean": float(np.mean(spur_cd)), "std": float(np.std(spur_cd))},
        "NIE_share_interventional": cmp0[0]["interventional"]["NIE_share"],
        "NIE_share_conditional": cmp0[0]["conditional"]["NIE_share"],
    }

    # ---- which explanation localizes the proxy? (two-regime comparison) ---
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.feature_selection import mutual_info_classif
    from proxyaudit.models import linear_part, transform_for_explainer
    from proxyaudit.shapley import LinearSHAP
    from proxyaudit.pls import dp_score_decomposition, reconstructability
    loc_methods = ["PLS (ours)", "D-only", "R-only", "Mutual info", "Permutation"]

    def _localize_ap(kw):
        ap = {k: [] for k in loc_methods}
        for s in SEEDS:
            dl = make_faircv_recipe(n=9000, seed=s, n_proxy=6, **kw)
            trl, tel = dl.split(test_frac=0.3, seed=s)
            Xtl = dl.X.iloc[trl].reset_index(drop=True); Xel = dl.X.iloc[tel].reset_index(drop=True)
            ml = make_model("lr", seed=s); ml.fit(Xtl.values, dl.y[trl])
            truth = np.array([1 if dl.proxy_truth.get(c) else 0 for c in dl.feature_names])
            phil = LinearSHAP(linear_part(ml), transform_for_explainer(ml, Xtl)
                              ).shap_values(transform_for_explainer(ml, Xel))
            Dl = np.abs(dp_score_decomposition(phil, dl.A[tel])[0])
            Rl = reconstructability(Xel, dl.A[tel])
            MIl = mutual_info_classif(Xel.values, dl.A[tel], random_state=s)
            base = roc_auc_score(dl.y[tel], ml.predict_proba(Xel.values)[:, 1])
            rng = np.random.default_rng(s); perm = []
            for j in range(Xel.shape[1]):
                Xp = Xel.values.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
                perm.append(abs(base - roc_auc_score(dl.y[tel], ml.predict_proba(Xp)[:, 1])))
            perm = np.array(perm)
            for k, sc in zip(loc_methods, [np.sqrt(Rl * Dl), Dl, Rl, MIl, perm]):
                ap[k].append(average_precision_score(truth, sc))
        return {k: (float(np.mean(v)), float(np.std(v))) for k, v in ap.items()}

    regA = _localize_ap(dict(n_semantic=2, legit_corr=0.7))   # spurious A-correlates
    regB = _localize_ap(dict(legit_diff_n=2, legit_diff_noise=3.0))  # legit group-diff
    headline["localizer_two_regime"] = {
        "methods": loc_methods,
        "regimeA_spurious": {k: regA[k] for k in loc_methods},
        "regimeB_legitdiff": {k: regB[k] for k in loc_methods},
    }
    # back-compat single-regime field (regime A) for the figure
    headline["localizer_comparison"] = [
        [k, regA[k][0], regA[k][1], 0.0, 0.0] for k in loc_methods]

    # ---- reviewer fix: bridge discrepancy, linear vs non-linear ----------
    nl_err, lin_err, nl_ap = [], [], []
    for s in SEEDS:
        for kind, bucket, apb in [("lr", lin_err, None), ("hgb", nl_err, nl_ap)]:
            ob = run_pcln(make_faircv_recipe(n=7000, seed=s), kind=kind, seed=s,
                          neutralize_mode="orthogonalize", n_explain=250, verbose=False)
            bucket.append(ob["pcc"]["corollary_abs_error"])
            if apb is not None:
                apb.append(ob["localization"]["average_precision"])
    headline["bridge_linear_vs_nonlinear"] = {
        "linear_abs_error": {"mean": float(np.mean(lin_err)), "std": float(np.std(lin_err))},
        "nonlinear_abs_error": {"mean": float(np.mean(nl_err)), "std": float(np.std(nl_err))},
        "nonlinear_localization_ap": float(np.mean(nl_ap)),
    }

    # ---- reviewer fix: conditional finding is not a Gaussian artefact ----
    from proxyaudit.shapley import GaussianConditionalSHAP, EmpiricalConditionalSHAP
    def _proxy_share(est, Xt, Ae, truth):
        phi = est.shap_values(Xt)
        d = np.abs(dp_score_decomposition(phi, Ae)[0])
        return float(d[truth].sum() / (d.sum() + 1e-12))
    iv_s, cg_s, ck_s = [], [], []
    for s in SEEDS[:3]:
        dk = make_faircv_recipe(n=5000, seed=s, n_proxy=6, legit_corr=0.7)
        trk, tek = dk.split(test_frac=0.3, seed=s)
        Xtk = dk.X.iloc[trk].reset_index(drop=True); Xek = dk.X.iloc[tek].reset_index(drop=True)
        mk = make_model("lr", seed=s); mk.fit(Xtk.values, dk.y[trk])
        ridx = np.random.default_rng(s).choice(len(Xek), 120, replace=False)
        bg = transform_for_explainer(mk, Xtk); Xt = transform_for_explainer(mk, Xek.iloc[ridx])
        Ae = dk.A[tek][ridx]
        truth = np.array([1 if dk.proxy_truth.get(c) else 0 for c in dk.feature_names]).astype(bool)
        bgs = bg[np.random.default_rng(0).choice(len(bg), 400, replace=False)]
        iv_s.append(_proxy_share(LinearSHAP(linear_part(mk), bg), Xt, Ae, truth))
        cg_s.append(_proxy_share(GaussianConditionalSHAP(mk, bg, n_perm=32, seed=s), Xt, Ae, truth))
        ck_s.append(_proxy_share(EmpiricalConditionalSHAP(mk, bgs, n_perm=24, k=30, seed=s), Xt, Ae, truth))
    headline["conditional_non_gaussian"] = {
        "interventional": {"mean": float(np.mean(iv_s)), "std": float(np.std(iv_s))},
        "conditional_gaussian": {"mean": float(np.mean(cg_s)), "std": float(np.std(cg_s))},
        "conditional_knn": {"mean": float(np.mean(ck_s)), "std": float(np.std(ck_s))},
    }

    # ---- HGB cross-model check (sampling SHAP, smaller explain set) ------
    ds0 = make_faircv_recipe(n=12000, seed=0)
    hgb = run_pcln(ds0, kind="hgb", seed=0, neutralize_mode="orthogonalize",
                   n_explain=300, verbose=True)
    headline["hgb_seed0"] = {
        "before": {"AUC": hgb["before_perf"]["AUC"], "DP_gap": hgb["before_fair"]["DP_gap"],
                   "DI": hgb["before_fair"]["DI"]},
        "after": {"AUC": hgb["after_perf"]["AUC"], "DP_gap": hgb["after_fair"]["DP_gap"],
                  "DI": hgb["after_fair"]["DI"]},
        "localization": hgb.get("localization", {}),
    }

    json.dump(headline, open(os.path.join(RES, "synth_results.json"), "w"), indent=2)
    pd.DataFrame(runs[0]["pls_table"]).to_csv(os.path.join(RES, "synth_pls.csv"), index=False)
    print("\nSaved results/synth_results.json")

    # ---- FIGURES (seed 0) -----------------------------------------------
    o0 = runs[0]; arr = o0["_arrays"]; pls = arr["pls"]
    ds_fig = make_faircv_recipe(n=12000, seed=0)
    ptruth = ds_fig.proxy_truth

    viz.fig_corr_heatmap(ds_fig.X, ds_fig.A, ptruth, COMPETENCY,
                         os.path.join(FIG, "fig_corr_heatmap.png"))
    viz.fig_pls_ranking(o0["pls_table"], ptruth, COMPETENCY,
                        os.path.join(FIG, "fig_pls_ranking.png"))
    viz.fig_dp_waterfall(pls, ptruth, COMPETENCY,
                         os.path.join(FIG, "fig_dp_waterfall.png"))
    viz.fig_scatter_RD(pls, ptruth, COMPETENCY,
                       os.path.join(FIG, "fig_scatter_RD.png"))
    viz.fig_pairplot(ds_fig.X, ds_fig.A, ptruth, COMPETENCY,
                     os.path.join(FIG, "fig_pairplot.png"))

    before = {"fair": o0["before_fair"], "faith": o0["before_faith"], "trust": o0["before_trust"]}
    after = {"fair": o0["after_fair"], "faith": o0["after_faith"], "trust": o0["after_trust"]}
    viz.fig_triad(before, after, os.path.join(FIG, "fig_triad.png"))
    viz.fig_density(arr["p_un"], arr["p_pc"], arr["Ate"],
                    os.path.join(FIG, "fig_density.png"))

    # trade-off frontier: unfair, unaware(=before leak), PCLN(after), comp-only
    pts = [
        ("Unaware (proxy leak)", o0["before_perf"]["AUC"], o0["before_fair"]["DP_gap"], "o", "#E4572E"),
        ("PCLN (ours)", o0["after_perf"]["AUC"], o0["after_fair"]["DP_gap"], "*", "#2E9E5B"),
    ]
    viz.fig_tradeoff(pts, os.path.join(FIG, "fig_tradeoff.png"))
    viz.fig_mode_comparison(headline["mode_comparison"],
                            os.path.join(FIG, "fig_mode_comparison.png"),
                            before=headline.get("mode_comparison_before"))
    viz.fig_pcc_effects(o0["pcc"], os.path.join(FIG, "fig_pcc_effects.png"))
    viz.fig_decision_map(o0["pcc"], os.path.join(FIG, "fig_decision_map.png"))
    viz.fig_before_after(o0["_arrays"], os.path.join(FIG, "fig_before_after.png"))
    viz.fig_pcc_corollary(headline["pcc_corollary_pairs"],
                          os.path.join(FIG, "fig_pcc_corollary.png"))
    viz.fig_causation_surface(headline["causation_rho"], headline["causation_mix"],
                              os.path.join(FIG, "fig_causation_surface.png"))
    if cmp0 is not None:
        _cmp, _truth = cmp0
        viz.fig_mediation(_cmp["interventional"]["per_feature"],
                          _cmp["conditional"]["per_feature"], set(_truth), "spur_",
                          _cmp["interventional"]["true_proxy_share"],
                          _cmp["conditional"]["true_proxy_share"],
                          os.path.join(FIG, "fig_mediation.png"))
    viz.fig_localizer_comparison(headline["localizer_comparison"],
                                 os.path.join(FIG, "fig_localizer_comparison.png"))

    # ---- sensitivity sweep over proxy strength --------------------------
    strengths = [0.3, 0.5, 0.7, 0.85, 0.95]
    dp_b, dp_a, aps, roc_curves = [], [], [], []
    for rho in strengths:
        ds = make_faircv_recipe(n=10000, seed=0, proxy_strength=rho)
        out = run_pcln(ds, kind="lr", seed=0, neutralize_mode=MODE,
                       n_explain=500, verbose=False)
        dp_b.append(out["before_fair"]["DP_gap"]); dp_a.append(out["after_fair"]["DP_gap"])
        aps.append(out["localization"]["average_precision"])
        pls_r = out["_arrays"]["pls"]
        truth = np.array([1 if ds.proxy_truth.get(n, False) else 0 for n in pls_r["names"]])
        roc_curves.append((f"rho={rho}", truth, pls_r["PLS"]))
    viz.fig_sensitivity(strengths, dp_b, dp_a, aps, os.path.join(FIG, "fig_sensitivity.png"))
    viz.fig_localization_roc([roc_curves[1], roc_curves[3], roc_curves[4]],
                             os.path.join(FIG, "fig_localization_roc.png"))
    headline["sensitivity"] = {"strengths": strengths, "dp_before": dp_b,
                               "dp_after": dp_a, "localization_ap": aps}
    json.dump(headline, open(os.path.join(RES, "synth_results.json"), "w"), indent=2)

    print("\nFigures written to figures/:")
    for f in sorted(os.listdir(FIG)):
        print("  ", f)
    print("\n--- HEADLINE (mean over seeds) ---")
    b, a = headline["before"], headline["after"]
    print(f"  DP gap : {b['DP_gap']['mean']:.4f} -> {a['DP_gap']['mean']:.4f}")
    print(f"  DI     : {b['DI']['mean']:.3f} -> {a['DI']['mean']:.3f}")
    print(f"  AUC    : {b['AUC']['mean']:.3f} -> {a['AUC']['mean']:.3f}")
    print(f"  ECE    : {b['ECE']['mean']:.4f} -> {a['ECE']['mean']:.4f}")
    print(f"  Localisation AP: {headline['localization']['average_precision']['mean']:.3f}")


if __name__ == "__main__":
    main()
