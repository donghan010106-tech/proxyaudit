"""Advanced visualisations for ProxyAudit.

Every figure is rendered from REAL arrays produced by the pipeline. Palette and
style are shared so the paper and the Streamlit app look consistent.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.metrics import roc_curve, auc as sk_auc

mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
})
COL = {"proxy": "#E4572E", "competency": "#2E86AB", "noise": "#9AA0A6",
       "before": "#E4572E", "after": "#2E9E5B", "A0": "#3A7FD5", "A1": "#E8567A"}


def _ftype(name, proxy_truth, competency):
    if proxy_truth.get(name, False) or name.startswith("emb_") or name.startswith("num"):
        return "proxy" if proxy_truth.get(name, name.startswith("emb_")) else "competency"
    if name in competency or name.startswith("lang") or name in (
            "suitability", "educ_attainment", "prev_experience",
            "recommendation", "availability"):
        return "competency"
    if name.startswith("noise"):
        return "noise"
    return "competency"


# ---------------------------------------------------------------------------
def fig_corr_heatmap(X, A, proxy_truth, competency, path, title="Feature correlation with structure"):
    """Correlation heatmap of features + a column showing |corr(feature, A)|."""
    df = X.copy()
    corr = df.corr().values
    cA = np.array([np.corrcoef(df[c].values, A)[0, 1] for c in df.columns])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                             gridspec_kw={"width_ratios": [5, 1]})
    sns.heatmap(corr, ax=axes[0], cmap="vlag", center=0, vmin=-1, vmax=1,
                xticklabels=df.columns, yticklabels=df.columns, cbar_kws={"shrink": .6})
    axes[0].set_title("(a) Feature-feature correlation", fontweight="bold")
    axes[0].tick_params(labelsize=6)
    order = np.argsort(-np.abs(cA))
    colors = [COL[_ftype(df.columns[i], proxy_truth, competency)] for i in order]
    axes[1].barh(range(len(cA)), np.abs(cA[order]), color=colors)
    axes[1].set_yticks(range(len(cA)))
    axes[1].set_yticklabels(np.array(df.columns)[order], fontsize=6)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("|corr(feature, A)|")
    axes[1].set_title("(b) Protected-attr.\ncorrelation", fontweight="bold")
    handles = [Patch(color=COL[k], label=k) for k in ["proxy", "competency", "noise"]]
    axes[1].legend(handles=handles, fontsize=7, loc="lower right")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_pls_ranking(pls_table, proxy_truth, competency, path, top=18):
    """Horizontal bar of PLS with type colours and ground-truth proxy markers."""
    tab = pls_table[:top]
    names = [r["feature"] for r in tab]
    vals = [r["PLS"] for r in tab]
    colors = [COL[_ftype(n, proxy_truth, competency)] for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors)
    for i, n in enumerate(names):
        if proxy_truth.get(n, False):
            ax.text(vals[i] + 0.005, i, "*", va="center", fontsize=14, color="black")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Proxy Leakage Score (PLS)")
    ax.set_title("Proxy-channel localisation  (* = ground-truth proxy)", fontweight="bold")
    handles = [Patch(color=COL[k], label=k) for k in ["proxy", "competency", "noise"]]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_dp_waterfall(pls, proxy_truth, competency, path, top=14):
    """Waterfall of per-feature DP contributions Delta_j summing to the DP gap
    (a visual proof of Proposition 1)."""
    names = pls["names"]; delta = pls["delta"]
    order = np.argsort(-np.abs(delta))[:top]
    names_o = [names[i] for i in order]; d_o = delta[order]
    colors = [COL[_ftype(n, proxy_truth, competency)] for n in names_o]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    cum = 0.0
    for i, (n, d, c) in enumerate(zip(names_o, d_o, colors)):
        ax.bar(i, d, bottom=cum if d > 0 else cum + d, color=c, edgecolor="white")
        cum += d
    ax.axhline(pls["dp_score_gap"], ls="--", color="black", lw=1,
               label=f"DP score gap = {pls['dp_score_gap']:.2f}")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(range(len(names_o))); ax.set_xticklabels(names_o, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel(r"DP contribution  $\Delta_j$ (logit)")
    ax.set_title("Exact DP decomposition: " r"$\sum_j \Delta_j = $ DP gap", fontweight="bold")
    handles = [Patch(color=COL[k], label=k) for k in ["proxy", "competency", "noise"]]
    handles.append(plt.Line2D([0], [0], ls="--", color="black", label="DP gap"))
    ax.legend(handles=handles, fontsize=7)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_scatter_RD(pls, proxy_truth, competency, path):
    """Scatter of reconstructability R vs |DP contribution|; proxies sit top-right."""
    names = pls["names"]; R = pls["R"]; absd = np.abs(pls["delta"])
    colors = [COL[_ftype(n, proxy_truth, competency)] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(R, absd, c=colors, s=70, edgecolor="white", zorder=3)
    for n, x, yv in zip(names, R, absd):
        if (proxy_truth.get(n, False)) or yv > np.quantile(absd, 0.8):
            ax.annotate(n, (x, yv), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axvline(0.3, ls=":", color="gray"); ax.axhline(np.quantile(absd, 0.6), ls=":", color="gray")
    ax.set_xlabel("Reconstructability  $R_j = 2|AUC(x_j\\to A)-0.5|$")
    ax.set_ylabel(r"|DP contribution|  $|\Delta_j|$")
    ax.set_title("Proxy quadrant: high $R_j$ AND high $|\\Delta_j|$", fontweight="bold")
    handles = [Patch(color=COL[k], label=k) for k in ["proxy", "competency", "noise"]]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_pairplot(X, A, proxy_truth, competency, path, n_sample=1200, seed=0):
    """Pair-plot of 2 proxy + 2 competency features coloured by A."""
    rng = np.random.default_rng(seed)
    proxies = [c for c in X.columns if proxy_truth.get(c, False)][:2]
    comps = [c for c in X.columns if _ftype(c, proxy_truth, competency) == "competency"][:2]
    cols = proxies + comps
    if len(cols) < 2:
        cols = list(X.columns)[:4]
    idx = rng.choice(len(X), min(n_sample, len(X)), replace=False)
    d = X.iloc[idx][cols].copy()
    d["group"] = np.where(A[idx] == 1, "A=1", "A=0")
    g = sns.pairplot(d, hue="group", palette={"A=0": COL["A0"], "A=1": COL["A1"]},
                     plot_kws=dict(s=12, alpha=0.5, edgecolor="none"),
                     diag_kind="kde", corner=False, height=1.7)
    g.figure.suptitle("Pair-plot: proxies separate A (top-left block), "
                       "competencies do not", y=1.02, fontweight="bold", fontsize=11)
    g.figure.savefig(path, bbox_inches="tight"); plt.close(g.figure)


def fig_triad(before, after, path, title="Before / After: Fair . Faith . Trust"):
    """Grouped bars across the three pillars."""
    metrics = [
        ("DP gap\n(lower better)", before["fair"]["DP_gap"], after["fair"]["DP_gap"]),
        ("1-DI\n(lower better)", 1 - before["fair"]["DI"], 1 - after["fair"]["DI"]),
        ("EOO gap\n(lower better)", before["fair"]["EOO_gap"], after["fair"]["EOO_gap"]),
        ("ECE\n(lower better)", before["trust"]["ECE"], after["trust"]["ECE"]),
        ("1-Faith.suff\n(context)", 1 - min(1, before["faith"]["suff_ratio"]),
                                    1 - min(1, after["faith"]["suff_ratio"])),
    ]
    labels = [m[0] for m in metrics]
    b = [m[1] for m in metrics]; a = [m[2] for m in metrics]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(x - w/2, b, w, label="Before (unaware, proxy leak)", color=COL["before"])
    ax.bar(x + w/2, a, w, label="After (PCLN)", color=COL["after"])
    for i, (bb, aa) in enumerate(zip(b, a)):
        ax.text(i - w/2, bb, f"{bb:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w/2, aa, f"{aa:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontweight="bold"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_tradeoff(points, path):
    """Fairness-utility frontier. points: list of (label, AUC, DP, marker, color)."""
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for label, a, dp, mk, c in points:
        ax.scatter(dp, a, s=130, marker=mk, color=c, edgecolor="white", zorder=3, label=label)
        ax.annotate(label, (dp, a), fontsize=8, xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("DP gap  (lower = fairer)")
    ax.set_ylabel("AUC  (higher = better utility)")
    ax.set_title("Fairness-utility frontier", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_density(p_before, p_after, A, path):
    """Probability-density of predictions by group, before vs after (cf. FairJob Fig.5)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, p, name in [(axes[0], p_before, "Before (unaware, proxy leak)"),
                        (axes[1], p_after, "After (PCLN)")]:
        for a, c, lab in [(0, COL["A0"], "A=0"), (1, COL["A1"], "A=1")]:
            sns.kdeplot(p[A == a], ax=ax, fill=True, alpha=0.4, color=c, label=lab, clip=(0, 1))
        ax.set_title(name, fontweight="bold"); ax.set_xlabel("P(positive)"); ax.legend(fontsize=8)
    fig.suptitle("Predicted-probability density by protected group", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_localization_roc(curves, path):
    """ROC of PLS vs ground-truth proxy labels, one curve per proxy_strength.
    curves: list of (label, truth, scores)."""
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for label, truth, scores in curves:
        fpr, tpr, _ = roc_curve(truth, scores)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={sk_auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Proxy localisation: PLS vs ground truth", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_sensitivity(strengths, dp_before, dp_after, ap, path):
    """DP (before/after) and localisation AP vs proxy strength."""
    fig, ax1 = plt.subplots(figsize=(7.0, 4.6))
    ax1.plot(strengths, dp_before, "o-", color=COL["before"], label="DP before")
    ax1.plot(strengths, dp_after, "s-", color=COL["after"], label="DP after (PCLN)")
    ax1.set_xlabel("Proxy strength  $\\rho$"); ax1.set_ylabel("DP gap")
    ax1.legend(loc="upper left", fontsize=8)
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(strengths, ap, "^--", color="#6A4C93", label="Localisation AP")
    ax2.set_ylabel("Average precision", color="#6A4C93"); ax2.set_ylim(0, 1.02)
    ax2.legend(loc="lower right", fontsize=8)
    ax1.set_title("Sensitivity to proxy strength", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_mode_comparison(mode_cmp, path, before=None):
    """Grouped bars: after-DP and after-AUC for drop / suppress / orthogonalize.

    mode_cmp: {mode: {metric: {'mean':..,'std':..}}}. Shows that orthogonalize
    matches the others on DP while preserving the most utility. If `before`
    (dict with AUC/DP_gap of the unaware model) is given, it is drawn as a
    reference line so the utility cost of each neutraliser is visible.
    """
    modes = ["drop", "suppress", "orthogonalize"]
    labels = ["Drop", "Suppress", "Orthogonalize\n(ours)"]
    dp = [mode_cmp[m]["DP_gap"]["mean"] for m in modes]
    dp_e = [mode_cmp[m]["DP_gap"]["std"] for m in modes]
    auc = [mode_cmp[m]["AUC"]["mean"] for m in modes]
    auc_e = [mode_cmp[m]["AUC"]["std"] for m in modes]
    x = np.arange(len(modes))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    cols = ["#9AA0A6", "#6A4C93", COL["after"]]
    axL.bar(x, dp, yerr=dp_e, color=cols, capsize=4, edgecolor="white")
    if before is not None:
        axL.axhline(before["DP_gap"], ls="--", color=COL["before"], lw=1.3,
                    label=f"before = {before['DP_gap']:.3f}")
        axL.legend(fontsize=8)
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=9)
    axL.set_ylabel("DP gap after (lower = fairer)")
    axL.set_title("Fairness after neutralisation", fontweight="bold")
    for xi, v in zip(x, dp):
        axL.text(xi, v + max(dp) * 0.03, f"{v:.3f}", ha="center", fontsize=8)
    axR.bar(x, auc, yerr=auc_e, color=cols, capsize=4, edgecolor="white")
    if before is not None:
        axR.axhline(before["AUC"], ls="--", color="#2E86AB", lw=1.3,
                    label=f"before = {before['AUC']:.3f}")
        axR.legend(fontsize=8, loc="lower left")
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=9)
    axR.set_ylabel("AUC after (higher = better)")
    axR.set_ylim(min(auc + ([before["AUC"]] if before else [])) - 0.05, 1.0)
    axR.set_title("Utility after neutralisation", fontweight="bold")
    for xi, v in zip(x, auc):
        axR.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("Neutraliser comparison on mixed proxies (mean over seeds)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_pcc_effects(pcc, path):
    """Individual proxy-cluster counterfactual effects.

    Left: distribution of e(x)=f(x)-f(x^CF) by protected group -- the directed
    cluster counterfactual separates the groups (the leak suppresses A=1 and
    inflates A=0). Right: directional flip/necessity rates.
    """
    arr = pcc["_arrays"]; eff = arr["effect"]; A = arr["Ate"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.0),
                                   gridspec_kw={"width_ratios": [1.4, 1.0]})
    axL.hist(eff[A == 0], bins=45, alpha=0.6, color=COL["A0"], density=True, label="A=0 (advantaged)")
    axL.hist(eff[A == 1], bins=45, alpha=0.6, color=COL["A1"], density=True, label="A=1 (disadvantaged)")
    axL.axvline(0, color="gray", lw=1, ls="--")
    axL.set_xlabel(r"cluster-CF effect $e(x)=f(x)-f(x^{\mathrm{CF}})$")
    axL.set_ylabel("density"); axL.legend(fontsize=8)
    axL.set_title("Individual cluster-CF effect by group", fontweight="bold")

    bars = [("necessary\n(neutralize\u2192flip)", pcc["necessity_A1"], COL["after"]),
            ("sufficient\n(cluster alone)", pcc["sufficiency_A1"], "#6A4C93"),
            ("CAUSED\n(nec.\u2227suf.)", pcc["caused_A1"], COL["proxy"]),
            ("overall\nflip", pcc["flip_rate"], COL["noise"])]
    xs = np.arange(len(bars))
    axR.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], edgecolor="white")
    axR.set_xticks(xs); axR.set_xticklabels([b[0] for b in bars], fontsize=8)
    axR.set_ylabel("rate (disadvantaged group)")
    axR.set_title("Decision certificates (A=1 rejected)", fontweight="bold")
    for x, b in zip(xs, bars):
        axR.text(x, b[1] + 0.004, f"{b[1]:.3f}", ha="center", fontsize=8)
    fig.suptitle("Proxy-Cluster Counterfactuals (PCC)", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_pcc_corollary(pairs, path):
    """Corollary consistency: per-seed E[e|A=1]-E[e|A=0] vs sum_{j in P} Delta_j.
    pairs: list of (cluster_delta_prop1, effect_group_diff).
    """
    pairs = np.asarray(pairs, dtype=float)
    x, y = pairs[:, 0], pairs[:, 1]
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color="gray", lw=1,
            label="identity")
    ax.scatter(x, y, s=90, color=COL["after"], edgecolor="white", zorder=3)
    ax.set_xlabel(r"Prop. 1 cluster term $\sum_{j\in P}\Delta_j$")
    ax.set_ylabel(r"PCC group difference $\mathbb{E}[e|A{=}1]-\mathbb{E}[e|A{=}0]$")
    ax.set_title("Corollary: PCC averages to the\nProposition-1 cluster term",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_causation_surface(rho_rows, mix_rows, path):
    """Necessity / sufficiency / causation as the proxy regime changes.

    rho_rows: list of (rho, nec, suf, caused)  -- pure-A proxies, varying strength.
    mix_rows: list of (n_semantic, nec, suf, caused) -- mixed proxies.
    Shows that sufficiency saturates for strong pure leaks (caused = necessity)
    but drops for mixed proxies, making the causal certificate conservative
    exactly when the channel also carries legitimate signal.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    r = np.array(rho_rows, dtype=float)
    axL.plot(r[:, 0], r[:, 1], "o-", color=COL["after"], label="necessity")
    axL.plot(r[:, 0], r[:, 2], "s-", color="#6A4C93", label="sufficiency")
    axL.plot(r[:, 0], r[:, 3], "^-", color=COL["proxy"], label="caused (nec.\u2227suf.)")
    axL.set_xlabel(r"proxy strength $\rho$ (pure-A)")
    axL.set_ylabel("rate (A=1 rejected)"); axL.set_ylim(0, 1.05)
    axL.legend(fontsize=8, loc="center right")
    axL.set_title("Strong pure leaks: sufficiency saturates", fontweight="bold")

    m = np.array(mix_rows, dtype=float)
    x = np.arange(len(m)); w = 0.26
    axR.bar(x - w, m[:, 1], w, color=COL["after"], label="necessity")
    axR.bar(x, m[:, 2], w, color="#6A4C93", label="sufficiency")
    axR.bar(x + w, m[:, 3], w, color=COL["proxy"], label="caused")
    axR.set_xticks(x); axR.set_xticklabels([f"{int(k)} mixed" for k in m[:, 0]], fontsize=9)
    axR.set_ylim(0, 1.05); axR.set_ylabel("rate (A=1 rejected)")
    axR.legend(fontsize=8, loc="upper right")
    axR.set_title("Mixed proxies: sufficiency drops \u2192 conservative cause",
                  fontweight="bold")
    fig.suptitle("Necessity, sufficiency, and causation across proxy regimes",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_mediation(per_feature_iv, per_feature_cd, truth_proxies, spur_prefix,
                  share_iv, share_cd, path):
    """Interventional vs conditional Shapley for NDE/NIE attribution.

    Left: |Delta_j| per feature under each estimator, coloured by type
    (true proxy / spurious A-correlate / legitimate). Right: the share of total
    |Delta| concentrated on the TRUE proxies -- interventional is faithful,
    conditional disperses credit onto correlated non-causal features.
    """
    names = list(per_feature_iv.keys())
    def kind(n):
        if n in truth_proxies: return "proxy"
        if n.startswith(spur_prefix): return "spurious"
        return "legit"
    col = {"proxy": COL["proxy"], "spurious": "#C9A227", "legit": COL["competency"]}
    order = sorted(names, key=lambda n: -abs(per_feature_iv[n]))[:14]
    x = np.arange(len(order)); w = 0.4
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3),
                                   gridspec_kw={"width_ratios": [2.1, 1]})
    axL.bar(x - w/2, [abs(per_feature_iv[n]) for n in order], w,
            color=[col[kind(n)] for n in order], label="interventional", edgecolor="white")
    axL.bar(x + w/2, [abs(per_feature_cd[n]) for n in order], w,
            color=[col[kind(n)] for n in order], alpha=0.55, hatch="//",
            label="conditional", edgecolor="white")
    axL.set_xticks(x); axL.set_xticklabels(order, rotation=55, ha="right", fontsize=7.5)
    axL.set_ylabel(r"$|\Delta_j|$ (contribution to gap)")
    axL.set_title("Per-feature attribution: solid=interventional, hatched=conditional",
                  fontweight="bold", fontsize=10)
    from matplotlib.patches import Patch
    axL.legend(handles=[Patch(color=col["proxy"], label="true proxy"),
                        Patch(color=col["spurious"], label="spurious A-correlate"),
                        Patch(color=col["legit"], label="legitimate")],
               fontsize=8, loc="upper right")
    axR.bar([0, 1], [share_iv, share_cd], color=[COL["after"], "#9AA0A6"], edgecolor="white")
    axR.set_xticks([0, 1]); axR.set_xticklabels(["interventional\n(ours)", "conditional"], fontsize=9)
    axR.set_ylabel("share of |\u0394| on TRUE proxies"); axR.set_ylim(0, 1)
    for xi, v in zip([0, 1], [share_iv, share_cd]):
        axR.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    axR.set_title("Faithful NIE localization", fontweight="bold", fontsize=10)
    fig.suptitle("Interventional vs. conditional Shapley for proxy attribution (NDE/NIE)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_decision_map(pcc, path):
    """Advanced per-applicant decision map.

    Each test applicant is placed at (factual score, cluster-CF effect e(x)) and
    coloured by group. The vertical line is the hiring threshold; the shaded band
    is the 'flip zone' where neutralizing the proxy moves an applicant across it.
    Applicants whose decision is CAUSED (necessary AND sufficient) by the proxy
    channel are ringed. This makes the geometry of proxy causation visible: the
    disadvantaged who sit just below the line only because the leak pushed them
    there, and the advantaged who sit just above it for the same reason.
    """
    a = pcc["_arrays"]
    p0 = np.asarray(a["p0"]); eff = np.asarray(a["effect"])
    A = np.asarray(a["Ate"]); d0 = np.asarray(a["d0"])
    d1 = np.asarray(a["d1"]); dsuf = np.asarray(a["d_suf"])
    thr = pcc["select_threshold"]
    caused = ((d1 != d0) & (dsuf == d0))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    # flip zone: applicants within +-|eff| of threshold can cross it
    ax.axvspan(thr - 0.0, thr + 0.0, color="none")
    ax.axvline(thr, color="#333", lw=1.4, ls="--", zorder=2)
    ax.axhline(0, color="gray", lw=0.8, zorder=1)
    for g, c, lab in [(0, COL["A0"], "A=0 advantaged"), (1, COL["A1"], "A=1 disadvantaged")]:
        m = (A == g) & (~caused)
        ax.scatter(p0[m], eff[m], s=12, color=c, alpha=0.35, edgecolor="none",
                   label=lab, zorder=2)
    # ringed: proxy-caused decisions
    mc = caused
    ax.scatter(p0[mc], eff[mc], s=46, facecolor="none",
               edgecolor=COL["proxy"], linewidths=1.5, zorder=4,
               label="proxy-CAUSED decision")
    ax.annotate("hiring\nthreshold", xy=(thr, ax.get_ylim()[1]*0.86),
                xytext=(thr + 0.06, ax.get_ylim()[1]*0.86), fontsize=8,
                color="#333", va="center")
    ax.text(0.02, 0.95, "inflated\n(leak raised score)", transform=ax.transAxes,
            fontsize=8, color=COL["A0"], va="top")
    ax.text(0.02, 0.16, "suppressed\n(leak lowered score)", transform=ax.transAxes,
            fontsize=8, color=COL["A1"], va="bottom")
    ax.set_xlabel("factual hiring score  $f(x)$  (probability)")
    ax.set_ylabel(r"cluster-CF effect  $e(x)=f(x)-f(x^{\mathrm{CF}})$")
    ax.set_title("Decision map: where proxy discrimination lives",
                 fontweight="bold")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_localizer_comparison(rows, path):
    """Compare explanation/importance methods at localizing the proxy channel.

    rows: list of (method, AP_mean, AP_std, ROC_mean, ROC_std).
    Permutation importance (default XAI) conflates predictive with
    discriminatory; pure A-association admits spurious correlates; the
    contribution- and PLS-based rankers localize the true proxies faithfully.
    """
    rows = list(rows)
    names = [r[0] for r in rows]
    ap = [r[1] for r in rows]; aps = [r[2] for r in rows]
    rc = [r[3] for r in rows]; rcs = [r[4] for r in rows]
    y = np.arange(len(names))[::-1]
    cols = [COL["after"] if "ours" in n else
            (COL["proxy"] if ("Permutation" in n) else COL["competency"]) for n in names]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
    axL.barh(y, ap, xerr=aps, color=cols, edgecolor="white", capsize=3)
    axL.set_yticks(y); axL.set_yticklabels(names, fontsize=9)
    axL.set_xlim(0, 1.05); axL.set_xlabel("Average precision (proxy recovery)")
    axL.set_title("Localization AP \u2014 higher is better", fontweight="bold", fontsize=10)
    for yi, v in zip(y, ap):
        axL.text(min(v + 0.02, 1.0), yi, f"{v:.2f}", va="center", fontsize=8)
    axR.barh(y, rc, xerr=rcs, color=cols, edgecolor="white", capsize=3)
    axR.set_xlim(0, 1.05); axR.set_xlabel("ROC-AUC (proxy recovery)")
    axR.set_title("Localization ROC-AUC", fontweight="bold", fontsize=10)
    for yi, v in zip(y, rc):
        axR.text(min(v + 0.02, 1.0), yi, f"{v:.2f}", va="center", fontsize=8)
    for ax in (axL, axR):
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    fig.suptitle("Which explanation localizes the proxy channel?",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_before_after(arrays, path):
    """Paired before/after view of the score distribution by group.

    Left: factual (unaware) model scores, showing the group gap. Right: repaired
    model scores, with the gap closed. The shared threshold line marks the hiring
    cutoff. This visualizes the data the audit acts on, before and after the
    targeted repair.
    """
    p0 = np.asarray(arrays["p_un"]); p1 = np.asarray(arrays["p_pc"])
    A = np.asarray(arrays["Ate"])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)
    for ax, p, title in [(axL, p0, "Before: unaware model"),
                         (axR, p1, "After: targeted repair")]:
        ax.hist(p[A == 0], bins=40, alpha=0.6, color=COL["A0"], density=True,
                label="A=0 advantaged")
        ax.hist(p[A == 1], bins=40, alpha=0.6, color=COL["A1"], density=True,
                label="A=1 disadvantaged")
        m0, m1 = p[A == 0].mean(), p[A == 1].mean()
        ax.axvline(m0, color=COL["A0"], ls="--", lw=1.3)
        ax.axvline(m1, color=COL["A1"], ls="--", lw=1.3)
        ax.annotate("", xy=(m0, ax.get_ylim()[1]*0.74), xytext=(m1, ax.get_ylim()[1]*0.74),
                    arrowprops=dict(arrowstyle="<->", color="#444", lw=1.0))
        ax.text((m0+m1)/2, ax.get_ylim()[1]*0.80, f"gap {abs(m0-m1):.2f}",
                ha="center", fontsize=8, color="#333")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("model score (probability)")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axL.set_ylabel("density"); axR.legend(fontsize=8, loc="upper center")
    fig.suptitle("Group score distributions, before and after the repair",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_proxy_sweep(sweep, path):
    """DP before/after, AUC and localization AP across proxy strength."""
    rho = [r["rho"] for r in sweep]
    dpb = [r["dp_before"] for r in sweep]; dpa = [r["dp_after"] for r in sweep]
    auc = [r["auc_after"] for r in sweep]; ap = [r["loc_ap"] for r in sweep]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(rho, dpb, "o-", color=COL["proxy"], lw=2, label="DP gap before")
    ax.plot(rho, dpa, "s-", color=COL["after"], lw=2, label="DP gap after repair")
    ax.fill_between(rho, dpa, dpb, color=COL["after"], alpha=0.10)
    ax.set_xlabel("proxy strength $\\rho$"); ax.set_ylabel("demographic-parity gap")
    ax.set_ylim(0, max(dpb) * 1.15)
    ax2 = ax.twinx()
    ax2.plot(rho, auc, "^--", color=COL["competency"], lw=1.6, label="AUC after")
    ax2.plot(rho, ap, "d:", color="#888", lw=1.6, label="localization AP")
    ax2.set_ylabel("AUC / localization AP"); ax2.set_ylim(0.5, 1.02)
    for sp in ["top"]:
        ax.spines[sp].set_visible(False)
    l1, b1 = ax.get_legend_handles_labels(); l2, b2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, b1 + b2, fontsize=8, loc="center right", framealpha=0.9)
    ax.set_title("Repair is uniform across proxy strength", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_real_summary(rows, path):
    """Grouped before/after DP bars across real-data audits.

    rows: list of (label, dp_before, dp_after).
    """
    labels = [r[0] for r in rows]
    before = [r[1] for r in rows]; after = [r[2] for r in rows]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.bar(x - w/2, before, w, color=COL["proxy"], label="before", edgecolor="white")
    ax.bar(x + w/2, after, w, color=COL["after"], label="after repair", edgecolor="white")
    for xi, (bb, aa) in enumerate(zip(before, after)):
        ax.text(xi - w/2, bb + 0.004, f"{bb:.3f}", ha="center", fontsize=8)
        ax.text(xi + w/2, aa + 0.004, f"{aa:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("demographic-parity gap")
    ax.set_title("Real-data audits: parity gap before and after",
                 fontweight="bold")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_frontier(rows, path):
    """Fairness-utility frontier: DP gap (x) vs fair-label accuracy (y)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    styles = {
        "Unaware (no repair)": dict(c="#9aa3ad", m="o"),
        "Group thresholds [base]": dict(c="#7a6cc4", m="s"),
        "Reweighing [base]": dict(c="#c98a2b", m="^"),
        "Blanket repair (retrain)": dict(c=COL["competency"], m="D"),
        "Selective (ours, certified)": dict(c=COL["proxy"], m="*"),
    }
    for r in rows:
        st = styles.get(r["method"], dict(c="#888", m="o"))
        big = "ours" in r["method"]
        ax.scatter(r["DP"], r["fair_acc"], s=320 if big else 130,
                   marker=st["m"], color=st["c"], zorder=3,
                   edgecolor="white", linewidth=1.4)
        dy = 0.0016 if not big else -0.0034
        ax.annotate(r["method"].replace(" [base]", "").replace(" (retrain)", ""),
                    (r["DP"], r["fair_acc"]), fontsize=8.5,
                    xytext=(6, 6 if not big else -12), textcoords="offset points",
                    fontweight="bold" if big else "normal")
    ax.set_xlabel("demographic-parity gap  (lower is fairer)")
    ax.set_ylabel("accuracy vs. fair label  (higher is better)")
    ax.set_title("Recovering fair decisions at low disparity",
                 fontweight="bold")
    ax.invert_xaxis()  # fairer (smaller gap) to the right
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.18)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
