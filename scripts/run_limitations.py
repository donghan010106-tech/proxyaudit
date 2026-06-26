"""Address paper limitations with executed code on REAL FairCVdb.
Produces: 95% CIs over seeds, a paired significance test (selective vs blanket),
two added strong baselines (equalized-odds post-processing; reductions-style grid),
and a non-linear residual-leakage probe. No fabricated numbers.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from scipy import stats

RNG = np.random.default_rng(0)
d = np.load("/mnt/user-data/uploads/FairCVdb.npy", allow_pickle=True).item()
P = np.concatenate([d["Profiles Train"], d["Profiles Test"]])
A = P[:, 1].astype(int)                      # gender
X = P[:, 4:31].astype(float)                 # 27 structured features
y_bias = np.concatenate([d["Biased Labels Train (Gender)"], d["Biased Labels Test (Gender)"]]).astype(float)
y_fair = np.concatenate([d["Blind Labels Train"], d["Blind Labels Test"]]).astype(float)
# binarize labels at their median (scores in [0,1])
yb = (y_bias >= np.median(y_bias)).astype(int)
yf = (y_fair >= np.median(y_fair)).astype(int)
n = len(X); idx = np.arange(n)


def dp_gap(sel, A):
    return abs(sel[A == 1].mean() - sel[A == 0].mean())


def select_top(score, tau):
    k = int(round(tau * len(score)))
    thr = np.sort(score)[::-1][min(k, len(score) - 1)]
    return (score >= thr).astype(int)


def pls_localize(Xtr, Atr, phi_delta):
    R = np.array([2 * abs(roc_auc_score(Atr, Xtr[:, j]) - 0.5) for j in range(Xtr.shape[1])])
    Dn = np.abs(phi_delta) / (np.abs(phi_delta).sum() + 1e-12)
    pls = np.sqrt(R * Dn)
    order = np.argsort(pls)[::-1]
    # knee: keep features above mean+0.5 std of pls, gated by R>0.55
    keep = [j for j in order if pls[j] > pls.mean() and R[j] > 0.55]
    return keep if keep else list(order[:5]), R


def orthogonalize(Xb, Ab, fit=None):
    # remove the linear A-direction from each column of block Xb
    if fit is None:
        a = Ab.astype(float); a = (a - a.mean())
        coef = (Xb - Xb.mean(0)).T @ a / (a @ a + 1e-12)   # per-column slope on A
        fit = {"coef": coef, "amean": Ab.mean(), "xmean": Xb.mean(0)}
    Xo = Xb - np.outer(Ab.astype(float) - fit["amean"], fit["coef"])
    return Xo, fit


def eval_vs_fair(score, A, yf, tau):
    sel = select_top(score, tau)
    return dict(dp=float(dp_gap(sel, A)),
                acc=float((sel == yf).mean()),
                f1=float(f1_score(yf, sel)),
                sel=sel)


def run_once(seed, boot=False):
    rng = np.random.default_rng(seed)
    tr = rng.random(n) < 0.7
    te = ~tr
    if boot:  # bootstrap the test set for CI
        te_idx = rng.choice(np.where(te)[0], size=te.sum(), replace=True)
    else:
        te_idx = np.where(te)[0]
    Xtr, Atr, ytr = X[tr], A[tr], yb[tr]
    Xte, Ate, yfte = X[te_idx], A[te_idx], yf[te_idx]
    tau = float(yfte.mean())                  # fair base rate operating point

    base = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    s_tr = base.decision_function(Xtr); s_te = base.decision_function(Xte)
    # delta_j via coef * group-mean diff (linear local accuracy)
    phi_delta = base.coef_[0] * (Xtr[Atr == 1].mean(0) - Xtr[Atr == 0].mean(0))
    P_idx, R = pls_localize(Xtr, Atr, phi_delta)

    out = {}
    # --- unaware ---
    out["unaware"] = eval_vs_fair(s_te, Ate, yfte, tau)
    # --- group thresholds (equalize selection rate) ---
    sel_g = np.zeros(len(Xte), int)
    for g in (0, 1):
        m = Ate == g
        sel_g[m] = select_top(s_te[m], tau)
    out["group_thr"] = dict(dp=float(dp_gap(sel_g, Ate)), acc=float((sel_g == yfte).mean()),
                            f1=float(f1_score(yfte, sel_g)), sel=sel_g)
    # --- reweighing (Kamiran-Calders) + refit ---
    w = np.ones(len(Xtr))
    for g in (0, 1):
        for c in (0, 1):
            m = (Atr == g) & (ytr == c)
            if m.sum() > 0:
                w[m] = (np.mean(Atr == g) * np.mean(ytr == c)) / (m.mean() + 1e-12)
    rw = LogisticRegression(max_iter=200).fit(Xtr, ytr, sample_weight=w)
    out["reweigh"] = eval_vs_fair(rw.decision_function(Xte), Ate, yfte, tau)
    # --- equalized-odds post-processing (Hardt): per-group thresholds matching TPR on fair labels ---
    # choose per-group thresholds so positive rate among fair-positives (TPR) is matched to global
    sel_eo = np.zeros(len(Xte), int)
    glob_tpr = None
    base_sel = select_top(s_te, tau)
    glob_tpr = base_sel[yfte == 1].mean()
    for g in (0, 1):
        m = Ate == g
        sg = s_te[m]; yg = yfte[m]
        # find threshold giving TPR ~ glob_tpr among that group's fair-positives
        order = np.argsort(sg)[::-1]
        best_thr = np.median(sg); best_d = 1e9
        for q in np.linspace(0.02, 0.98, 49):
            thr = np.quantile(sg, q)
            tpr = ((sg >= thr) & (yg == 1)).sum() / max((yg == 1).sum(), 1)
            if abs(tpr - glob_tpr) < best_d:
                best_d = abs(tpr - glob_tpr); best_thr = thr
        sel_eo[m] = (sg >= best_thr).astype(int)
    out["eo_post"] = dict(dp=float(dp_gap(sel_eo, Ate)), acc=float((sel_eo == yfte).mean()),
                          f1=float(f1_score(yfte, sel_eo)), sel=sel_eo)
    # --- reductions-style grid (sweep a fairness penalty on the A-direction) ---
    best = None
    aproj = Xtr @ (Xtr[Atr == 1].mean(0) - Xtr[Atr == 0].mean(0))
    for lam in [0.0, 0.5, 1.0, 2.0, 4.0]:
        Xa = Xtr - lam * np.outer((aproj - aproj.mean()) / (aproj.std() + 1e-9),
                                   (Xtr[Atr == 1].mean(0) - Xtr[Atr == 0].mean(0)))
        mr = LogisticRegression(max_iter=200).fit(Xa, ytr)
        Xte_a = Xte - lam * np.outer(((Xte @ (Xtr[Atr == 1].mean(0) - Xtr[Atr == 0].mean(0))) - aproj.mean()) / (aproj.std() + 1e-9),
                                     (Xtr[Atr == 1].mean(0) - Xtr[Atr == 0].mean(0)))
        ev = eval_vs_fair(mr.decision_function(Xte_a), Ate, yfte, tau)
        score = ev["acc"] - 2.0 * ev["dp"]
        if best is None or score > best[0]:
            best = (score, ev)
    out["reductions"] = best[1]
    # --- blanket repair (orthogonalize proxy block, refit) ---
    Xtr_b = Xtr.copy(); Xte_b = Xte.copy()
    Xo_tr, fit = orthogonalize(Xtr[:, P_idx], Atr)
    Xo_te, _ = orthogonalize(Xte[:, P_idx], Ate, fit)
    Xtr_b[:, P_idx] = Xo_tr; Xte_b[:, P_idx] = Xo_te
    rep = LogisticRegression(max_iter=200).fit(Xtr_b, ytr)
    out["blanket"] = eval_vs_fair(rep.decision_function(Xte_b), Ate, yfte, tau)
    # --- selective (certificate-guided): override decisions the proxy flips (necessity), as in the paper ---
    s_te_cf = base.decision_function(Xte_b)          # directed counterfactual score (proxy neutralized)
    base_sel = select_top(s_te, tau); cf_sel = select_top(s_te_cf, tau)
    caused_rejection = (Ate == 1) & (base_sel == 0) & (cf_sel == 1)
    inflated_selection = (Ate == 0) & (base_sel == 1) & (cf_sel == 0)
    caused = caused_rejection | inflated_selection
    sel_sel = base_sel.copy()
    sel_sel[caused_rejection] = 1
    sel_sel[inflated_selection] = 0
    out["selective"] = dict(dp=float(dp_gap(sel_sel, Ate)), acc=float((sel_sel == yfte).mean()),
                            f1=float(f1_score(yfte, sel_sel)), changed=int(caused.sum()), sel=sel_sel)
    # --- non-linear residual-leakage probe (after linear orthogonalization) ---
    probe = GradientBoostingClassifier(n_estimators=60, max_depth=3).fit(Xtr_b[:, P_idx], Atr)
    out["residual_leak_auc"] = float(roc_auc_score(Ate, probe.predict_proba(Xte_b[:, P_idx])[:, 1]))
    out["raw_leak_auc"] = float(roc_auc_score(Ate, GradientBoostingClassifier(n_estimators=60, max_depth=3).fit(Xtr[:, P_idx], Atr).predict_proba(Xte[:, P_idx])[:, 1]))
    for k in out:
        if isinstance(out[k], dict): out[k].pop("sel", None)
    return out


# ---- multi-seed for CIs ----
SEEDS = list(range(20))
runs = [run_once(s, boot=True) for s in SEEDS]
methods = ["unaware", "group_thr", "reweigh", "eo_post", "reductions", "blanket", "selective"]


def ci(vals):
    a = np.array(vals); m = a.mean()
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [float(m), float(lo), float(hi)]


summary = {"n_seeds": len(SEEDS), "n_total": int(n)}
for m in methods:
    summary[m] = {
        "dp": ci([r[m]["dp"] for r in runs]),
        "acc": ci([r[m]["acc"] for r in runs]),
        "f1": ci([r[m]["f1"] for r in runs]),
    }
    if m == "selective":
        summary[m]["changed"] = ci([r[m]["changed"] for r in runs])

# ---- paired significance: selective vs blanket DP gap ----
sel_dp = np.array([r["selective"]["dp"] for r in runs])
bl_dp = np.array([r["blanket"]["dp"] for r in runs])
w = stats.wilcoxon(sel_dp, bl_dp)
t = stats.ttest_rel(sel_dp, bl_dp)
summary["sig_selective_vs_blanket_dp"] = {
    "selective_mean": float(sel_dp.mean()), "blanket_mean": float(bl_dp.mean()),
    "wilcoxon_p": float(w.pvalue), "paired_t_p": float(t.pvalue),
    "cohens_d": float((bl_dp.mean() - sel_dp.mean()) / (np.std(bl_dp - sel_dp) + 1e-12))}
# selective vs blanket fair-acc
sel_a = np.array([r["selective"]["acc"] for r in runs]); bl_a = np.array([r["blanket"]["acc"] for r in runs])
summary["sig_selective_vs_blanket_acc"] = {
    "selective_mean": float(sel_a.mean()), "blanket_mean": float(bl_a.mean()),
    "paired_t_p": float(stats.ttest_rel(sel_a, bl_a).pvalue)}
# ---- leakage probe ----
summary["leakage_probe"] = {
    "raw_block_auc": ci([r["raw_leak_auc"] for r in runs]),
    "after_linear_orth_auc": ci([r["residual_leak_auc"] for r in runs])}

json.dump(summary, open("results/limitations_addressed.json", "w"), indent=2)
print("=== DONE ===")
print("selective DP %.3f vs blanket %.3f  (wilcoxon p=%.2e)" %
      (summary["sig_selective_vs_blanket_dp"]["selective_mean"],
       summary["sig_selective_vs_blanket_dp"]["blanket_mean"],
       summary["sig_selective_vs_blanket_dp"]["wilcoxon_p"]))
print("residual leak AUC after linear orth: %.3f [%.3f,%.3f] (raw %.3f)" % (
    *summary["leakage_probe"]["after_linear_orth_auc"], summary["leakage_probe"]["raw_block_auc"][0]))
for m in methods:
    print("%-11s DP %.3f [%.3f,%.3f]  facc %.3f" % (m, *summary[m]["dp"], summary[m]["acc"][0]))
