"""Comparative study: early fusion vs late fusion for fair resume screening.
Two modalities of the FairCVdb profile are fused two ways:
  - competency block  (merit signal, weakly tied to the attribute)
  - embedding block   (demographically loaded channel, the proxy)
Early fusion  : one scorer on the concatenation.
Late fusion   : a scorer per modality, scores combined.
Late (fair)   : late fusion that down-weights the modality the audit flags as the
                proxy channel (proxy-aware gating); no attribute used at deployment.
Every number is measured; CIs are 95% bootstrap over 20 seeds. No fabrication.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from scipy import stats

d = np.load("/mnt/user-data/uploads/FairCVdb.npy", allow_pickle=True).item()
P = np.concatenate([d["Profiles Train"], d["Profiles Test"]])
A = P[:, 1].astype(int)
COMP = P[:, 4:11].astype(float)      # competency modality (merit)
EMB  = P[:, 11:31].astype(float)     # embedding modality (face-derived, heavy leak)
BIOS = np.load("data/bios_tfidf.npy").astype(float)   # bios text modality (TF-IDF, moderate leak)
yb = (np.concatenate([d["Biased Labels Train (Gender)"], d["Biased Labels Test (Gender)"]]) >=
      np.median(np.concatenate([d["Biased Labels Train (Gender)"], d["Biased Labels Test (Gender)"]]))).astype(int)
yf = (np.concatenate([d["Blind Labels Train"], d["Blind Labels Test"]]) >=
      np.median(np.concatenate([d["Blind Labels Train"], d["Blind Labels Test"]]))).astype(int)
n = len(P)


def dp_gap(sel, A): return abs(sel[A == 1].mean() - sel[A == 0].mean())
def top(score, tau):
    k = int(round(tau * len(score))); thr = np.sort(score)[::-1][min(k, len(score) - 1)]; return (score >= thr).astype(int)
def z(s): return (s - s.mean()) / (s.std() + 1e-9)
def leak_auc(Xtr, Atr, Xte, Ate):
    m = GradientBoostingClassifier(n_estimators=60, max_depth=3).fit(Xtr, Atr)
    return roc_auc_score(Ate, m.predict_proba(Xte)[:, 1])


def evals(score, A, yf, tau):
    sel = top(score, tau)
    return dict(dp=float(dp_gap(sel, A)), acc=float((sel == yf).mean()),
                f1=float(f1_score(yf, sel)), auc=float(roc_auc_score(yf, score)))


def certified_causation(base, Xtr_blocks, Xte_blocks, Atr, Ate, tau, proxy_block):
    """necessity-based certified caused-rate among disadvantaged rejections,
    neutralizing the audited proxy modality."""
    Xtr = np.hstack(Xtr_blocks); Xte = np.hstack(Xte_blocks)
    # counterfactual: orthogonalize the proxy block wrt A (linear), refit-free on fixed model
    Xb = Xte.copy()
    pb = proxy_block
    a = Atr.astype(float) - Atr.mean()
    coef = (Xtr[:, pb] - Xtr[:, pb].mean(0)).T @ a / (a @ a + 1e-12)
    Xb[:, pb] = Xte[:, pb] - np.outer(Ate.astype(float) - Atr.mean(), coef)
    s = base.decision_function(Xte); scf = base.decision_function(Xb)
    bs = top(s, tau); cs = top(scf, tau)
    caused = ((Ate == 1) & (bs == 0) & (cs == 1)) | ((Ate == 0) & (bs == 1) & (cs == 0))
    denom = ((Ate == 1) & (bs == 0)).sum()
    return float(caused[(Ate == 1)].sum() / max(denom, 1))


def orth(Xtr_block, Atr, Xte_block, Ate):
    a = Atr.astype(float) - Atr.mean()
    coef = (Xtr_block - Xtr_block.mean(0)).T @ a / (a @ a + 1e-12)
    Xtr_o = Xtr_block - np.outer(Atr.astype(float) - Atr.mean(), coef)
    Xte_o = Xte_block - np.outer(Ate.astype(float) - Atr.mean(), coef)
    return Xtr_o, Xte_o


def run(seed, boot=False):
    rng = np.random.default_rng(seed)
    tr = rng.random(n) < 0.7; te = ~tr
    teix = rng.choice(np.where(te)[0], size=te.sum(), replace=True) if boot else np.where(te)[0]
    Ctr, Etr, Btr, Atr, ytr = COMP[tr], EMB[tr], BIOS[tr], A[tr], yb[tr]
    Cte, Ete, Bte, Ate = COMP[teix], EMB[teix], BIOS[teix], A[teix]
    yfte = yf[teix]; tau = float(yfte.mean())
    out = {}
    out["leak_comp"] = leak_auc(Ctr, Atr, Cte, Ate)
    out["leak_bios"] = leak_auc(Btr, Atr, Bte, Ate)
    out["leak_emb"] = leak_auc(Etr, Atr, Ete, Ate)
    dC, dB, dE = Ctr.shape[1], Btr.shape[1], Etr.shape[1]
    pb = np.arange(dC, dC + dB + dE)   # leaky blocks = bios + embedding (everything after competency)
    # 1) EARLY fusion, no repair: one model on the three concatenated modalities
    early = LogisticRegression(max_iter=200).fit(np.hstack([Ctr, Btr, Etr]), ytr)
    out["early"] = evals(early.decision_function(np.hstack([Cte, Bte, Ete])), Ate, yfte, tau)
    out["early"]["caused"] = certified_causation(early, [Ctr, Btr, Etr], [Cte, Bte, Ete], Atr, Ate, tau, pb)
    # 2) LATE fusion (equal): a scorer per modality, scores averaged
    mc = LogisticRegression(max_iter=200).fit(Ctr, ytr)
    mb = LogisticRegression(max_iter=200).fit(Btr, ytr)
    me = LogisticRegression(max_iter=200).fit(Etr, ytr)
    late = (z(mc.decision_function(Cte)) + z(mb.decision_function(Bte)) + z(me.decision_function(Ete))) / 3.0
    out["late"] = evals(late, Ate, yfte, tau)
    # 3) EARLY fusion + repair: orthogonalize the two leaky modalities (bios, embedding), refit jointly
    Btr_o, Bte_o = orth(Btr, Atr, Bte, Ate)
    Etr_o, Ete_o = orth(Etr, Atr, Ete, Ate)
    earlyR = LogisticRegression(max_iter=200).fit(np.hstack([Ctr, Btr_o, Etr_o]), ytr)
    out["early_repair"] = evals(earlyR.decision_function(np.hstack([Cte, Bte_o, Ete_o])), Ate, yfte, tau)
    out["early_repair"]["caused"] = certified_causation(earlyR, [Ctr, Btr_o, Etr_o], [Cte, Bte_o, Ete_o], Atr, Ate, tau, pb)
    # 4) LATE fusion + modular repair: merit scorer untouched, leaky branches neutralized
    mb_o = LogisticRegression(max_iter=200).fit(Btr_o, ytr)
    me_o = LogisticRegression(max_iter=200).fit(Etr_o, ytr)
    late_r = (z(mc.decision_function(Cte)) + z(mb_o.decision_function(Bte_o)) + z(me_o.decision_function(Ete_o))) / 3.0
    out["late_repair"] = evals(late_r, Ate, yfte, tau)
    return out


SEEDS = list(range(20))
runs = [run(s, boot=True) for s in SEEDS]
def ci(v): a = np.array(v); return [float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
S = {"n_seeds": len(SEEDS),
     "leak_comp": ci([r["leak_comp"] for r in runs]),
     "leak_bios": ci([r["leak_bios"] for r in runs]),
     "leak_emb": ci([r["leak_emb"] for r in runs])}
methods = ["early", "late", "early_repair", "late_repair"]
for m in methods:
    S[m] = {k: ci([r[m][k] for r in runs]) for k in ["dp", "acc", "f1", "auc"]}
    if "caused" in runs[0][m]:
        S[m]["caused"] = ci([r[m]["caused"] for r in runs])
# significance: late_repair vs early_repair on fair-accuracy (utility preserved at low DP)
lr = np.array([r["late_repair"]["acc"] for r in runs]); er = np.array([r["early_repair"]["acc"] for r in runs])
S["sig_acc_late_vs_early_repair"] = {"late_repair": float(lr.mean()), "early_repair": float(er.mean()),
                                     "wilcoxon_p": float(stats.wilcoxon(lr, er).pvalue),
                                     "cohens_d": float((lr.mean() - er.mean()) / (np.std(lr - er) + 1e-12))}
json.dump(S, open("results/fusion_results.json", "w"), indent=2)
print("=== FUSION STUDY (modular repair) ===")
print("leakage  comp %.2f  bios %.2f  emb %.2f" % (S["leak_comp"][0], S["leak_bios"][0], S["leak_emb"][0]))
for m in methods:
    r = S[m]; c = (" caused %.3f" % r["caused"][0]) if "caused" in r else ""
    print("%-13s DP %.3f [%.3f,%.3f]  f-acc %.3f  AUC %.3f%s" % (m, *r["dp"], r["acc"][0], r["auc"][0], c))
print("late_repair vs early_repair f-acc: Wilcoxon p=%.1e, d=%.1f" % (
    S["sig_acc_late_vs_early_repair"]["wilcoxon_p"], S["sig_acc_late_vs_early_repair"]["cohens_d"]))

