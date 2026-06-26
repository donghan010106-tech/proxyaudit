# ProxyAudit

**Localize the proxy channel that leaks a protected attribute, neutralize it surgically, and certify the repair — at the population *and* the individual level.**

[![ci](https://github.com/your-org/proxyaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/proxyaudit/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

Two influential benchmarks — **FairCVtest** (CVPRW 2020) and **FairJob** (NeurIPS 2024 D&B) — both show that *fairness through unawareness fails*: deleting a protected attribute `A` does not delete its influence, because the model re-learns it through correlated **proxy** features (`A → X → Ŷ`). They establish *that* proxies exist. **ProxyAudit** (the PCLN method) answers the three questions they leave open, and adds a fourth:

1. **Where** is the leak? — per-feature **Proxy Leakage Score** localization.
2. **How much** does each feature contribute? — an **exact** demographic-parity decomposition (Proposition 1).
3. **How** to remove it without destroying legitimate signal? — **targeted** neutralization (orthogonalize).
4. **Was *this* decision caused by it?** — **proxy-cluster counterfactuals** with per-decision *necessity*, *sufficiency*, and *causation* certificates.

## Headline results (controlled testbed, ground-truth proxies, 5 seeds)

| | |
|---|---|
| Localization | precision@k = ROC-AUC = AP = **1.00** |
| Repair | DP gap **0.193 → 0.016 (−92%)**, DI **0.68 → 0.97**, AUC cost **≤ 0.02** |
| Orthogonalize vs delete (mixed proxies) | AUC **0.958** vs **0.777** at equal fairness |
| Corollary (PCC ↔ DP) | individual effect averages to the cluster term, error **0.03** |
| Per-decision cause | **15.7%** of disadvantaged rejections are *necessary ∧ sufficient*-caused by the proxy channel |
| Faithful attribution | interventional Shapley keeps **0.63** of credit on true proxies vs conditional's **0.39** (both exact) |
| Honest tension | DP-targeted repair **raises** the equal-opportunity gap (reported, not hidden) |

## Install

```bash
git clone https://github.com/your-org/proxyaudit && cd proxyaudit
pip install -e .            # core only (no shap/xgboost/streamlit needed)
# or:  pip install -e ".[app,dev]"   # + Streamlit app + tests
```

## Quickstart

```bash
proxyaudit-synth                       # one-line audit + necessity/sufficiency cert
make synth                             # reproduce all results + paper figures
make app                               # role-based interactive auditor (Streamlit)
make test                              # run the test suite
make paper                             # build the PDF (XeLaTeX)
```

```python
from proxyaudit.data import make_faircv_recipe
from proxyaudit.pipeline import run_pcln
from proxyaudit.counterfactual import individual_certificate

ds  = make_faircv_recipe(n=12000, seed=0)
out = run_pcln(ds, kind="lr", neutralize_mode="orthogonalize")
print(out["after_fair"]["DP_gap"], out["pcc"]["caused_A1"])
print(individual_certificate(out["pcc"], i=0)["verdict"])
```

## The interactive app (two roles)

`streamlit run app/streamlit_app.py`

- **Recruiter / auditor** — fairness + bias + utility dashboard, proxy-channel localization, who-is-selected before/after, and a population necessity/sufficiency/causation certificate.
- **Job-seeker / applicant** — inspect one application: were you *suppressed* or *inflated* by proxy bias? Is the proxy channel **necessary**, **sufficient**, or a **cause** of your decision? What would a *fair* model decide?

## Audit the real datasets (your licensed copies)

```bash
python scripts/export_faircv_csv.py FairCVdb.npy data/FairCVdb.csv
python scripts/run_faircv_real.py data/FairCVdb.csv
python scripts/run_fairjob.py --real criteo/FairJob --nrows 200000
```

## Repository layout

```
src/proxyaudit/        the installable package
  data.py              testbed recipe (+ injected ground-truth proxies), real loaders, FairJob twin
  shapley.py           LinearSHAP (interventional, exact) + SamplingSHAP + GaussianConditionalSHAP
  pls.py               Proposition 1 decomposition, PLS, knee+gate selection
  neutralize.py        drop / suppress / orthogonalize
  counterfactual.py    PCC: necessity, sufficiency, causation, corollary, per-individual certificate
  mediation.py         NDE/NIE split + interventional-vs-conditional Shapley contrast
  fairness.py · faithfulness.py · trust.py    triad metrics
  pipeline.py          run_pcln: localize → neutralize → triad → PCC
  viz.py               all figures
  cli.py               `proxyaudit-synth` console entry point
scripts/               run_synth · run_faircv_real · run_fairjob · export_faircv_csv
app/streamlit_app.py   role-based interactive auditor
tests/                 pytest suite (Prop. 1 identity, localization, PCC corollary, …)
paper/                 main.tex (XeLaTeX, 12 pp.), refs.bib, figures/
docs/                  architecture & method notes
```

## A note on honesty / reproducibility

All **measured** numbers come from `scripts/run_synth.py` on the controlled FairCVtest-recipe testbed (the original FairCVtest is itself synthetic, so reproducing its documented recipe is faithful). Because the raw FairCVdb embeddings and the million-row FairJob log are not bundled, the **real-world anchors** are FairJob's *published* baselines (unaware XGB AUC ≈ 0.758, DP ≈ 0.0028); the FairJob-calibrated twin is **disclosed** and used only as a protocol demo / app preview, **never** as a real measurement. No benchmark number was fabricated. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation

See [`CITATION.cff`](CITATION.cff). Paper: *Causal Certificates for Proxy Discrimination: An Exact Bridge from Demographic Parity to Per-Decision Necessity and Sufficiency* (`paper/main.pdf`).
