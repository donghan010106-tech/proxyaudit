# Architecture & method notes

## Pipeline (`pipeline.run_pcln`)

```
unaware model f  ──►  explain (SHAP, score space)  ──►  Δⱼ  (Proposition 1)
                                                          │
                              reconstructability Rⱼ ──────┤
                                                          ▼
                                       PLSⱼ = √(Rⱼ·Dⱼ)  ──►  knee + gate  ──►  cluster P
                                                          │
                       ┌──────────────────────────────────┤
                       ▼                                   ▼
        neutralize(P)  →  retrain f'  (AFTER)     proxy-cluster counterfactual (PCC)
                       │                                   │
                       ▼                                   ▼
         Fair · Faith · Trust triad            necessity / sufficiency / causation
```

## Core objects

| module | object | one-line definition |
|---|---|---|
| `pls` | `Δⱼ` | `E[φⱼ\|A=1] − E[φⱼ\|A=0]`; **exact** per-feature DP contribution (Prop. 1) |
| `pls` | `PLSⱼ` | `√(Rⱼ·Dⱼ)`; high only if feature reconstructs `A` **and** drives the gap |
| `neutralize` | orthogonalize | remove the `A`-direction *inside* the cluster; needs no `A` at inference |
| `counterfactual` | `e(x)` | `f(x) − f(xᶜᶠ)`; individual cluster-CF effect |
| `counterfactual` | necessity | neutralize cluster ⇒ decision flips |
| `counterfactual` | sufficiency | cluster alone (on an average profile) reproduces the decision |
| `counterfactual` | causation | necessary **and** sufficient ⇒ per-decision *cause* |

## Key guarantees

- **Proposition 1** (exact, tested in `tests/test_proposition1.py`): `Σⱼ Δⱼ = DPᶠ` in score space, by SHAP local accuracy.
- **Corollary** (tested in `tests/test_counterfactual.py`): `E[e\|A=1] − E[e\|A=0] = Σ_{j∈P} Δⱼ` — the individual PCC effect averages back to the population cluster term.

## Dependency policy

The package runs on numpy / pandas / scikit-learn / matplotlib / seaborn / scipy only. `shap`, `xgboost`, `streamlit`, `datasets` are optional extras; the code degrades gracefully (e.g. a dependency-free Monte-Carlo Shapley estimator stands in for `shap`).
