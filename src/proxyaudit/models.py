"""Model factory + training regimes.

Regimes (mirroring FairJob's terminology and extending it):
  * 'unfair'  : train on all features INCLUDING the protected attribute A.
  * 'unaware' : train without A (fairness through unawareness)  -- the baseline
                that both FairCVtest and FairJob show is insufficient.
  * 'pcln'    : train on the PCLN-neutralised features (our method).

Estimators use only scikit-learn so the pipeline runs offline. The optional
`xgboost` backend is documented in the README for the user's larger runs.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def make_model(kind="lr", seed=0, class_weight="balanced"):
    if kind == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0,
                               class_weight=class_weight, random_state=seed))
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=250,
            l2_regularization=1.0, random_state=seed,
            class_weight=class_weight)
    if kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2,
                          learning_rate_init=5e-4, max_iter=300,
                          random_state=seed))
    raise ValueError(kind)


def linear_part(model):
    """Return the underlying linear estimator for exact SHAP, or None."""
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "coef_"):
                return step
    if hasattr(model, "coef_"):
        return model
    return None


def transform_for_explainer(model, X):
    """Apply any preprocessing (scaler) so attributions live in model space."""
    if hasattr(model, "named_steps"):
        Xt = X
        for name, step in model.named_steps.items():
            if step is list(model.named_steps.values())[-1]:
                break
            Xt = step.transform(Xt)
        return np.asarray(Xt)
    return np.asarray(X)
