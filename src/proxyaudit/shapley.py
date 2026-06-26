"""Dependency-free feature attribution (SHAP-compatible).

We avoid a hard dependency on the `shap` package so the whole pipeline runs in
a minimal offline environment. Two estimators are provided:

  * LinearSHAP : exact Shapley values for linear / logistic models in the
                 model's score (logit) space:  phi_j(x) = w_j * (x_j - E[x_j]).
                 These satisfy local accuracy exactly:  f(x) = phi_0 + sum_j phi_j.

  * SamplingSHAP : Monte-Carlo Shapley values (Strumbelj & Kononenko, 2014) in
                   score space for arbitrary models, using a background sample
                   for the "absent feature" expectation. Suitable for the modest
                   feature counts in this study.

Both return attributions in SCORE space (logit), which is what the
demographic-parity decomposition in `pls.py` requires.

If the optional `shap` package is installed, `auto_explainer` will prefer it.
"""
from __future__ import annotations
import numpy as np


def _logit_score(model, X):
    """Return the model's additive score (logit) for the positive class."""
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return s
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class LinearSHAP:
    """Exact Shapley values for a fitted linear/logistic sklearn model."""

    def __init__(self, model, background):
        self.model = model
        self.coef = model.coef_.ravel().astype(float)
        self.intercept = float(np.ravel(model.intercept_)[0])
        self.bg_mean = np.asarray(background).mean(axis=0)
        self.base = self.intercept + float(self.coef @ self.bg_mean)

    def shap_values(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.bg_mean) * self.coef  # (n, d)

    @property
    def expected_value(self):
        return self.base


class SamplingSHAP:
    """Monte-Carlo Shapley values in score space for arbitrary models."""

    def __init__(self, model, background, n_perm=64, seed=0, max_bg=128):
        self.model = model
        bg = np.asarray(background, dtype=float)
        if len(bg) > max_bg:
            rng = np.random.default_rng(seed)
            bg = bg[rng.choice(len(bg), max_bg, replace=False)]
        self.bg = bg
        self.n_perm = n_perm
        self.rng = np.random.default_rng(seed)
        self.base = float(_logit_score(model, self.bg).mean())

    @property
    def expected_value(self):
        return self.base

    def _f(self, X):
        return _logit_score(self.model, X)

    def shap_values(self, X):
        """Monte-Carlo Shapley in score space.

        Mathematically identical to the per-step permutation estimator of
        Strumbelj & Kononenko (2014), but all (d+1) coalition states of every
        permutation for a given instance are stacked into a single batched
        model call, which is ~d*n_perm times fewer predict() invocations.
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        m = self.n_perm
        phi = np.zeros((n, d))
        bg = self.bg
        for i in range(n):
            x = X[i]
            orders = np.array([self.rng.permutation(d) for _ in range(m)])  # (m,d)
            refs = bg[self.rng.integers(len(bg), size=m)].copy()            # (m,d)
            # states[p,k] = ref_p with first k features (in order_p) set to x
            states = np.repeat(refs[:, None, :], d + 1, axis=1)             # (m,d+1,d)
            for p in range(m):
                op = orders[p]
                for k in range(1, d + 1):
                    states[p, k, op[:k]] = x[op[:k]]
            flat = states.reshape(m * (d + 1), d)
            F = self._f(flat).reshape(m, d + 1)                            # (m,d+1)
            diffs = F[:, 1:] - F[:, :-1]                                   # (m,d) marginal gains
            acc = np.zeros(d)
            for p in range(m):
                np.add.at(acc, orders[p], diffs[p])
            phi[i] = acc / m
        return phi


def auto_explainer(model, background, kind="auto", **kw):
    """Pick the best available explainer."""
    if kind == "linear" or (kind == "auto" and hasattr(model, "coef_")):
        return LinearSHAP(model, background)
    return SamplingSHAP(model, background, **kw)


class GaussianConditionalSHAP:
    """Observational (conditional) Shapley values via Gaussian conditional means.

    Interventional Shapley (LinearSHAP / marginal SamplingSHAP) fills an absent
    feature from its *marginal* distribution, breaking its correlation with the
    present features. Conditional Shapley instead respects the data manifold:
    an absent feature is replaced by its conditional mean given the present ones,
    E[x_{\\bar S} | x_S], under a Gaussian fit (mu, Sigma) of the background.

    For a model that is linear in score space this conditional-mean plug-in gives
    the *exact* conditional Shapley value (since E[f|x_S] = f(x_S, E[x_{\\bar S}|x_S]));
    for non-linear models it is the standard conditional-mean approximation.
    The contrast between the two estimators is what reveals proxy-credit smearing:
    conditional Shapley spreads a proxy's contribution onto correlated legitimate
    features, blurring the direct/indirect (NDE/NIE) separation.
    """

    def __init__(self, model, background, n_perm=48, seed=0, ridge=1e-3):
        bg = np.asarray(background, dtype=float)
        self.model = model
        self.mu = bg.mean(axis=0)
        d = bg.shape[1]
        self.Sigma = np.cov(bg, rowvar=False) + ridge * np.eye(d)
        self.n_perm = n_perm
        self.rng = np.random.default_rng(seed)
        self.base = float(_logit_score(model, self.mu[None, :])[0])

    @property
    def expected_value(self):
        return self.base

    def _cond_fill(self, x, present_mask):
        """Return a full vector: present features from x, absent set to their
        Gaussian conditional mean given the present ones."""
        S = np.where(present_mask)[0]
        Sb = np.where(~present_mask)[0]
        out = self.mu.copy()
        out[S] = x[S]
        if len(S) and len(Sb):
            Sigma_bs = self.Sigma[np.ix_(Sb, S)]
            Sigma_ss = self.Sigma[np.ix_(S, S)]
            try:
                sol = np.linalg.solve(Sigma_ss, (x[S] - self.mu[S]))
                out[Sb] = self.mu[Sb] + Sigma_bs @ sol
            except np.linalg.LinAlgError:
                pass
        return out

    def shap_values(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        phi = np.zeros((n, d))
        for i in range(n):
            x = X[i]
            acc = np.zeros(d)
            for _ in range(self.n_perm):
                order = self.rng.permutation(d)
                present = np.zeros(d, dtype=bool)
                states = [self._cond_fill(x, present.copy())]
                for j in order:
                    present[j] = True
                    states.append(self._cond_fill(x, present.copy()))
                F = _logit_score(self.model, np.array(states))
                gains = np.diff(F)
                np.add.at(acc, order, gains)
            phi[i] = acc / self.n_perm
        return phi


class EmpiricalConditionalSHAP:
    """Non-parametric (kNN) conditional Shapley, free of the Gaussian assumption.

    Where GaussianConditionalSHAP fills absent features from a Gaussian
    conditional mean, this estimator draws the conditional reference from the k
    nearest background rows in the PRESENT features and averages the model over
    them. It therefore makes no distributional assumption, which lets us check
    that the interventional-vs-conditional conclusion is not an artefact of the
    Gaussian fit on discrete competencies.
    """

    def __init__(self, model, background, n_perm=40, k=40, seed=0):
        self.model = model
        self.bg = np.asarray(background, dtype=float)
        self.mu = self.bg.mean(axis=0)
        self.std = self.bg.std(axis=0) + 1e-9
        self.n_perm = n_perm
        self.k = min(k, len(self.bg))
        self.rng = np.random.default_rng(seed)
        self.base = float(_logit_score(model, self.mu[None, :])[0])

    @property
    def expected_value(self):
        return self.base

    def _cond_value(self, x, present_mask):
        S = np.where(present_mask)[0]
        if len(S) == 0:
            return self.base
        # distance to background on present features (standardized)
        diff = (self.bg[:, S] - x[S]) / self.std[S]
        d = np.einsum("ij,ij->i", diff, diff)
        idx = np.argpartition(d, self.k - 1)[: self.k]
        ref = self.bg[idx].copy()
        ref[:, S] = x[S]                      # fix present features to x
        return float(_logit_score(self.model, ref).mean())

    def shap_values(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        phi = np.zeros((n, d))
        for i in range(n):
            x = X[i]; acc = np.zeros(d)
            for _ in range(self.n_perm):
                order = self.rng.permutation(d)
                present = np.zeros(d, dtype=bool)
                prev = self._cond_value(x, present)
                for j in order:
                    present[j] = True
                    cur = self._cond_value(x, present)
                    acc[j] += cur - prev
                    prev = cur
            phi[i] = acc / self.n_perm
        return phi
