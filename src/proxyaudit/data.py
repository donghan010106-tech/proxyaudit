"""Data layer for ProxyAudit.

This module provides FOUR data entry points, with explicit honesty about
what is real and what is synthetic:

  1. load_faircv_real(csv_path)
        Loads the *real* FairCVdb (24,000 synthetic-but-released profiles) once
        the user has exported FairCVdb.npy -> FairCVdb.csv (see scripts/).
        Columns follow the official FairCVtest schema.

  2. load_fairjob_real(parquet_or_hf)
        Loads the *real* Criteo FairJob dataset (1,072,226 rows) from a local
        parquet/csv or directly from the HuggingFace hub. Requires network +
        `datasets` (only used by the user, never fabricated here).

  3. make_faircv_recipe(...)
        A FAITHFUL re-implementation of the *documented* FairCVtest generative
        recipe for the STRUCTURED competency stream (US-Census education bands,
        discrete competencies, blind score = linear combo + Gaussian noise,
        biased score = group penalty). On top of the documented recipe we inject
        an EXPLICIT, GROUND-TRUTH proxy channel (a block of "embedding-like"
        features that encode the protected attribute to a controllable degree).
        Because the proxy indices are known, this testbed lets us *validate*
        whether PCLN recovers the true leakage channel. This is the primary
        empirical testbed of the paper and is fully reproducible offline.

  4. make_fairjob_sim(...)
        A FairJob-CALIBRATED synthetic twin (click target, protected_attribute,
        senior, rank, displayrandom, cat/num blocks, ~0.7% positives, selection
        bias) matched to the published FairJob statistics (Tables 4-8 of
        Vladimirova et al., 2024). Used ONLY to demonstrate the transfer protocol
        and to drive the Streamlit preview. It is NEVER presented as real FairJob
        results; the real-world anchors in the paper are the *published* baseline
        numbers, and `scripts/run_fairjob.py` reproduces PCLN on the real data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Column conventions
# ----------------------------------------------------------------------------
COMPETENCY = ["suitability", "educ_attainment", "prev_experience",
              "recommendation", "availability",
              "lang_prof_1", "lang_prof_2", "lang_prof_3"]
COMP_NAMES = ["Suitability", "Education", "Experience", "Recommendation",
              "Availability", "Language 1", "Language 2", "Language 3"]


@dataclass
class Dataset:
    """A lightweight container used across the pipeline."""
    X: pd.DataFrame                 # feature matrix (all model-visible features)
    y: np.ndarray                   # binary target
    A: np.ndarray                   # binary protected attribute
    feature_names: list             # ordered feature names
    proxy_truth: dict = field(default_factory=dict)  # name -> True if injected proxy
    meta: dict = field(default_factory=dict)         # free-form metadata
    name: str = "dataset"

    @property
    def n(self):
        return len(self.y)

    def split(self, test_frac=0.2, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n)
        cut = int(self.n * (1 - test_frac))
        tr, te = idx[:cut], idx[cut:]
        return tr, te


# ============================================================================
# 1 & 3.  FairCVtest -- real loader + faithful recipe generator
# ============================================================================
def make_faircv_recipe(n=12000, seed=0,
                       proxy_strength=0.85, n_proxy=8, n_noise=8,
                       bias_penalty=0.07, protected="gender",
                       n_semantic=0, semantic_leak=0.6, semantic_signal=0.8,
                       legit_corr=0.0, legit_diff_n=0, legit_diff_shift=0.45,
                       legit_diff_noise=2.2, legit_diff_weight=0.9):
    """Reproduce the documented FairCVtest structured recipe + a ground-truth
    proxy channel.

    Documented recipe (Pena et al., CVPRW 2020; BiDAlab/FairCVtest README):
      * 2 demographic attributes; here we expose ONE binary protected attribute
        A (gender: 0 male / 1 female) for clarity. Demographics are *agnostic*
        to the blind score by construction.
      * 5 competency blocks -> 8 discrete competency features on documented grids.
      * blind score = normalised linear combination of competencies + suitability
        + small Gaussian noise  (demographic-agnostic == FAIR ground truth).
      * biased score = blind score with a penalty applied to one group
        (the documented Gender/Ethnicity bias injection).

    Injected proxy channel (our addition, ground truth KNOWN):
      * n_proxy "embedding-like" features each = proxy_strength * sign(A) + noise.
        These emulate how ResNet face embeddings / biased labels leak gender
        into otherwise-neutral feature blocks. Their indices are recorded in
        `proxy_truth` so PCLN's localisation can be scored.
      * n_noise pure-noise features (distractors, no A signal).
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, size=n)  # 0/1 protected attribute

    # latent legitimate competency, observable only via semantic proxies (if any)
    g_latent = rng.normal(0, 1, size=n) if n_semantic > 0 else None

    # --- documented competency grids -------------------------------------
    suitability = rng.choice([0.25, 0.5, 0.75, 1.0], size=n)
    educ = rng.choice([0.2, 0.4, 0.6, 0.8, 1.0], size=n,
                      p=[0.10, 0.20, 0.35, 0.25, 0.10])  # ~US-census shaped
    prev_exp = rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=n)
    recommendation = rng.integers(0, 2, size=n).astype(float)
    availability = rng.choice([0.2, 0.4, 0.6, 0.8, 1.0], size=n)
    lang1 = rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=n)
    lang2 = rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=n)
    lang3 = rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=n)

    comp = np.column_stack([suitability, educ, prev_exp, recommendation,
                            availability, lang1, lang2, lang3])

    # --- blind (fair) score: linear combo + gaussian noise ----------------
    w = np.array([0.20, 0.18, 0.18, 0.10, 0.08, 0.10, 0.08, 0.08])
    blind = comp @ w + rng.normal(0, 0.03, size=n)
    if g_latent is not None:
        blind = blind + 0.15 * g_latent      # latent legitimate competency
    blind = (blind - blind.min()) / (blind.max() - blind.min())

    # --- biased score: documented group penalty ---------------------------
    biased = blind.copy()
    biased[A == 1] -= bias_penalty           # penalise group A==1 (e.g. female)
    biased = np.clip(biased, 0, 1)

    # --- injected proxy channel (GROUND TRUTH) ----------------------------
    a_signed = np.where(A == 1, 1.0, -1.0)
    proxy_block = (proxy_strength * a_signed[:, None]
                   + rng.normal(0, np.sqrt(max(1e-6, 1 - proxy_strength**2)),
                                size=(n, n_proxy)))
    noise_block = rng.normal(0, 1.0, size=(n, n_noise))

    # Optional SPURIOUS correlate(s): correlated with A (hence with the proxy
    # block) but with NO causal effect on the label. A linear model gives them
    # ~zero coefficient, so *interventional* Shapley attributes ~0 to them
    # (correct). *Conditional* Shapley, respecting correlations, smears proxy
    # credit onto them -- the failure mode that biases an NDE/NIE split.
    spur_block = None
    n_spur = 2 if legit_corr > 0 else 0
    if n_spur:
        spur_block = (legit_corr * a_signed[:, None]
                      + np.sqrt(max(1e-6, 1 - legit_corr ** 2))
                      * rng.normal(0, 1, size=(n, n_spur)))

    # Optional LEGITIMATE group-differential feature(s): a real, highly weighted
    # competency with a small genuine group mean shift but large within-group
    # variance. It therefore has a LARGE contribution to the gap (high |Delta|,
    # because the model weight is high) yet LOW reconstructability (the groups
    # barely separate on it). A |Delta|-only ranker flags it as a proxy; the PLS
    # reconstructability gate correctly rejects it. It is genuinely predictive of
    # the label, so it is not part of the proxy ground truth.
    legit_diff_block = None
    if legit_diff_n > 0:
        legit_diff_block = (legit_diff_shift * a_signed[:, None]
                            + legit_diff_noise * rng.normal(0, 1, size=(n, legit_diff_n)))
        # make it genuinely predictive: add its (weighted) signal to the score
        blind = blind + legit_diff_weight * legit_diff_block.mean(axis=1)
        blind = (blind - blind.min()) / (blind.max() - blind.min())
        biased = blind.copy(); biased[A == 1] -= bias_penalty; biased = np.clip(biased, 0, 1)

    # --- semantic (mixed) proxies: leak A *and* carry legitimate signal ---
    # A latent competency `g` genuinely raises the score (affects y) but is
    # observable ONLY through the semantic proxy block, so dropping that block
    # destroys real predictive signal. Each semantic feature mixes the latent
    # with an A-leak:  sem = semantic_leak*sign(A) + semantic_signal*g + noise.
    # Orthogonalisation strips the A-aligned part while retaining g.
    semantic_block = None
    if n_semantic > 0:
        semantic_block = (semantic_leak * a_signed[:, None]
                          + semantic_signal * g_latent[:, None]
                          + rng.normal(0, 0.3, size=(n, n_semantic)))

    # --- assemble feature frame ------------------------------------------
    cols = list(COMPETENCY)
    data = {c: comp[:, i] for i, c in enumerate(COMPETENCY)}
    for j in range(n_proxy):
        name = f"emb_{j}"
        data[name] = proxy_block[:, j]
        cols.append(name)
    for j in range(n_noise):
        name = f"noise_{j}"
        data[name] = noise_block[:, j]
        cols.append(name)
    sem_names = []
    if semantic_block is not None:
        for j in range(n_semantic):
            name = f"sem_{j}"
            data[name] = semantic_block[:, j]
            cols.append(name); sem_names.append(name)
    spur_names = []
    if spur_block is not None:
        for j in range(spur_block.shape[1]):
            name = f"spur_{j}"
            data[name] = spur_block[:, j]
            cols.append(name); spur_names.append(name)
    legit_diff_names = []
    if legit_diff_block is not None:
        for j in range(legit_diff_block.shape[1]):
            name = f"legitdiff_{j}"
            data[name] = legit_diff_block[:, j]
            cols.append(name); legit_diff_names.append(name)
    X = pd.DataFrame(data, columns=cols)

    # target: we model the *biased* label (the realistic deployment target),
    # binarised at its median so classes are balanced.
    thr = np.median(biased)
    y = (biased >= thr).astype(int)

    proxy_truth = {c: (c.startswith("emb_") or c.startswith("sem_")) for c in cols}
    meta = dict(blind=blind, biased=biased, protected=protected,
                bias_penalty=bias_penalty, proxy_strength=proxy_strength,
                competency=COMPETENCY)
    return Dataset(X=X, y=y, A=A, feature_names=cols,
                   proxy_truth=proxy_truth, meta=meta, name="FairCV-recipe")


def load_faircv_real(csv_path, protected="gender", label="biased_label_gender"):
    """Load the real FairCVdb.csv (24k profiles).  Schema follows the official
    FairCVtest release (see scripts/export_faircv_csv.py to build the CSV from
    the LFS .npy). The 20-D face embeddings act as the (real) proxy channel;
    here we have no per-feature ground truth, so proxy_truth is left empty.
    """
    df = pd.read_csv(csv_path)
    A = df[protected].values.astype(int)
    thr = df["blind_label"].median()
    y = (df[label] >= thr).astype(int).values
    feat = COMPETENCY + [f"face_emb_{i}" for i in range(20)]
    feat = [c for c in feat if c in df.columns]
    X = df[feat].copy()
    return Dataset(X=X, y=y, A=A, feature_names=feat,
                   proxy_truth={}, meta=dict(protected=protected),
                   name="FairCVdb-real")


# ============================================================================
# 2 & 4.  FairJob -- real loader + calibrated synthetic twin
# ============================================================================
def make_fairjob_sim(n=40000, seed=0, click_rate=0.012, proxy_strength=0.6,
                     n_cat=13, n_num=20, n_proxy_num=5):
    """A FairJob-CALIBRATED synthetic twin (DISCLOSED, not real FairJob).

    Calibrated to published FairJob statistics (Vladimirova et al., 2024):
      * protected_attribute ~ 50/50
      * senior positives ~ 66%
      * displayrandom ~ 10%
      * strong class imbalance (click positives < 1%; we use a slightly higher
        rate so the offline demo trains quickly while preserving the regime)
      * a proxy channel: a subset of numerical features encodes A.

    NOTE: every metric computed on this twin is a *protocol demonstration*, not
    a measurement of the real dataset. The real anchors are the published
    baseline numbers; `scripts/run_fairjob.py` reproduces PCLN on the real data.
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, size=n)
    senior = (rng.random(n) < 0.666).astype(int)
    displayrandom = (rng.random(n) < 0.099).astype(int)
    rank = rng.integers(1, 41, size=n)

    a_signed = np.where(A == 1, 1.0, -1.0)
    cats = {f"cat{j}": rng.integers(0, [9, 9, 1025, 98, 122, 1296, 2492, 3183,
                                        3541, 2879, 2314, 1436, 912][j], size=n)
            for j in range(n_cat)}
    nums = {}
    proxy_names = []
    for j in range(n_num):
        if j < n_proxy_num:
            v = proxy_strength * a_signed + rng.normal(
                0, np.sqrt(max(1e-6, 1 - proxy_strength**2)), size=n)
            proxy_names.append(f"num{16+j}")
        else:
            v = rng.normal(0, 1, size=n)
        nums[f"num{16+j}"] = v

    # latent click utility: legitimate signal from a few nums + senior + rank
    legit = (0.8 * nums["num31"] if "num31" in nums else 0.0)
    legit = (0.6 * nums[f"num{16+n_proxy_num}"] + 0.5 * nums[f"num{16+n_proxy_num+1}"]
             + 0.3 * senior - 0.04 * rank)
    # historical bias: group A==0 (female) slightly less likely to be served/click senior
    bias_term = -0.5 * ((A == 1) & (senior == 1))
    logit = -4.2 + legit + bias_term + rng.normal(0, 0.5, size=n)
    p = 1 / (1 + np.exp(-logit))
    # rescale to target click rate
    p = p * (click_rate / p.mean())
    y = (rng.random(n) < np.clip(p, 0, 1)).astype(int)

    data = {"protected_attribute": A, "senior": senior,
            "displayrandom": displayrandom, "rank": rank}
    data.update(cats)
    data.update(nums)
    cols = ["senior", "rank"] + list(cats.keys()) + list(nums.keys())
    X = pd.DataFrame({c: data[c] for c in cols}, columns=cols)
    proxy_truth = {c: (c in proxy_names) for c in cols}
    meta = dict(protected="gender_proxy", senior=senior, rank=rank,
                displayrandom=displayrandom, click_rate=float(y.mean()))
    return Dataset(X=X, y=y, A=A, feature_names=cols,
                   proxy_truth=proxy_truth, meta=meta, name="FairJob-sim")


def load_fairjob_real(source="data/fairjob.csv.gz", protected="protected_attribute",
                      drop_ids=True, nrows=None):
    """Load the *real* FairJob dataset, matching the official schema.

    The official release is ``fairjob.csv.gz``: a header row followed by numeric
    columns addressed BY POSITION (see criteo-research/fairjob-dataset
    ``functions.py::load_data``):

        col 0 = click (label)         col 3 = displayrandom
        col 1 = protected_attribute   col 4 = rank
        col 2 = senior                col 5+ = features X

    ``source`` may be that ``.csv``/``.csv.gz`` file (positional), a ``.parquet``
    with named columns, or a HuggingFace id. The protected attribute and click
    are removed from the model-visible features (unaware baseline) and returned
    separately. The senior slice is kept in ``meta`` for the parity estimate the
    FairJob benchmark prescribes.
    """
    senior = displayrandom = rank = None
    if source.endswith(".csv") or source.endswith(".csv.gz"):
        # official positional layout
        data = np.loadtxt(source, skiprows=1, delimiter=",", max_rows=nrows)
        y = data[:, 0].astype(int)
        A = data[:, 1].astype(int)
        senior = data[:, 2].astype(int)
        displayrandom = data[:, 3].astype(int)
        rank = data[:, 4]
        Xv = data[:, 5:]
        feat = [f"f{j}" for j in range(Xv.shape[1])]
        X = pd.DataFrame(Xv, columns=feat)
    else:
        if source.endswith(".parquet"):
            df = pd.read_parquet(source)
            if nrows:
                df = df.iloc[:nrows]
        else:
            from datasets import load_dataset  # user-side dependency
            df = load_dataset(source, split="train").to_pandas()
            if nrows:
                df = df.iloc[:nrows]
        A = df[protected].values.astype(int)
        y = df["click"].values.astype(int)
        for nm in ("senior", "displayrandom", "rank"):
            if nm in df:
                v = df[nm].values
                if nm == "senior":
                    senior = v.astype(int)
                elif nm == "displayrandom":
                    displayrandom = v.astype(int)
                else:
                    rank = v
        drop = [protected, "click", "senior", "displayrandom", "rank"]
        if drop_ids:
            drop += [c for c in ["user_id", "impression_id", "product_id"] if c in df.columns]
        feat = [c for c in df.columns if c not in drop]
        X = df[feat].copy()
    meta = dict(protected="gender_proxy", senior=senior, rank=rank,
                displayrandom=displayrandom)
    return Dataset(X=X, y=y, A=A, feature_names=list(X.columns),
                   proxy_truth={}, meta=meta, name="FairJob-real")


def load_faircv_npy(npy_path, config="gender", binarize="median"):
    """Load the official FairCVdb.npy dict directly (no CSV export needed).

    Matches ``FairCV.py::loadDataset`` for the gender experiment: the dict keys
    are ``Profiles Train/Test``, ``Blind/Biased Labels ...``. Demographic columns
    are ``profiles[:,0]=ethnicity`` and ``profiles[:,1]=gender``; the structured
    feature block is ``profiles[:,4:31]``, whose tail (the face-embedding dims)
    is the real proxy channel that encodes the protected attribute. We expose
    every structured feature to an unaware model and let PLS find the channel.
    """
    fair = np.load(npy_path, allow_pickle=True).item()
    p = np.concatenate([fair["Profiles Train"], fair["Profiles Test"]], axis=0)
    demo_col = 1 if config == "gender" else 0
    A = p[:, demo_col].astype(int)
    key = "Biased Labels Train (Gender)" if config == "gender" else "Biased Labels Train (Ethnicity)"
    keyte = key.replace("Train", "Test")
    score = np.concatenate([fair[key], fair[keyte]], axis=0).astype(float)
    thr = np.median(score) if binarize == "median" else float(binarize)
    y = (score >= thr).astype(int)
    Xv = p[:, 4:31].astype(float)            # structured competencies + face embedding
    feat = [f"feat_{j}" for j in range(Xv.shape[1])]
    X = pd.DataFrame(Xv, columns=feat)
    return Dataset(X=X, y=y, A=A, feature_names=feat, proxy_truth={},
                   meta=dict(protected=config), name="FairCVdb-real")
