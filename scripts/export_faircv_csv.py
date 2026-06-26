"""Export the released FairCVdb .npy archive to a flat CSV that
`proxyaudit.data.load_faircv_real` can read.

The FairCVtest release (https://github.com/BiDAlab/FairCVtest) ships
`FairCVdb.npy` as a git-LFS object (~200 MB). Once you have pulled the real
file (``git lfs pull``), run:

    python scripts/export_faircv_csv.py /path/to/FairCVdb.npy data/FairCVdb.csv

The .npy is a Python dict (saved with allow_pickle). Field names vary slightly
between releases, so we map the documented schema defensively and write only the
columns our loader expects: the 8 competency features, gender/ethnicity, the
blind and biased labels, and the 20-D face embedding (the real proxy channel).
"""
from __future__ import annotations
import sys, os
import numpy as np
import pandas as pd

COMP = ["suitability", "educ_attainment", "prev_experience", "recommendation",
        "availability", "lang_prof_1", "lang_prof_2", "lang_prof_3"]


def _get(d, *names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default


def main(npy_path, out_csv):
    raw = np.load(npy_path, allow_pickle=True)
    d = raw.item() if isinstance(raw, np.ndarray) and raw.dtype == object else raw
    if not isinstance(d, dict):
        raise SystemExit("Unexpected FairCVdb format; expected a pickled dict.")

    n = len(_get(d, "blind_label", "blindLabels", "scores_blind"))
    out = {}

    # competencies (a (n, 8) matrix under various names)
    comp = _get(d, "profiles", "competencies", "features")
    comp = np.asarray(comp)
    if comp.ndim == 2 and comp.shape[1] >= 8:
        for i, c in enumerate(COMP):
            out[c] = comp[:, i]

    # demographics
    out["gender"] = np.asarray(_get(d, "gender", "genders")).astype(int)
    eth = _get(d, "ethnicity", "ethnicities")
    if eth is not None:
        out["ethnicity"] = np.asarray(eth).astype(int)

    # labels
    out["blind_label"] = np.asarray(_get(d, "blind_label", "blindLabels", "scores_blind")).ravel()
    bg = _get(d, "biased_label_gender", "biasedLabels_gender", "scores_biased_gender")
    if bg is not None:
        out["biased_label_gender"] = np.asarray(bg).ravel()
    be = _get(d, "biased_label_ethnicity", "biasedLabels_ethnicity")
    if be is not None:
        out["biased_label_ethnicity"] = np.asarray(be).ravel()

    # face embeddings (n, 20) -> the real proxy channel
    emb = _get(d, "face_embeddings", "embeddings", "resnet_embeddings")
    if emb is not None:
        emb = np.asarray(emb)
        for i in range(min(20, emb.shape[1])):
            out[f"face_emb_{i}"] = emb[:, i]

    df = pd.DataFrame(out)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {df.shape[0]} rows x {df.shape[1]} cols")
    print("columns:", list(df.columns))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python scripts/export_faircv_csv.py FairCVdb.npy out.csv")
    main(sys.argv[1], sys.argv[2])
