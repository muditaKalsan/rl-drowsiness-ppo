"""
Export a compact Excel file that contains three sheets:

1. **info** - sample index, subject id, binary label.
2. **features** - average (over all chunks) of the scaled band‑power features for each sample.
3. **raw_sample0** - the *raw* EEG values of the first training sample (30 × 384).

The Excel workbook is written to `results/eeg_overview.xlsx`.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import RESULTS_DIR, DATA_DIR, SCALER_PATH, MAT_FILE
from features import get_splits
import scipy.io
import h5py

def _load_raw_eeg():
    """Load the original raw EEG matrix from the .mat file (fallback to h5py)."""
    try:
        mat = scipy.io.loadmat(str(MAT_FILE), squeeze_me=True)
        eeg = mat["EEGsample"]
    except NotImplementedError:
        with h5py.File(str(MAT_FILE), "r") as f:
            eeg = np.array(f["EEGsample"]).transpose()
    return eeg

def main():
    # Load split data (features are already scaled)
    splits = get_splits()
    X_train, y_train, subj_train = splits["train"]

    # ---- Sheet 1: info ----
    info_df = pd.DataFrame({
        "sample_idx": np.arange(len(y_train)),
        "subject_id": subj_train,
        "label": y_train,
    })

    # ---- Sheet 2: averaged features ----
    avg_feat = X_train.mean(axis=1)  # (samples, n_features)
    feat_df = pd.DataFrame(avg_feat)
    feat_df.insert(0, "sample_idx", np.arange(len(y_train)))

    # ---- Sheet 3: raw EEG of first training sample ----
    raw_eeg = _load_raw_eeg()
    raw_sample0 = raw_eeg[0]  # shape (30, 384)
    raw_df = pd.DataFrame(raw_sample0)

    # Write to Excel
    out_path = RESULTS_DIR / "eeg_overview.xlsx"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        info_df.to_excel(writer, sheet_name="info", index=False)
        feat_df.to_excel(writer, sheet_name="features", index=False)
        raw_df.to_excel(writer, sheet_name="raw_sample0", index=False)
    print(f"[INFO] Excel file written to {out_path}")

if __name__ == "__main__":
    main()
