"""
Utility functions to load the EEG dataset, detect the drowsy label, extract band‑power features,
scale them using a StandardScaler fitted only on training subjects, and cache everything.
"""

import numpy as np
import scipy.io
import h5py
from pathlib import Path
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
import joblib

from config import (
    DATA_DIR,
    MAT_FILE,
    FEATURES_PATH,
    LABELS_PATH,
    SUBJECTS_PATH,
    SCALER_PATH,
    FS,
    POINTS_PER_CHUNK,
    TRAIN_SUBJECTS,
)

# ----------------------------------------------------------------------
# Band definitions (Hz)
# ----------------------------------------------------------------------
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}

def _load_mat() -> dict:
    """Load the .mat file, trying scipy first, then h5py fallback."""
    try:
        mat = scipy.io.loadmat(str(MAT_FILE), squeeze_me=True)
        return mat
    except NotImplementedError as exc:
        if "v7.3" in str(exc):
            with h5py.File(str(MAT_FILE), "r") as f:
                eeg = np.array(f["EEGsample"]).transpose()
                substate = np.array(f["substate"]).squeeze()
                subindex = np.array(f["subindex"]).squeeze()
                return {"EEGsample": eeg, "substate": substate, "subindex": subindex}
        else:
            raise

def _detect_drowsy_label(labels: np.ndarray, eeg: np.ndarray) -> int:
    """Return the integer label (0 or 1) that has the higher (theta+alpha)/beta power ratio."""
    ratios = []
    for i in range(eeg.shape[0]):
        flat = eeg[i].ravel()
        f, psd = welch(flat, fs=FS, nperseg=FS*2)
        power = {}
        for band, (low, high) in BANDS.items():
            idx = np.logical_and(f >= low, f < high)
            power[band] = np.mean(psd[idx])
        ratio = (power["theta"] + power["alpha"]) / (power["beta"] + 1e-12)
        ratios.append(ratio)
    ratios = np.array(ratios)
    avg0 = ratios[labels == 0].mean()
    avg1 = ratios[labels == 1].mean()
    drowsy = 0 if avg0 > avg1 else 1
    print(f"[INFO] Detected drowsy label = {drowsy} (higher (θ+α)/β ratio)")
    return drowsy

def _extract_features(eeg: np.ndarray) -> np.ndarray:
    """Extract log10 band‑power features for each chunk.
    Returns array of shape (samples, n_chunks, n_channels * n_bands).
    """
    n_samples, n_channels, n_time = eeg.shape
    n_chunks = n_time // POINTS_PER_CHUNK
    n_bands = len(BANDS)
    feats = np.empty((n_samples, n_chunks, n_channels * n_bands), dtype=np.float32)

    for s in range(n_samples):
        for c in range(n_chunks):
            start = c * POINTS_PER_CHUNK
            stop = start + POINTS_PER_CHUNK
            chunk = eeg[s, :, start:stop]  # (30, pts)
            ch_feat = []
            for ch in range(n_channels):
                f, psd = welch(chunk[ch], fs=FS, nperseg=POINTS_PER_CHUNK)
                for low, high in BANDS.values():
                    idx = np.logical_and(f >= low, f < high)
                    power = np.mean(psd[idx])
                    ch_feat.append(np.log10(power + 1e-12))
            feats[s, c] = np.array(ch_feat, dtype=np.float32)
    return feats

def prepare_and_cache():
    """Load data, detect drowsy label, extract & scale features, cache everything."""
    if not MAT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {MAT_FILE}\n"
            "Please download the Figshare file and place it in the `data/` folder."
        )
    print("[INFO] Loading .mat file …")
    mat = _load_mat()
    eeg = mat["EEGsample"]
    labels = mat["substate"]
    subjects = mat["subindex"]

    # Detect drowsy label (just for printing – not used later)
    _ = _detect_drowsy_label(labels, eeg)

    print("[INFO] Extracting band‑power features …")
    feats = _extract_features(eeg)  # (samples, chunks, 120)

    # Split to train subjects for scaler fitting
    from features import split_by_subject
    X_train, _, subj_train = split_by_subject(feats, labels, subjects, TRAIN_SUBJECTS)
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    joblib.dump(scaler, SCALER_PATH)

    print("[INFO] Scaling all features …")
    feats_scaled = scaler.transform(feats.reshape(-1, feats.shape[-1])).reshape(feats.shape)

    # Cache to disk
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_PATH, feats_scaled)
    np.save(LABELS_PATH, labels)
    np.save(SUBJECTS_PATH, subjects)
    print("[INFO] Caching complete:")
    print(f"  features : {FEATURES_PATH} ({feats_scaled.nbytes/1e6:.1f} MB)")
    print(f"  labels   : {LABELS_PATH}")
    print(f"  subjects : {SUBJECTS_PATH}")
    print(f"  scaler   : {SCALER_PATH}")

if __name__ == "__main__":
    prepare_and_cache()
