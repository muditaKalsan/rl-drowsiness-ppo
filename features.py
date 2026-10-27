"""
Helper functions to load cached feature arrays and provide subject‑wise splits.
"""

import numpy as np
from pathlib import Path
from config import (
    FEATURES_PATH,
    LABELS_PATH,
    SUBJECTS_PATH,
    TRAIN_SUBJECTS,
    VAL_SUBJECT,
    TEST_SUBJECTS,
)

def load_all():
    """Load the cached feature, label, and subject arrays.
    Raises a clear error if the cache is missing.
    """
    try:
        X = np.load(FEATURES_PATH)
        y = np.load(LABELS_PATH)
        subjects = np.load(SUBJECTS_PATH)
        return X, y, subjects
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Cached feature files not found. Run `python data_utils.py` first."
        ) from exc

def split_by_subject(X, y, subjects, subject_ids):
    """Return samples belonging to the specified subject IDs (list)."""
    mask = np.isin(subjects, subject_ids)
    return X[mask], y[mask], subjects[mask]

def get_splits():
    """Return a dict with train/val/test splits (features already scaled)."""
    X, y, subj = load_all()
    splits = {}
    splits["train"] = split_by_subject(X, y, subj, TRAIN_SUBJECTS)
    splits["val"]   = split_by_subject(X, y, subj, VAL_SUBJECT)
    splits["test"]  = split_by_subject(X, y, subj, TEST_SUBJECTS)
    return splits
