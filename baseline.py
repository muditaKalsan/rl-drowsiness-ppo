"""
Baseline classifiers trained on the *average* of the first k chunks.
We evaluate k = 1, 2, 3 (i.e. 1 s, 2 s, 3 s windows).
Metrics printed and also saved for later plotting.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from config import RANDOM_SEED
from features import get_splits
import os
import pandas as pd
from pathlib import Path
from config import RESULTS_DIR

def _aggregate_first_k(X, k):
    """Average the first k chunks -> shape (samples, features)"""
    return X[:, :k, :].mean(axis=1)

def _train_and_evaluate(clf, X_train, y_train, X_test, y_test, name, k):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    cm = confusion_matrix(y_test, pred)
    print(f"\n=== {name} | k={k} chunks ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    return {"accuracy": acc, "f1": f1, "confusion": cm, "pred": pred}

def run_baselines():
    splits = get_splits()
    X_train, y_train = splits["train"][:2]
    X_val, y_val = splits["val"][:2]
    X_test, y_test = splits["test"][:2]

    results = {}
    for k in (1, 2, 3):
        X_tr_k = _aggregate_first_k(X_train, k)
        X_te_k = _aggregate_first_k(X_test,  k)

        svm = SVC(kernel="rbf", C=1.0, random_state=RANDOM_SEED)
        rf  = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)

        results[f"SVM_k{k}"] = _train_and_evaluate(svm, X_tr_k, y_train, X_te_k, y_test, "SVM", k)
        results[f"RF_k{k}"]  = _train_and_evaluate(rf,  X_tr_k, y_train, X_te_k, y_test, "RandomForest", k)

    # Save confusion matrices for plotting later
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for key, val in results.items():
        cm_path = RESULTS_DIR / f"confusion_{key}.npy"
        np.save(cm_path, val["confusion"])
    return results

if __name__ == "__main__":
    run_baselines()
