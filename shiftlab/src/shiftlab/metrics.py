# -*- coding: utf-8 -*-
# Declares UTF-8 encoding (safe default for text + comments)

from __future__ import annotations
# Postpones evaluation of type hints (helps with forward refs + runtime stability)

from typing import Dict
# We return metrics as a dictionary mapping metric-name -> float

import numpy as np
# Numerical operations, arrays, vectorized masking

import pandas as pd
# Used for DataFrame-based operations (worst-group split)


from sklearn.metrics import (
    # Basic classification metrics
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,

    # Ranking / probability metrics
    roc_auc_score, average_precision_score,

    # Proper scoring rules / probabilistic losses
    brier_score_loss, log_loss,

    # Confusion matrix and agreement metrics
    confusion_matrix, matthews_corrcoef, cohen_kappa_score
)


def ece_score(y_true, p_pos, n_bins: int = 15) -> float:
    """
    ECE = Expected Calibration Error.

    Intuition:
    - Bin predictions by confidence (probability)
    - For each bin:
        * accuracy of the bin = mean(y_true)
        * confidence of the bin = mean(p_pos)
      add weighted |accuracy - confidence| over bins.

    Lower is better (perfect calibration -> 0.0).
    """

    # Convert inputs to clean numpy arrays with known dtypes
    y_true = np.asarray(y_true).astype(int)      # labels must be 0/1 ints
    p_pos = np.asarray(p_pos).astype(float)      # probabilities in [0,1]

    # Create bin edges between 0 and 1 inclusive
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    # Accumulate ECE across bins
    ece = 0.0

    # Iterate each probability bin [lo, hi)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]  # bin boundaries

        # Build a mask selecting samples that fall inside this bin
        # Last bin is inclusive on the right to include p_pos == 1.0
        mask = (p_pos >= lo) & (p_pos < hi) if i < n_bins - 1 else (p_pos >= lo) & (p_pos <= hi)

        # If no samples in this bin, skip it
        if mask.sum() == 0:
            continue

        # Bin accuracy = fraction of positives among samples in the bin
        acc = float(y_true[mask].mean())

        # Bin confidence = average predicted probability in the bin
        conf = float(p_pos[mask].mean())

        # Weight the bin gap by how frequent the bin is
        # mask.mean() = (#samples in bin) / (total samples)
        ece += float(mask.mean()) * abs(acc - conf)

    # Return as float for JSON/CSV friendliness
    return float(ece)


def pick_threshold_f1(y_true, p_pos) -> float:
    """
    Choose a decision threshold that maximizes F1 score.

    Strategy:
    - Consider candidate thresholds from the unique predicted probabilities.
    - If too many, subsample 200 quantiles to keep it fast.
    - Evaluate F1 for each threshold and return the best threshold.
    """

    # Ensure clean numpy arrays with correct dtypes
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p_pos).astype(float)

    # Candidate thresholds = unique probability values (clipped away from 0/1)
    # Clipping avoids numerical issues with log loss later
    thr_list = np.unique(np.clip(p, 1e-6, 1.0 - 1e-6))

    # If too many thresholds, downsample to 200 quantile points
    if len(thr_list) > 200:
        thr_list = np.quantile(thr_list, np.linspace(0.02, 0.98, 200))

    # Track the best threshold and best score found so far
    best_thr, best = 0.5, -1e9

    # Try each candidate threshold
    for thr in thr_list:
        # Convert probabilities to hard predictions
        y_pred = (p >= thr).astype(int)

        # Compute F1 (zero_division=0 prevents warnings if no positive preds)
        sc = f1_score(y_true, y_pred, zero_division=0)

        # Keep the best threshold
        if sc > best:
            best = sc
            best_thr = float(thr)

    return float(best_thr)


def worst_group_acc(
    df_test_raw: pd.DataFrame,
    y_true,
    y_pred,
    group_col: str,
    wga_bins: int,
    wga_min_group: int
) -> float:
    """
    Worst-Group Accuracy (WGA) for a numeric grouping feature.

    Steps:
    - Take a numeric column group_col from the raw test dataframe
    - Split it into quantile bins (pd.qcut)
    - Compute accuracy per bin
    - Return the minimum accuracy among bins with enough samples

    Returns NaN if grouping is impossible / invalid.
    """

    # If the grouping column doesn't exist, we cannot compute WGA
    if group_col not in df_test_raw.columns:
        return float("nan")

    # Convert grouping column to numeric; non-numeric values become NaN
    g = pd.to_numeric(df_test_raw[group_col], errors="coerce")

    # Need at least 2 distinct values to form groups
    if g.nunique(dropna=True) < 2:
        return float("nan")

    # Quantile-based binning:
    # - q = number of bins (limited by unique values)
    # - duplicates="drop" avoids errors when data has repeated quantiles
    try:
        bins = pd.qcut(g, q=min(wga_bins, g.nunique(dropna=True)), duplicates="drop")
    except Exception:
        return float("nan")

    # Track the worst (minimum) accuracy across sufficiently large bins
    worst = None

    # Iterate over unique bins (excluding NaNs)
    for b in bins.dropna().unique():
        # Boolean mask of samples belonging to this bin
        idx = (bins == b).to_numpy()

        # Skip bins that are too small (unstable estimate)
        if idx.sum() < wga_min_group:
            continue

        # Accuracy inside this bin
        acc = float(np.mean((y_pred[idx] == y_true[idx]).astype(float)))

        # Update the worst accuracy
        worst = acc if worst is None else min(worst, acc)

    # If we never had a valid bin, return NaN
    return float(worst) if worst is not None else float("nan")


def compute_metrics(y_true, p_pos, thr: float) -> Dict[str, float]:
    """
    Compute a suite of metrics from:
    - y_true: true binary labels (0/1)
    - p_pos: predicted probability for class 1
    - thr: decision threshold for converting p_pos into hard predictions

    Returns a dict suitable for CSV logging.
    """

    # Normalize input types
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos).astype(float)

    # Convert probabilities into predicted labels using the threshold
    y_pred = (p_pos >= thr).astype(int)

    # Confusion matrix gives counts of (TN, FP, FN, TP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # ROC-AUC and PR-AUC are only defined if both classes exist in y_true
    roc = roc_auc_score(y_true, p_pos) if len(np.unique(y_true)) > 1 else float("nan")
    pr = average_precision_score(y_true, p_pos) if len(np.unique(y_true)) > 1 else float("nan")

    # Clip probabilities to avoid log_loss hitting log(0)
    p_clip = np.clip(p_pos, 1e-6, 1.0 - 1e-6)

    # Return a flat dictionary of metrics (easy for DataFrame/CSV)
    return dict(
        threshold=float(thr),  # threshold used to produce y_pred

        # Standard classification metrics
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),

        # Ranking-based metrics
        roc_auc=float(roc),
        pr_auc=float(pr),

        # Probabilistic losses
        log_loss=float(log_loss(y_true, p_clip)),
        brier=float(brier_score_loss(y_true, p_pos)),

        # Agreement / correlation metrics
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        kappa=float(cohen_kappa_score(y_true, y_pred)),

        # Confusion-matrix counts (useful for analysis)
        tn=float(tn), fp=float(fp), fn=float(fn), tp=float(tp),

        # Predicted positive rate (how many positives the model outputs)
        pred_pos_rate=float(np.mean(y_pred)),
    )
