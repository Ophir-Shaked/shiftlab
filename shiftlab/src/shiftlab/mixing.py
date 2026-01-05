# -*- coding: utf-8 -*-
# Declares UTF-8 encoding (safe default)

from __future__ import annotations
# Enables postponed evaluation of type annotations

import numpy as np
# Numerical operations: arrays, stacking, permutation

from sklearn.utils import check_random_state
# Provides a reproducible random number generator from a seed


def _sample_indices(n: int, k: int, rng) -> np.ndarray:
    """
    Sample k indices uniformly at random from range [0, n),
    with replacement.

    Parameters:
    - n   : size of the source array
    - k   : number of indices to sample
    - rng : random number generator (from check_random_state)

    Returns:
    - NumPy array of indices (length k)
    """

    # If no samples are requested, return an empty integer array
    if k <= 0:
        return np.zeros((0,), dtype=int)

    # Sample k integers uniformly from [0, n)
    return rng.randint(0, n, size=k)


def build_mixed_matrix(
    X_real,
    y_real,
    X_syn,
    y_syn,
    n_total: int,
    syn_ratio: float,
    seed: int
):
    """
    Build a mixed dataset containing both real and synthetic samples.

    Steps:
    1) Decide how many real vs. synthetic samples to use
    2) Sample with replacement from each source
    3) Concatenate real and synthetic data
    4) Shuffle the result

    Parameters:
    - X_real : real feature matrix
    - y_real : real labels
    - X_syn  : synthetic feature matrix
    - y_syn  : synthetic labels
    - n_total: total number of samples in the mixed dataset
    - syn_ratio : fraction of synthetic samples (0.0 -> all real, 1.0 -> all synthetic)
    - seed   : random seed for reproducibility

    Returns:
    - X_mix : mixed and shuffled feature matrix
    - y_mix : mixed and shuffled label vector
    """

    # Initialize reproducible random generator
    rng = check_random_state(seed)

    # Number of synthetic samples to draw
    n_syn = int(round(n_total * syn_ratio))

    # Number of real samples is whatever remains
    n_real = n_total - n_syn

    # Sample indices (with replacement) from real and synthetic sets
    idx_r = _sample_indices(len(y_real), n_real, rng)
    idx_s = _sample_indices(len(y_syn), n_syn, rng)

    # Stack features:
    # - If n_syn > 0, combine real and synthetic
    # - Otherwise, use only real samples
    X_mix = (
        np.vstack([X_real[idx_r], X_syn[idx_s]])
        if n_syn > 0
        else X_real[idx_r]
    )

    # Concatenate labels in the same order as features
    y_mix = (
        np.concatenate([y_real[idx_r], y_syn[idx_s]])
        if n_syn > 0
        else y_real[idx_r]
    )

    # Randomly permute the mixed dataset to remove ordering bias
    perm = rng.permutation(len(y_mix))

    # Apply the same permutation to features and labels
    return X_mix[perm], y_mix[perm]
