# -*- coding: utf-8 -*-

from __future__ import annotations
# Enables postponed evaluation of type annotations

from dataclasses import dataclass
# NOTE: Imported but not used in this file (can be removed if you want, no behavior change)

from typing import List, Optional, Tuple, Dict, Any
# Type hints used for inputs/outputs

import random
# Python's built-in RNG (used for reproducibility)

import numpy as np
# NumPy RNG + numeric operations (quantiles, NaN handling)

import pandas as pd
# DataFrame operations

from sklearn.model_selection import train_test_split
# Standard train/test splitting with stratification


def set_global_seed(seed: int) -> None:
    """
    Set global random seeds to make the experiment reproducible.

    This affects:
    - Python's random module
    - NumPy's random module

    (Other libraries may need their own seed setting separately.)
    """
    random.seed(seed)      # Fix Python random RNG sequence
    np.random.seed(seed)   # Fix NumPy random RNG sequence


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to reduce dataset-specific naming quirks.

    Example changes:
    - strips leading/trailing spaces
    - replaces spaces with underscores

    This helps when matching columns across datasets.
    """
    df = df.copy()  # avoid mutating caller's DataFrame
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]  # normalize each column name
    return df


def find_col_case_insensitive(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find the first column in df that matches one of the candidates
    (case-insensitive).

    Returns:
    - the real column name as it appears in df (preserves original casing)
    - or None if no candidate matches
    """
    # Map lowercase column name -> original column name
    low_map = {str(c).lower(): c for c in df.columns}

    # Check candidate names in order (first match wins)
    for cand in candidates:
        k = str(cand).lower()  # normalize candidate to lowercase
        if k in low_map:
            return low_map[k]  # return the original df column name

    return None  # no matches found


def ensure_binary_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, str]:
    """
    Convert a target column to binary {0,1}.

    Behavior:
    1) If target has more than 2 classes:
       - keep only the top-2 most frequent classes
       - drop the rest (so the problem becomes binary)
    2) Choose which class is treated as the positive class:
       - prefer known 'positive' labels like ">50K", "1", "yes", etc.
       - otherwise choose the lexicographically largest label

    Returns:
    - df: a copy with target converted to int 0/1
    - pos: the original label string used as positive class (for printing/metadata)
    """
    df = df.copy()  # work on a copy to avoid side effects

    # Convert labels to strings to handle mixed dtypes consistently
    y = df[target_col].astype(str)

    # Count occurrences of each class label
    vc = y.value_counts()

    # If more than 2 classes, keep only the two most frequent classes
    if len(vc) > 2:
        keep = list(vc.index[:2])                    # top-2 labels by frequency
        df = df[y.isin(keep)].reset_index(drop=True) # filter rows to only those labels
        y = df[target_col].astype(str)               # re-extract y after filtering

    # Collect unique labels (sorted for stable ordering)
    unique = sorted(set(y))

    # Try to pick a "natural" positive class if present
    pos = None
    for cand in [">50K", "1", "yes", "true", "TRUE", "True", "2"]:
        if cand in unique:
            pos = cand
            break

    # If none of the preferred labels exist, pick the "largest" label as positive
    if pos is None:
        pos = unique[-1]

    # Convert target to binary int:
    # 1 if label == pos else 0
    df[target_col] = (y == pos).astype(int)

    return df, pos


def split_iid(df_in: pd.DataFrame, seed: int, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standard IID train/test split.

    Uses stratification on "__target__" to preserve class balance.
    Returns train and test DataFrames with clean indexes.
    """
    tr, te = train_test_split(
        df_in,
        test_size=test_size,
        stratify=df_in["__target__"],  # keep label distribution similar in train/test
        random_state=seed              # reproducible split
    )
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def split_shift_quantile(
    df_in: pd.DataFrame,
    seed: int,
    col: str,
    q: float,
    test_size_fallback: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a distribution shift split based on a numeric column.

    Logic:
    - Compute quantile threshold thr = quantile(col, q)
    - Train = rows where col <= thr
    - Test  = rows where col > thr

    Guardrails:
    - If the column is all NaN -> fallback to IID split
    - If either split is too small or becomes single-class -> fallback to IID split

    Returns:
    - (train_df, test_df)
    """
    # Convert the chosen shift column to numeric (invalid parsing becomes NaN)
    x = pd.to_numeric(df_in[col], errors="coerce")

    # If everything is NaN, we cannot create a shift split -> fallback IID
    if x.isna().all():
        return split_iid(df_in, seed, test_size_fallback)

    # Compute the quantile threshold (ignoring NaNs)
    thr = float(np.nanquantile(x, q))

    # Split by threshold
    tr = df_in[x <= thr]  # "low" values -> train
    te = df_in[x > thr]   # "high" values -> test

    # Guardrails:
    # - avoid tiny splits (unstable)
    # - avoid splits with only one class (metrics undefined / training unstable)
    if (
        len(tr) < 800 or len(te) < 800 or
        tr["__target__"].nunique() < 2 or te["__target__"].nunique() < 2
    ):
        return split_iid(df_in, seed, test_size_fallback)

    # Shuffle each split (helps avoid any ordering artifacts from filtering)
    tr = tr.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    te = te.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)

    return tr, te
