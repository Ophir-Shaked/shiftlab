# -*- coding: utf-8 -*-

from __future__ import annotations
# Enables postponed evaluation of type annotations
# Helps avoid runtime issues with forward references in type hints

from typing import List
# Type hint: lists of column names

from sklearn.compose import ColumnTransformer
# Allows applying different preprocessing pipelines to different column subsets

from sklearn.pipeline import Pipeline
# Chains multiple preprocessing steps into a single reusable object

from sklearn.preprocessing import OneHotEncoder, StandardScaler
# - OneHotEncoder: converts categorical variables to binary indicators
# - StandardScaler: standardizes numeric features (mean=0, std=1)

from sklearn.impute import SimpleImputer
# Handles missing values by replacing them with a chosen statistic


def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """
    Build a preprocessing pipeline for tabular data with
    categorical and numeric features.

    The returned ColumnTransformer:
    - Applies a categorical pipeline to categorical columns
    - Applies a numeric pipeline to numeric columns
    - Drops any columns not explicitly listed

    Parameters:
    - cat_cols : list of categorical column names
    - num_cols : list of numeric column names

    Returns:
    - A fitted-ready ColumnTransformer compatible with scikit-learn
    """

    # --------------------------------------------------------
    # Categorical feature pipeline
    # --------------------------------------------------------
    cat_pipe = Pipeline(steps=[
        # Step 1: Fill missing categorical values
        # Uses the most frequent category per column
        ("imputer", SimpleImputer(strategy="most_frequent")),

        # Step 2: One-hot encode categories
        # - handle_unknown="ignore" prevents errors on unseen categories at test time
        # - sparse_output=False returns a dense NumPy array
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # --------------------------------------------------------
    # Numeric feature pipeline
    # --------------------------------------------------------
    num_pipe = Pipeline(steps=[
        # Step 1: Fill missing numeric values
        # Uses the median, which is robust to outliers
        ("imputer", SimpleImputer(strategy="median")),

        # Step 2: Standardize numeric features
        # - with_mean=True  -> center to mean 0
        # - with_std=True   -> scale to unit variance
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    # --------------------------------------------------------
    # Combine pipelines into a ColumnTransformer
    # --------------------------------------------------------
    return ColumnTransformer(
        transformers=[
            # Apply categorical pipeline to categorical columns
            ("cat", cat_pipe, cat_cols),

            # Apply numeric pipeline to numeric columns
            ("num", num_pipe, num_cols),
        ],

        # Drop any columns not explicitly listed above
        remainder="drop"
    )
