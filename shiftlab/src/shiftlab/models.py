# -*- coding: utf-8 -*-


from __future__ import annotations
# Enables postponed evaluation of type annotations
# Helps avoid runtime issues with forward references

from typing import Any
# Generic return type for models (sklearn / xgboost objects)

from sklearn.linear_model import LogisticRegression
# Linear baseline classifier

from sklearn.ensemble import HistGradientBoostingClassifier
# Tree-based gradient boosting model optimized for tabular data


def has_xgboost() -> bool:
    """
    Check whether xgboost is installed in the environment.

    Returns:
    - True  -> xgboost can be imported
    - False -> xgboost is not available

    This allows the pipeline to conditionally enable / disable XGB.
    """

    try:
        import xgboost 
        # Import succeeds -> xgboost is installed
        return True
    except Exception:
        # Any import failure means xgboost is unavailable
        return False


def make_model(name: str, seed: int) -> Any:
    """
    Factory function that creates and returns a model instance
    based on a string identifier.

    Parameters:
    - name : model identifier ("logreg", "hgb", "xgb")
    - seed : random seed for reproducibility (used where supported)

    Returns:
    - A model object implementing fit() / predict() / predict_proba()
    """

    # ------------------------------------------------------------
    # Logistic Regression (baseline linear model)
    # ------------------------------------------------------------
    if name == "logreg":
        return LogisticRegression(
            max_iter=2000,   # Allow enough iterations for convergence
            solver="lbfgs",  # Stable solver for dense numeric features
        )

    # ------------------------------------------------------------
    # Histogram Gradient Boosting (sklearn)
    # ------------------------------------------------------------
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.07,  # Step size shrinkage
            max_depth=7,         # Tree depth (controls complexity)
            max_iter=250,        # Number of boosting iterations
            random_state=seed,   # Ensures reproducible results
        )

    # ------------------------------------------------------------
    # XGBoost (optional dependency)
    # ------------------------------------------------------------
    if name == "xgb":
        try:
            import xgboost as xgb
            # Import inside the function to keep xgboost optional
        except Exception as e:
            # Fail fast with a clear error if xgboost is requested but missing
            raise RuntimeError(
                "xgboost is not installed. Install it or remove 'xgb' from config."
            ) from e

        return xgb.XGBClassifier(
            n_estimators=260,     # Number of trees
            max_depth=6,          # Tree depth
            learning_rate=0.06,   # Shrinkage rate
            subsample=0.9,        # Row subsampling per tree
            colsample_bytree=0.9,# Column subsampling per tree
            reg_lambda=1.0,       # L2 regularization
            random_state=seed,    # Reproducibility
            n_jobs=-1,            # Use all available CPU cores
            eval_metric="logloss",# Proper probabilistic loss
            tree_method="hist",   # Fast histogram-based training
        )

    # ------------------------------------------------------------
    # Unknown model name
    # ------------------------------------------------------------
    raise ValueError(f"Unknown model: {name}")
