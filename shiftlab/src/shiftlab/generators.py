# -*- coding: utf-8 -*-
# Declares UTF-8 encoding (safe default for text and comments)

from __future__ import annotations
# Allows postponed evaluation of type annotations
# Prevents runtime issues with forward references in type hints

from typing import Optional, Dict, List, Tuple
# Type hints used throughout the generators

import numpy as np
# Numerical operations (arrays, statistics, random sampling)

import pandas as pd
# DataFrame handling (tabular data)

from scipy.stats import norm
# Used for Gaussian CDF / inverse CDF in copula-based generation

from sklearn.utils import check_random_state
# Ensures consistent, reproducible random number generation


# ============================================================
# Bootstrap + Noise Generator (label-preserving)
# ============================================================

class BootstrapNoiseLabeledGenerator:
    """
    Generates synthetic samples by:
    - Bootstrapping rows from the real training data (with replacement)
    - Adding Gaussian noise to numeric columns
    - Preserving labels and categorical features exactly
    """

    def __init__(
        self,
        feature_cols: List[str],
        num_cols: List[str],
        noise_std: float = 0.03
    ):
        # List of all feature column names
        self.feature_cols = feature_cols

        # Subset of feature columns that are numeric
        self.num_cols = num_cols

        # Noise scale factor (relative to each column's std)
        self.noise_std = float(noise_std)

        # Will store the training DataFrame after fitting
        self.df_train: Optional[pd.DataFrame] = None

        # Will store per-column standard deviations for numeric features
        self.num_stds: Optional[Dict[str, float]] = None


    def fit(self, df_train: pd.DataFrame) -> "BootstrapNoiseLabeledGenerator":
        """
        Fit the generator on real training data.

        Computes the empirical standard deviation of each numeric column,
        which is later used to scale the injected Gaussian noise.
        """

        # Store a copy of the training data to avoid mutating the original
        self.df_train = df_train.copy()

        # Dictionary mapping numeric column -> standard deviation
        stds: Dict[str, float] = {}

        for c in self.num_cols:
            # Convert column to numeric, coercing errors to NaN
            x = pd.to_numeric(
                self.df_train[c],
                errors="coerce"
            ).astype(float).values

            # Compute standard deviation, ignoring NaNs
            s = np.nanstd(x)

            # Use std if valid, otherwise fall back to 1.0
            stds[c] = float(s) if np.isfinite(s) and s > 0 else 1.0

        # Save computed stds for later sampling
        self.num_stds = stds

        # Return self to allow chaining (fit().sample())
        return self


    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """
        Generate n synthetic samples.

        Steps:
        1) Bootstrap rows from the training data
        2) Add Gaussian noise to numeric columns
        """

        # Ensure fit() was called before sampling
        assert self.df_train is not None and self.num_stds is not None

        # Create a reproducible random generator
        rng = check_random_state(seed)

        # Sample rows with replacement from training data
        out = (
            self.df_train
            .sample(n=n, replace=True, random_state=seed)
            .reset_index(drop=True)
        )

        # Add Gaussian noise to numeric columns
        for c in self.num_cols:
            # Convert column to numeric values
            x = pd.to_numeric(out[c], errors="coerce").astype(float).values

            # Draw Gaussian noise scaled by column std
            noise = rng.normal(
                loc=0.0,
                scale=self.noise_std * self.num_stds[c],
                size=n
            )

            # Add noise to the column
            out[c] = x + noise

        return out


# ============================================================
# Gaussian Copula Generator (unlabeled)
# ============================================================

class GaussianCopulaGenerator:
    """
    Generates synthetic tabular data using a Gaussian copula:
    - Categorical features are sampled from empirical distributions
    - Numeric features are modeled via a Gaussian copula
    """

    def __init__(self, cat_cols: List[str], num_cols: List[str]):
        # Names of categorical columns
        self.cat_cols = cat_cols

        # Names of numeric columns
        self.num_cols = num_cols

        # Empirical sorted values for each numeric column
        self.emp_num: Dict[str, np.ndarray] = {}

        # Mean vector of the latent Gaussian
        self.mu: Optional[np.ndarray] = None

        # Covariance matrix of the latent Gaussian
        self.cov: Optional[np.ndarray] = None

        # Empirical categorical distributions:
        # column -> (categories, probabilities)
        self.cat_dist: Dict[str, Tuple[List[object], np.ndarray]] = {}

        # Numeric columns that are valid for copula modeling
        self.num_valid_cols: List[str] = []


    def fit(self, df_train: pd.DataFrame) -> "GaussianCopulaGenerator":
        """
        Fit empirical distributions and estimate the Gaussian copula
        from real training data.
        """

        # --------------------------------------------------------
        # Fit categorical distributions
        # --------------------------------------------------------

        for c in self.cat_cols:
            vc = (
                df_train[c]
                .astype("object")
                .fillna("__MISSING__")
                .value_counts(normalize=True)
            )
            self.cat_dist[c] = (
                vc.index.tolist(),
                vc.values.astype(float)
            )

        # --------------------------------------------------------
        # Fit numeric distributions
        # --------------------------------------------------------

        valid: List[str] = []

        for c in self.num_cols:
            x = pd.to_numeric(df_train[c], errors="coerce").astype(float).values
            x = x[np.isfinite(x)]

            # Skip columns with too little data
            if len(x) < 50:
                continue

            self.emp_num[c] = np.sort(x)
            valid.append(c)

        self.num_valid_cols = valid

        # If no numeric columns are usable, fall back to trivial Gaussian
        if not valid:
            self.mu = np.zeros((1,), dtype=float)
            self.cov = np.eye(1, dtype=float)
            return self

        # --------------------------------------------------------
        # Convert numeric data to Gaussian latent space
        # --------------------------------------------------------

        Z = []

        for c in valid:
            xs = self.emp_num[c]

            x_full = pd.to_numeric(df_train[c], errors="coerce").astype(float).values
            fill = float(np.nanmedian(xs))

            # Replace NaNs with median
            x_full = np.where(np.isfinite(x_full), x_full, fill)

            # Rank-based empirical CDF
            ranks = np.searchsorted(xs, x_full, side="left")
            u = (ranks + 1.0) / (len(xs) + 2.0)

            # Map uniform -> Gaussian
            Z.append(norm.ppf(u))

        # Stack latent variables into matrix
        Z = np.stack(Z, axis=1)

        # Estimate Gaussian parameters
        self.mu = np.mean(Z, axis=0)
        self.cov = np.cov(Z, rowvar=False) + 1e-6 * np.eye(Z.shape[1])

        return self


    def _inv_emp(self, c: str, u: np.ndarray) -> np.ndarray:
        """
        Inverse empirical CDF for numeric column c.
        Maps uniform samples back to data space.
        """

        xs = self.emp_num[c]

        # Avoid extreme quantiles
        q = np.clip(u, 1e-6, 1.0 - 1e-6)

        # Convert quantiles to indices
        idx = (q * (len(xs) - 1)).astype(int)

        return xs[idx]


    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """
        Generate n synthetic samples (without labels).
        """

        rng = check_random_state(seed)
        out = pd.DataFrame()

        # Sample categorical columns
        for c in self.cat_cols:
            cats, probs = self.cat_dist[c]
            samp = rng.choice(cats, size=n, replace=True, p=probs).astype(object)
            samp[samp == "__MISSING__"] = None
            out[c] = samp

        # Sample numeric columns
        if self.num_valid_cols:
            assert self.mu is not None and self.cov is not None

            Zs = rng.multivariate_normal(self.mu, self.cov, size=n)
            U = norm.cdf(Zs)

            for j, c in enumerate(self.num_valid_cols):
                out[c] = self._inv_emp(c, U[:, j])
        else:
            # If no numeric columns, fill with NaN
            for c in self.num_cols:
                out[c] = np.nan

        return out


# ============================================================
# Conditional Gaussian Copula Generator (label-aware)
# ============================================================

class ConditionalGaussianCopulaLabeledGenerator:
    """
    Generates labeled synthetic data by:
    - Training one copula per class (y=0 and y=1)
    - Sampling class-conditional features
    - Assigning labels explicitly
    """

    def __init__(self, cat_cols: List[str], num_cols: List[str]):
        self.cat_cols = cat_cols
        self.num_cols = num_cols

        # Separate copula models for each class
        self.g0 = GaussianCopulaGenerator(cat_cols, num_cols)
        self.g1 = GaussianCopulaGenerator(cat_cols, num_cols)

        # Empirical positive class probability
        self.p1: float = 0.5


    def fit(
        self,
        df_train: pd.DataFrame,
        y_col: str = "__target__"
    ) -> "ConditionalGaussianCopulaLabeledGenerator":
        """
        Fit separate generators for each class label.
        """

        # Split data by class
        df0 = df_train[df_train[y_col] == 0].copy()
        df1 = df_train[df_train[y_col] == 1].copy()

        # Estimate class prior
        self.p1 = float(df_train[y_col].mean())

        # Fit copulas, falling back to full data if class too small
        self.g0.fit(
            df0.drop(columns=[y_col], errors="ignore")
            if len(df0) > 50 else
            df_train.drop(columns=[y_col], errors="ignore")
        )

        self.g1.fit(
            df1.drop(columns=[y_col], errors="ignore")
            if len(df1) > 50 else
            df_train.drop(columns=[y_col], errors="ignore")
        )

        return self


    def sample(
        self,
        n: int,
        seed: int,
        p1: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate n labeled synthetic samples.
        """

        # Use provided class prior or fallback to empirical one
        p = self.p1 if p1 is None else float(p1)

        # Determine number of samples per class
        n1 = int(round(n * p))
        n0 = n - n1

        # Sample each class independently
        df0 = self.g0.sample(n0, seed + 11)
        df0["__target__"] = 0

        df1 = self.g1.sample(n1, seed + 17)
        df1["__target__"] = 1

        # Combine and shuffle
        out = pd.concat([df0, df1], ignore_index=True)
        out = (
            out
            .sample(frac=1.0, random_state=seed + 23)
            .reset_index(drop=True)
        )

        return out
