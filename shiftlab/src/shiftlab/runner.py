# -*- coding: utf-8 -*-

from __future__ import annotations
# Postpones evaluation of type annotations (helps typing + avoids runtime issues)

import os, math, time, warnings
# os -> filesystem paths/dirs
# math -> ceil, etc.
# time -> timing experiment runtime
# warnings -> suppress noisy sklearn/openml warnings

from typing import Dict, Any, List, Tuple
# Type hints: config dictionary, lists of seeds/models, etc.

warnings.filterwarnings("ignore")
# Suppress warnings to keep console output clean (optional)

import numpy as np
# Core numerical operations

import pandas as pd
# DataFrame manipulation

import openml
# Loading datasets from OpenML

from sklearn.model_selection import train_test_split
# Used for subsampling the dataset and splitting train->train/val

# -------------------- Internal project imports --------------------
from .utils import (
    set_global_seed,             # sets numpy/random seeds globally
    normalize_cols,              # normalizes column names for consistency
    ensure_binary_target,        # ensures target is binary and returns positive label
    find_col_case_insensitive,   # find a column using several name candidates
    split_iid,                   # standard IID train/test split
    split_shift_quantile         # quantile-based shifted split on a feature
)

from .preprocessing import build_preprocessor
# Builds ColumnTransformer for (categorical + numeric) preprocessing

from .generators import BootstrapNoiseLabeledGenerator, ConditionalGaussianCopulaLabeledGenerator
# Synthetic data generators (bootstrap+noise, class-conditional copula)

from .models import make_model, has_xgboost
# Factory to create model objects; has_xgboost checks optional dependency

from .mixing import build_mixed_matrix
# Mixes real+synthetic samples according to syn_ratio

from .metrics import ece_score, pick_threshold_f1, compute_metrics, worst_group_acc
# Metrics: calibration, threshold selection, standard metrics, worst-group accuracy

from .plots import plot_selected_curves, plot_confusion_matrix
# Plotting utilities: metric curves + confusion matrix PNG


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Convert raw decision scores to probabilities using sigmoid.
    Used when model does not implement predict_proba (e.g., some linear models).
    """
    return 1.0 / (1.0 + np.exp(-x))


def run_experiment(cfg: Dict[str, Any]) -> None:
    """
    Main entrypoint:
    - Load OpenML datasets
    - Create IID + shifted splits
    - Build preprocessing
    - Generate synthetic pools
    - Train models across seeds/ratios
    - Evaluate on test, aggregate across seeds
    - Select best configs, save plots & confusion matrices
    """

    # -------------------- Output directories --------------------
    out_dir = cfg.get("out_dir", "results")               # root output folder
    plots_dir = os.path.join(out_dir, "selected_plots")   # where we save selected figures
    os.makedirs(out_dir, exist_ok=True)                   # create results folder if missing
    os.makedirs(plots_dir, exist_ok=True)                 # create plots folder if missing

    # -------------------- Global reproducibility --------------------
    global_seed = int(cfg.get("global_seed", 1337))       # global seed default
    set_global_seed(global_seed)                          # sets RNG seeds across libraries

    # -------------------- Experiment grid --------------------
    seeds: List[int] = list(cfg.get("seeds", [1, 2]))                          # repeat runs
    ratios: List[float] = list(cfg.get("ratios", [0.0, 0.25, 0.5, 0.75, 1.0])) # syn mix ratios
    n_max_rows = int(cfg.get("n_max_rows", 20000))                             # cap dataset rows

    # -------------------- Splits configuration --------------------
    test_size_iid = float(cfg.get("test_size_iid", 0.2))        # IID test fraction
    val_size_in_train = float(cfg.get("val_size_in_train", 0.2))# val fraction inside train
    shift_q = float(cfg.get("shift_q", 0.6))                    # quantile used for shift split

    # -------------------- Synthetic pool configuration --------------------
    syn_pool_multiplier = float(cfg.get("syn_pool_multiplier", 1.5))  # pool size factor
    boot_noise_std = float(cfg.get("boot_noise_std", 0.03))           # bootstrap noise scale

    # -------------------- Worst-group accuracy configuration --------------------
    wga_bins = int(cfg.get("wga_bins", 10))          # number of quantile groups
    wga_min_group = int(cfg.get("wga_min_group", 25))# ignore small groups

    # -------------------- Model selection configuration --------------------
    select_score_alpha = float(cfg.get("select_score_alpha", 0.0))
    # score = roc_auc - alpha * ece (alpha=0 -> only roc_auc)

    # -------------------- Plotting / artifact configuration --------------------
    show_plots = bool(cfg.get("show_plots", False))                    # show GUI windows
    plot_representative_only = bool(cfg.get("plot_representative_only", True))  # plot only reps
    save_cm_png = bool(cfg.get("save_cm_png", True))                   # save CM figures or not

    # -------------------- Models & generators --------------------
    models: List[str] = list(cfg.get("models", ["logreg", "hgb", "xgb"]))
    generators: List[str] = list(cfg.get("generators", ["bootstrap_noise_labeled", "gaussian_copula_conditional"]))

    # If xgboost requested but not installed -> drop it gracefully (no hard-fail)
    if "xgb" in models and not has_xgboost():
        models = [m for m in models if m != "xgb"]

    # -------------------- Dataset list validation --------------------
    datasets: List[Dict[str, Any]] = list(cfg.get("datasets", []))
    if not datasets:
        raise ValueError("Config must include datasets list.")

    # Print run overview (useful for logs)
    print("OUT_DIR:", out_dir)
    print("SEEDS:", seeds)
    print("DATASETS:", [d.get("name") for d in datasets])
    print("GENERATORS:", generators)
    print("MODELS:", models)
    print("RATIOS:", ratios)

    # Start timer for runtime measurement
    t0 = time.time()

    # We'll collect one row per (dataset, seed, split, generator, ratio, model)
    rows: List[Dict[str, Any]] = []

    # ============================================================
    # Loop over datasets
    # ============================================================
    for ds_cfg in datasets:
        ds_name = ds_cfg["name"]  # dataset name used in output

        # Section header
        print("\n" + "=" * 90)
        print("DATASET:", ds_name)
        print("=" * 90)

        # -------------------- Load dataset from OpenML --------------------
        oml = ds_cfg["openml"]
        if oml["by"] == "name":
            ds = openml.datasets.get_dataset(oml["value"], download_all_files=False)
        else:
            ds = openml.datasets.get_dataset(int(oml["value"]), download_all_files=False)

        # X = features dataframe, y = target series
        X, y, *_ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)

        # Build a single DataFrame with a standard target column name
        df = X.copy()
        df["__target__"] = y

        # Normalize column names (e.g., spaces/uppercase) for consistent matching
        df = normalize_cols(df)

        # Ensure target is binary; also returns "pos_label" metadata for printing
        df, pos_label = ensure_binary_target(df, "__target__")

        # -------------------- Cap dataset size (optional) --------------------
        if n_max_rows and len(df) > n_max_rows:
            # Stratified subsample to maintain class balance
            df, _ = train_test_split(
                df,
                train_size=n_max_rows,
                stratify=df["__target__"],
                random_state=global_seed
            )
            df = df.reset_index(drop=True)

        # Feature list (everything except target)
        feature_cols = [c for c in df.columns if c != "__target__"]

        # Replace literal '?' with NaN in categorical columns (common in Adult dataset)
        for c in feature_cols:
            if df[c].dtype == "object" or str(df[c].dtype).startswith("category"):
                df[c] = df[c].replace("?", np.nan)

        # Identify categorical vs numeric columns
        cat_cols = [c for c in feature_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
        num_cols = [c for c in feature_cols if c not in cat_cols]

        # -------------------- Choose shift features --------------------
        shift_candidates = ds_cfg.get("shift_candidates", [[], []])

        # Try to find columns from candidate lists (case-insensitive)
        shift1 = find_col_case_insensitive(df, shift_candidates[0]) if shift_candidates else None
        shift2 = find_col_case_insensitive(df, shift_candidates[1]) if shift_candidates else None

        # Fallback selection if not found in candidates
        if shift1 is None:
            shift1 = num_cols[0] if len(num_cols) > 0 else feature_cols[0]
        if shift2 is None:
            shift2 = num_cols[1] if len(num_cols) > 1 else feature_cols[1]

        # Ensure shift columns are numeric (needed for quantiles / qcut)
        df[shift1] = pd.to_numeric(df[shift1], errors="coerce")
        df[shift2] = pd.to_numeric(df[shift2], errors="coerce")

        # Print dataset summary
        print(f"Rows: {len(df)} | Features: {len(feature_cols)} | Pos rate: {df['__target__'].mean():.4f} | Pos label: {pos_label}")
        print(f"Numeric cols: {len(num_cols)} | Categorical cols: {len(cat_cols)}")
        print("Shift cols:", shift1, ",", shift2)

        # -------------------- Define split functions --------------------
        splits = {
            # IID split
            "iid": lambda d, s: split_iid(d, s, test_size=test_size_iid),

            # Shift split using feature shift1
            f"{shift1}_shift": lambda d, s: split_shift_quantile(d, s, shift1, q=shift_q, test_size_fallback=test_size_iid),

            # Shift split using feature shift2
            f"{shift2}_shift": lambda d, s: split_shift_quantile(d, s, shift2, q=shift_q, test_size_fallback=test_size_iid),
        }

        # ============================================================
        # Loop over seeds and splits
        # ============================================================
        for seed in seeds:
            print(f"\n--- SEED {seed} ---")

            for split_name, split_fn in splits.items():
                # Build train/test for this split type
                df_train, df_test = split_fn(df, seed)

                # If train or test contains only one class -> skip (metrics undefined)
                if df_train["__target__"].nunique() < 2 or df_test["__target__"].nunique() < 2:
                    print("Skipping split (single-class):", split_name)
                    continue

                # Split train into train_sub and validation
                df_train_sub, df_val = train_test_split(
                    df_train,
                    test_size=val_size_in_train,
                    stratify=df_train["__target__"],
                    random_state=seed
                )

                # Reset indexes for clean alignment
                df_train_sub = df_train_sub.reset_index(drop=True)
                df_val = df_val.reset_index(drop=True)
                df_test = df_test.reset_index(drop=True)

                # Build preprocessing (fit on train_sub only -> avoids leakage)
                pre = build_preprocessor(cat_cols, num_cols)

                # Fit preprocessor on train_sub features and transform to numeric arrays
                X_real = pre.fit_transform(df_train_sub[feature_cols])
                y_real = df_train_sub["__target__"].astype(int).values

                # Transform validation and test using same preprocessor
                X_val = pre.transform(df_val[feature_cols])
                y_val = df_val["__target__"].astype(int).values

                X_test = pre.transform(df_test[feature_cols])
                y_test = df_test["__target__"].astype(int).values

                # Define group column for worst-group accuracy
                # - For IID: use shift1
                # - For shift split: the split feature name without "_shift"
                group_col = shift1 if split_name == "iid" else split_name.replace("_shift", "")

                # Synthetic pool size:
                # n_pool is larger than train to allow sampling at multiple ratios
                n_train = len(df_train_sub)
                n_pool = int(math.ceil(syn_pool_multiplier * n_train))

                # Pre-generate synthetic pools per generator (efficient)
                syn_pools = {}

                for gen_name in generators:
                    if gen_name == "bootstrap_noise_labeled":
                        # Bootstrap rows from train_sub and add numeric noise
                        gen = BootstrapNoiseLabeledGenerator(
                            feature_cols=feature_cols,
                            num_cols=num_cols,
                            noise_std=boot_noise_std
                        ).fit(df_train_sub)

                        # Sample labeled synthetic data
                        df_syn = gen.sample(n_pool, seed=seed + 100)

                        # Transform to model-ready matrix using same preprocessor
                        y_syn = df_syn["__target__"].astype(int).values
                        X_syn = pre.transform(df_syn[feature_cols])

                        syn_pools[gen_name] = (X_syn, y_syn)

                    elif gen_name == "gaussian_copula_conditional":
                        # Fit class-conditional copulas and sample with similar class prior
                        gen = ConditionalGaussianCopulaLabeledGenerator(cat_cols, num_cols).fit(
                            df_train_sub, y_col="__target__"
                        )
                        df_syn = gen.sample(
                            n_pool,
                            seed=seed + 200,
                            p1=float(df_train_sub["__target__"].mean())
                        )

                        y_syn = df_syn["__target__"].astype(int).values
                        X_syn = pre.transform(df_syn[feature_cols])

                        syn_pools[gen_name] = (X_syn, y_syn)

                    else:
                        raise ValueError(f"Unknown generator: {gen_name}")

                # ============================================================
                # Loop over generators, ratios, models
                # ============================================================
                for gen_name in generators:
                    X_syn, y_syn = syn_pools[gen_name]

                    for ratio in ratios:
                        # Mix real and synthetic into a single training set
                        X_mix, y_mix = build_mixed_matrix(
                            X_real, y_real, X_syn, y_syn,
                            n_total=len(y_real),
                            syn_ratio=float(ratio),

                            # Seed depends on seed + ratio + generator offset
                            # This keeps mixing reproducible but different per configuration
                            seed=seed + int(1000 * float(ratio)) + (0 if gen_name == "bootstrap_noise_labeled" else 7)
                        )

                        for model_name in models:
                            # Create model instance based on config name
                            model = make_model(model_name, seed)

                            # Train model on the mixed data
                            model.fit(X_mix, y_mix)

                            # Get validation and test probabilities
                            if hasattr(model, "predict_proba"):
                                p_val = model.predict_proba(X_val)[:, 1]
                                p_test = model.predict_proba(X_test)[:, 1]
                            else:
                                # Some models return raw scores, not probabilities
                                s_val = model.decision_function(X_val)
                                s_te = model.decision_function(X_test)
                                p_val = _sigmoid(s_val)
                                p_test = _sigmoid(s_te)

                            # Pick threshold on validation only (prevents leakage)
                            thr = pick_threshold_f1(y_val, p_val)

                            # Compute standard metrics on test using chosen threshold
                            met = compute_metrics(y_test, p_test, thr)

                            # Add calibration metric (ECE)
                            met["ece"] = ece_score(y_test, p_test)

                            # Compute worst-group accuracy based on group_col
                            y_pred = (p_test >= thr).astype(int)
                            met["worst_group_acc"] = worst_group_acc(
                                df_test, y_test, y_pred,
                                group_col, wga_bins, wga_min_group
                            )

                            # Append one experiment result row
                            rows.append({
                                "dataset": ds_name,
                                "seed": seed,
                                "split": split_name,
                                "generator": gen_name,
                                "syn_ratio": float(ratio),
                                "model": model_name,
                                "pos_label": pos_label,
                                "group_col": group_col,
                                "n_train": int(len(df_train_sub)),
                                "n_val": int(len(df_val)),
                                "n_test": int(len(df_test)),
                                "pos_rate_train": float(df_train_sub["__target__"].mean()),
                                "pos_rate_val": float(df_val["__target__"].mean()),
                                "pos_rate_test": float(df_test["__target__"].mean()),
                                **met
                            })

    # ============================================================
    # Save raw results
    # ============================================================
    df_all = pd.DataFrame(rows)
    raw_csv = os.path.join(out_dir, "results_raw.csv")
    df_all.to_csv(raw_csv, index=False)
    print("\nSaved:", raw_csv)
    print("Total time (s):", round(time.time() - t0, 2))

    # ============================================================
    # Aggregate mean/std across seeds
    # ============================================================
    agg_cols = ["dataset", "split", "generator", "syn_ratio", "model"]  # group keys
    drop_cols = ["seed", "pos_label", "group_col", "n_train", "n_val", "n_test", "pos_rate_train", "pos_rate_val", "pos_rate_test"]
    metric_cols = [c for c in df_all.columns if c not in agg_cols + drop_cols]  # numeric metrics columns

    # Compute mean metrics per config
    df_mean = df_all.groupby(agg_cols, as_index=False)[metric_cols].mean()

    # Compute std metrics per config (suffix with _std)
    df_std = df_all.groupby(agg_cols, as_index=False)[metric_cols].std().add_suffix("_std")

    # Merge mean + std into one table
    df_agg = df_mean.merge(df_std, left_on=agg_cols, right_on=[c + "_std" for c in agg_cols], how="left")

    # Drop duplicated grouping columns from the std-merge
    for c in agg_cols:
        if c + "_std" in df_agg.columns:
            df_agg.drop(columns=[c + "_std"], inplace=True)

    # Save aggregated CSV
    agg_csv = os.path.join(out_dir, "results_agg.csv")
    df_agg.to_csv(agg_csv, index=False)
    print("Saved:", agg_csv)

    # ============================================================
    # Select BEST model per (dataset, split, generator)
    # score = roc_auc - alpha * ece
    # ============================================================
    tmp = df_agg.copy()
    tmp["select_score"] = tmp["roc_auc"] - select_score_alpha * tmp["ece"]

    # Sort: higher score is better, lower ECE as tie-breaker
    tmp = tmp.sort_values(
        ["dataset", "split", "generator", "select_score", "ece"],
        ascending=[True, True, True, False, True]
    )

    # Take the first row per group (best)
    best = tmp.groupby(["dataset", "split", "generator"], as_index=False).first()

    # Save best table
    best_csv = os.path.join(plots_dir, "best_models_full_metrics.csv")
    best.to_csv(best_csv, index=False)
    print("Saved:", best_csv)

    # Print best configurations
    print("\n=== BEST CONFIG PER (dataset, split, generator) ===")
    print(best)

    # ============================================================
    # Select representative configurations for plotting
    # ============================================================
    best_plot = best.copy()

    if plot_representative_only:
        reps_rows = []

        for ds in best["dataset"].unique():
            # Best IID config for this dataset
            iid = best[(best["dataset"] == ds) & (best["split"] == "iid")].copy()
            iid = iid.sort_values(["select_score", "ece"], ascending=[False, True]).iloc[0]
            reps_rows.append(iid)

            # Worst shift split by ROC-AUC, then choose best config within it
            shifts = best[(best["dataset"] == ds) & (best["split"] != "iid")].copy()
            if not shifts.empty:
                worst_split = shifts.sort_values("roc_auc", ascending=True).iloc[0]["split"]
                worst = best[(best["dataset"] == ds) & (best["split"] == worst_split)].copy()
                worst = worst.sort_values(["select_score", "ece"], ascending=[False, True]).iloc[0]
                reps_rows.append(worst)

        # Use only representative subset
        best_plot = pd.DataFrame(reps_rows).reset_index(drop=True)

    print("\n=== PLOTTING THESE BEST CONFIGS ONLY ===")
    print(best_plot[["dataset", "split", "generator", "model", "syn_ratio", "roc_auc", "accuracy", "ece", "worst_group_acc", "select_score"]])

    # ============================================================
    # Generate "selected curves" plots (one PNG per representative config)
    # ============================================================
    for _, r in best_plot.iterrows():
        ds, sp, gn, md = r["dataset"], r["split"], r["generator"], r["model"]
        out_png = os.path.join(plots_dir, f"selected_{ds}__{sp}__{gn}__{md}.png".replace("/", "_"))
        plot_selected_curves(df_agg, ds, sp, gn, md, out_png, show_plots=show_plots)

    # ============================================================
    # Confusion matrices for representative configs (only for first seed)
    # ============================================================
    seed_for_cm = seeds[0]  # choose first seed to make CM reproducible and not duplicated

    for _, r in best_plot.iterrows():
        ds, sp, gn, md, rt = r["dataset"], r["split"], r["generator"], r["model"], r["syn_ratio"]

        # Find the exact row in raw results for this config and seed
        sub = df_all[
            (df_all["seed"] == seed_for_cm) &
            (df_all["dataset"] == ds) &
            (df_all["split"] == sp) &
            (df_all["generator"] == gn) &
            (df_all["model"] == md) &
            (df_all["syn_ratio"] == rt)
        ]

        # If not found, skip
        if sub.empty:
            continue

        # Take the first match (should be exactly one)
        row0 = sub.iloc[0]

        # Reconstruct confusion matrix from stored TN/FP/FN/TP
        cm = np.array([
            [row0["tn"], row0["fp"]],
            [row0["fn"], row0["tp"]],
        ], dtype=float)

        # Human-readable title
        cm_title = f"CM | {ds} | {sp} | {gn} | {md} | ratio={rt} (seed={seed_for_cm})"

        # Save path for confusion matrix PNG (optional)
        cm_png = None
        if save_cm_png:
            cm_png = os.path.join(
                plots_dir,
                f"cm_{ds}__{sp}__{gn}__{md}__r{rt}_seed{seed_for_cm}.png".replace("/", "_")
            )

        # Plot and optionally save/display
        plot_confusion_matrix(cm, cm_title, out_png=cm_png, show_plots=show_plots)

    # ============================================================
    # Final printed summary table
    # ============================================================
    print("\n" + "=" * 90)
    print("FINAL SELECTED MODELS SUMMARY (BEST per dataset/split/generator)")
    print("=" * 90)

    cols_to_show = ["dataset", "split", "generator", "model", "syn_ratio", "roc_auc", "accuracy", "f1", "ece", "worst_group_acc", "select_score"]
    summary_print = best[cols_to_show].copy()

    # Format numeric metrics to 4 decimals for pretty printing
    for c in ["roc_auc", "accuracy", "f1", "ece", "worst_group_acc", "select_score"]:
        summary_print[c] = summary_print[c].map(lambda x: f"{float(x):.4f}")

    print(summary_print.to_string(index=False))

    # Explain selection criterion used
    print("\nSelection criterion:")
    if select_score_alpha > 0:
        print(f"  score = ROC-AUC - {select_score_alpha} * ECE")
    else:
        print("  score = ROC-AUC")

    # Print artifact locations
    print("\nArtifacts:")
    print("  RAW  :", raw_csv)
    print("  AGG  :", agg_csv)
    print("  BEST :", best_csv)
    print("  PLOTS:", plots_dir)
    print("=" * 90)


if __name__ == "__main__":
    # Allow running the module directly:
    # python -m src.runner
    # This assumes configs/default.yaml is reachable from current working directory.
    from .config import load_config
    cfg = load_config("configs/default.yaml")
    run_experiment(cfg)
