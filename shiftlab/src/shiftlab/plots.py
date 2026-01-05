# -*- coding: utf-8 -*-

from __future__ import annotations
# Enables postponed evaluation of type annotations
# Helps avoid runtime issues with forward references

from typing import Optional, List, Tuple
# Type hints for optional outputs and metric definitions

import numpy as np
# Numerical operations (checking NaNs, array handling)

import matplotlib.pyplot as plt
# Plotting library used to generate and save figures


# ------------------------------------------------------------
# Metrics to plot:
# - Each tuple contains:
#   (column_name_in_dataframe, label_shown_on_plot)
# ------------------------------------------------------------
PLOT_METRICS: List[Tuple[str, str]] = [
    ("roc_auc", "ROC-AUC"),            # Area under the ROC curve
    ("accuracy", "Accuracy"),          # Standard classification accuracy
    ("ece", "ECE"),                    # Expected Calibration Error
    ("worst_group_acc", "Worst-group acc"),  # Worst-group accuracy across bins
]


def plot_selected_curves(
    df_agg,
    dataset: str,
    split: str,
    generator: str,
    model: str,
    out_path_png: str,
    show_plots: bool
):
    """
    Plot performance curves for a specific (dataset, split, generator, model)
    configuration.

    The x-axis is the synthetic data ratio.
    Each subplot corresponds to one metric defined in PLOT_METRICS.
    """

    # --------------------------------------------------------
    # Filter aggregated results to the requested configuration
    # --------------------------------------------------------
    sub = df_agg[
        (df_agg["dataset"] == dataset) &
        (df_agg["split"] == split) &
        (df_agg["generator"] == generator) &
        (df_agg["model"] == model)
    ].copy()

    # If no data matches the filter, exit silently
    if sub.empty:
        return

    # Sort by synthetic ratio to ensure a monotonic x-axis
    sub = sub.sort_values("syn_ratio").reset_index(drop=True)

    # Create a wide figure with one subplot per metric
    plt.figure(figsize=(4.2 * len(PLOT_METRICS), 4.2))

    # --------------------------------------------------------
    # Plot each metric in its own subplot
    # --------------------------------------------------------
    for i, (m, lab) in enumerate(PLOT_METRICS, start=1):
        # Select subplot position (1 row, N columns)
        plt.subplot(1, len(PLOT_METRICS), i)

        # X-axis: fraction of synthetic data
        x = sub["syn_ratio"].values

        # Y-axis: metric values
        y = sub[m].values

        # Optional error bars (standard deviation across seeds)
        yerr = sub[m + "_std"].values if (m + "_std") in sub.columns else None

        # Plot with error bars if available, otherwise simple line plot
        if yerr is not None and np.any(np.isfinite(yerr)):
            plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3)
        else:
            plt.plot(x, y, marker="o")

        # Axis labels
        plt.xlabel("Synthetic ratio")
        plt.ylabel(lab)

        # Light grid for readability
        plt.grid(True, alpha=0.25)

        # Subplot title shows the raw metric key
        plt.title(m)

    # --------------------------------------------------------
    # Figure-level title summarizing the configuration
    # --------------------------------------------------------
    plt.suptitle(
        f"{dataset} | {split} | {generator} | chosen={model}",
        y=1.05
    )

    # Adjust spacing to avoid overlaps
    plt.tight_layout()

    # Save figure to disk
    plt.savefig(out_path_png, dpi=160, bbox_inches="tight")

    # Optionally display the figure interactively
    if show_plots:
        plt.show()

    # Always close the figure to free memory
    plt.close()


def plot_confusion_matrix(
    cm2x2: np.ndarray,
    title: str,
    out_png: Optional[str],
    show_plots: bool
):
    """
    Plot a 2x2 confusion matrix.

    Parameters:
    - cm2x2     : 2x2 NumPy array [[TN, FP], [FN, TP]]
    - title     : title displayed above the plot
    - out_png   : path to save the image (if None, do not save)
    - show_plots: whether to display the plot interactively
    """

    # Create a square figure
    plt.figure(figsize=(4, 4))

    # Render the confusion matrix as an image
    plt.imshow(cm2x2)

    # Label x-axis (predicted labels)
    plt.xticks([0, 1], ["pred 0", "pred 1"])

    # Label y-axis (true labels)
    plt.yticks([0, 1], ["true 0", "true 1"])

    # Annotate each cell with its integer value
    for (i, j), val in np.ndenumerate(cm2x2):
        plt.text(j, i, int(val), ha="center", va="center")

    # Set the plot title
    plt.title(title)

    # Improve layout spacing
    plt.tight_layout()

    # Save the figure if an output path was provided
    if out_png is not None:
        plt.savefig(out_png, dpi=160, bbox_inches="tight")

    # Optionally display the plot
    if show_plots:
        plt.show()

    # Close the figure to prevent memory leaks
    plt.close()
