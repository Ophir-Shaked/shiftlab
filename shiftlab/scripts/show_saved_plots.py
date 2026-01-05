# -*- coding: utf-8 -*-
"""
SHOW SAVED PLOTS (PyCharm / terminal friendly)

Purpose:
- Locate all saved plot images under results/selected_plots
- Print how many plots were found and list their filenames
- Display each plot in a matplotlib window (one at a time)

Usage:
1) Run the experiment first so plots are generated:
   python main.py --config configs/default.yaml

2) Run this script:
   python show_saved_plots.py

Dependencies:
- Built-in: os, pathlib
- External: matplotlib  (pip install matplotlib)
"""

import os                    # Used for listing directory contents
from pathlib import Path     # Safer, cross-platform path handling

import matplotlib.image as mpimg   # Used to load image files (PNG/JPG)
import matplotlib.pyplot as plt    # Used to display images in windows

# Path to the directory where plots are saved
PLOTS_DIR = Path("results") / "selected_plots"


def main() -> None:
    """
    Main entry point for the script.
    Handles validation, printing, and displaying plots.
    """

    # Check that the plots directory exists
    if not PLOTS_DIR.exists():
        raise FileNotFoundError(
            f"Plots directory not found: {PLOTS_DIR.resolve()}"
        )

    # Collect all image files in the directory
    plot_files = sorted([
        f for f in os.listdir(PLOTS_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    # Print summary
    print(f"\nFound {len(plot_files)} plots in '{PLOTS_DIR}':\n")

    if not plot_files:
        print("No plots found.")
        return

    # Print filenames
    for fname in plot_files:
        print(f"  {fname}")

    # Display plots one by one
    for fname in plot_files:
        full_path = PLOTS_DIR / fname
        img = mpimg.imread(full_path)

        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.title(fname)
        plt.show()


if __name__ == "__main__":
    main()
