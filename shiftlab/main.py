# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Main entry point for the ShiftLab experiment framework.

This file exists mainly for convenience:
- Allows running the project easily from PyCharm (Run )
- Provides a simple CLI wrapper around shiftlab.runner.run_experiment

Recommended production usage is:
    python -m shiftlab.runner
But this file is perfectly valid and safe to keep.
"""

import argparse
import sys
from pathlib import Path


# ------------------------------------------------------------
# PyCharm / direct-run compatibility
# ------------------------------------------------------------
# When running this file directly (python main.py),
# Python does NOT automatically add "src/" to sys.path.
#
# This block makes sure that:
#   src/shiftlab/ is importable
# without requiring PYTHONPATH hacks.
#
# When running as a module (python -m shiftlab.runner),
# this block is harmless.
# ------------------------------------------------------------
if __name__ == "__main__":
    src_path = Path(__file__).parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))


# ------------------------------------------------------------
# Project imports (after sys.path fix)
# ------------------------------------------------------------
from shiftlab.config import load_config
from shiftlab.runner import run_experiment


def main() -> None:
    """
    Parse command-line arguments and run the ShiftLab experiment.

    This function:
    1) Loads the experiment configuration (YAML / JSON)
    2) Passes it to the main experiment runner
    """

    parser = argparse.ArgumentParser(
        description="ShiftLab – Synthetic Data under Distribution Shift (experiment runner)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML/JSON experiment config file",
    )

    args = parser.parse_args()

    # Load experiment configuration
    cfg = load_config(args.config)

    # Run the full experiment pipeline
    run_experiment(cfg)


# ------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
