# -*- coding: utf-8 -*-
from __future__ import annotations
# Enables postponed evaluation of type annotations.
# This allows using forward references and modern type hints
# without runtime import issues (and keeps typing consistent
# across Python versions).

"""
Main entry point for the ShiftLab experiment framework.

This file exists mainly for convenience:
- Allows running the project easily from PyCharm (Run button)
- Provides a simple CLI wrapper around shiftlab.runner.run_experiment

Recommended production usage is:
    python -m shiftlab.runner
"""

import argparse
# Used to define and parse command-line arguments such as:
#   --config configs/default.yaml
# This avoids hard-coding experiment parameters in the source code.

import os
# Provides operating-system–independent utilities.
# Here it is used to:
# - Locate the directory of this file
# - Build paths in a portable way (Windows / Linux / macOS)

import sys
# Gives access to the Python runtime environment.
# Here it is used to modify sys.path so that the "src/" directory
# is importable when running this file directly (python main.py).


# ------------------------------------------------------------
# PyCharm / direct-run compatibility
# ------------------------------------------------------------
# When running this file directly:
#     python main.py
# Python does NOT automatically include "src/" in its import path.
#
# This block ensures that:
#     src/shiftlab/
# can be imported without setting PYTHONPATH manually.
#
# When running as a module (python -m shiftlab.runner),
# this block has no negative effect.
# ------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.dirname(__file__)        # Directory containing main.py
    src_path = os.path.join(project_root, "src")   # <project_root>/src

    if os.path.isdir(src_path):
        sys.path.insert(0, src_path)               # Prepend src/ to import search path


# ------------------------------------------------------------
# Project imports (after sys.path fix)
# ------------------------------------------------------------
from shiftlab.config import load_config
from shiftlab.runner import run_experiment


def main() -> None:
    """
    Parse command-line arguments and run the ShiftLab experiment.

    High-level flow:
    1) Parse CLI arguments (e.g. --config)
    2) Load experiment configuration (YAML / JSON)
    3) Run the full experiment pipeline
    """

    parser = argparse.ArgumentParser(
        description="ShiftLab – Synthetic Data under Distribution Shift (experiment runner)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML or JSON experiment configuration file",
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
