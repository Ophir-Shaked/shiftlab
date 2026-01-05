# -*- coding: utf-8 -*-
# Declares the source file encoding (safe default for text + comments)
from __future__ import annotations
# Enables postponed evaluation of type annotations
# (avoids runtime import issues and improves typing performance)

from typing import Any, Dict
# Used for type hints: the config is returned as Dict[str, Any]

from pathlib import Path
# Modern, cross-platform path handling (preferred over os.path)


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a configuration file from disk.

    Behavior:
    - If the file ends with .json -> load using the built-in json module
    - Otherwise -> assume YAML and load using PyYAML (if installed)

    Design goals:
    - Minimal mandatory dependencies
    - Clear, explicit error messages
    """

    # Convert the input string path into a Path object
    # This makes path checks and file handling safer and cleaner
    path_obj = Path(path)

    # Fail fast if the config file does not exist
    # This avoids silent misconfigurations later in the pipeline
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # ------------------------------------------------------------
    # JSON loading branch (no external dependencies)
    # ------------------------------------------------------------

    # If the filename ends with ".json", treat it as JSON
    if path.lower().endswith(".json"):
        import json  # Standard library import (lazy-loaded)

        # Open the JSON file and parse it into a Python dictionary
        with open(path_obj, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------
    # YAML loading branch (optional dependency)
    # ------------------------------------------------------------

    # Attempt to import PyYAML only if needed
    try:
        import yaml  # type: ignore
        # type: ignore -> suppresses static type checker warnings
        # because PyYAML does not ship with type stubs by default
    except ImportError as e:
        # Raise a clear, actionable error message
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. "
            "Install 'pyyaml' or use a .json config."
        ) from e

    # Open the YAML file and parse it safely into a Python dictionary
    with open(path_obj, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
