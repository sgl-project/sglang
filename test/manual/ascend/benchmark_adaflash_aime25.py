#!/usr/bin/env python3
"""Compatibility wrapper for benchmark_adaflash.py defaulting to AIME2025."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_adaflash import main  # noqa: E402

if __name__ == "__main__":
    main(default_dataset="aime")
