#!/usr/bin/env python3
"""CPU-mock launch for DeepSeek-V4.1-Flash on 8×V100.

Does not allocate Engram tables or load shards. Exit 1 = budget, 2 = NUMA,
3 = SM90 import, 4 = config/remap. 0 = v1 shape is selected.

    PYTHONPATH=python python \\
      scripts/dsv41_v100_cpu_mock_launch.py
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from sglang.srt.mem_cache.dsv41_v100_cpu_mock import main

if __name__ == "__main__":
    raise SystemExit(main())
