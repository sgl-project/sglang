#!/usr/bin/env python3
"""Dry-run HBM + host-NUMA allocator for DeepSeek-V4.1-Flash on 8×V100.

CPU only. No weight download. No GPU.

    PYTHONPATH=/tmp/sglang-dsv41-v100/python \\
      python \\
      scripts/dsv41_v100_dryrun_allocator.py

Exit 1 if any rank's HBM is over 31 GiB. Exit 2 if Engram/spill cannot sit on
NUMA node 1 without silently crossing UPI (pass --allow-numa-split to place
overflow on node 0 and warn).
"""

from __future__ import annotations

import os
import sys

# Allow `python scripts/dsv41_v100_dryrun_allocator.py` from the worktree root
# without requiring the caller to set PYTHONPATH (venv still needed for torch
# if something else imports sglang; this module itself is torch-free).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from sglang.srt.mem_cache.dsv41_v100_budget import main

if __name__ == "__main__":
    raise SystemExit(main())
