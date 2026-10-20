"""Shutdown helper for custom all-reduce communicators."""

from __future__ import annotations

import atexit
from typing import Any


def register_comm_cleanup(comm: Any) -> None:
    """Register an idempotent shutdown for a custom-AR communicator."""

    def _safe_close() -> None:
        try:
            comm.close()
        except Exception:
            pass
        # Disable both class flavors' early-out paths in __del__/close.
        try:
            comm.disabled = True
        except Exception:
            pass
        # CustomAllReduceV2: drop ``obj`` so close() short-circuits next time.
        try:
            delattr(comm, "obj")
        except Exception:
            pass
        # CustomAllreduce: zero ``_ptr`` so close() short-circuits next time.
        try:
            comm._ptr = 0
        except Exception:
            pass

    atexit.register(_safe_close)
