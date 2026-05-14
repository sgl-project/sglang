"""Process-global capture-mode flags shared by the decode runner and the
speculative-draft runners. Read by model code that needs to take a
capture-time branch (e.g. lora dual-graph capture decides per-batch
which variant to use).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

# Detect whether the current forward pass is in capture mode.
is_capture_mode = False

# When capturing dual MoE backends, tracks which variant is being captured.
# None = not dual, "lora" = capturing lora variant, "nolora" = capturing nolora variant.
_capture_lora_variant: Optional[str] = None


def get_is_capture_mode() -> bool:
    return is_capture_mode


def get_capture_lora_variant() -> Optional[str]:
    """Return the lora variant being captured, or None if not in dual capture."""
    return _capture_lora_variant


def _set_capture_lora_variant(variant: Optional[str]) -> None:
    global _capture_lora_variant
    _capture_lora_variant = variant


@contextmanager
def model_capture_mode():
    global is_capture_mode
    is_capture_mode = True
    try:
        yield
    finally:
        is_capture_mode = False
