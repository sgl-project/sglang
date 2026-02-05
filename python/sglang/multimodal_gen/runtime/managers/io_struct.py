"""
I/O data structures for diffusion engine scheduler.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class UpdateWeightsFromDiskReq:
    """Request to update model weights from disk for diffusion models."""

    model_path: str
    load_format: str = "auto"
    flush_cache: bool = True
    target_modules: Optional[List[str]] = None
