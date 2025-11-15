"""Minimal Pillow stub for offline environments.

This package only implements the interfaces required by the multimodal
processor shim. It is **not** a full featured drop-in replacement for Pillow.
"""
from . import Image as Image  # re-export module
__all__ = ["Image"]
