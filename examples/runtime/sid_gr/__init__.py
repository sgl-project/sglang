"""Opt-in SID-compatible Qwen3 arithmetic for the generative recommendation demo."""

from .attention import register_reference_backend
from .paged_attention import register_paged_reference_backend

register_reference_backend()
register_paged_reference_backend()
