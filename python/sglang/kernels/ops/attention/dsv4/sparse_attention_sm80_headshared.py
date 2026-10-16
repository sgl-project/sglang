"""Compatibility alias for the isolated geometry diagnostic (not head sharing).

The launch options now live in the original tiled implementation to keep one
validation and numerical contract. No head-sharing kernel is claimed here.
"""
from .sparse_attention_sm80_tiled import sparse_attention_sm80_tiled

sparse_attention_sm80_headshared = sparse_attention_sm80_tiled
