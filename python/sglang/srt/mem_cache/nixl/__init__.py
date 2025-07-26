"""
NIXL (NVIDIA I/O Xfer Library) integration for HiCache.

This module provides high-performance storage using NIXL file plugins.
"""

from .hicache_nixl import HiCacheNixl
from .nixl_utils import NixlBackendSelection, NixlRegistration, NixlFileManager

__all__ = [
    'HiCacheNixl',
    'NixlBackendSelection', 
    'NixlRegistration',
    'NixlFileManager'
] 