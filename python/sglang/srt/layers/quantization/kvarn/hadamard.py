# SPDX-License-Identifier: Apache-2.0
"""Hadamard matrix utilities for KVarN.

Generates a normalised Sylvester-type Hadamard matrix
of size ``d × d`` where ``d`` is a power of 2.
"""

from __future__ import annotations

import functools
import math

import torch


@functools.cache
def hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    """Sylvester Hadamard, normalised, cached per (d, device).

    ``d`` must be a power of 2.
    """
    H = torch.ones(1, 1)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str)).float()


def build_hadamard(d: int, device: torch.device) -> torch.Tensor:
    """Build a normalised Hadamard matrix of size ``d × d``."""
    return hadamard_cached(d, str(device))
