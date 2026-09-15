# SPDX-License-Identifier: Apache-2.0
"""Shared decode-scoped gate for optional VAE fast paths."""

from contextlib import contextmanager
from weakref import WeakKeyDictionary

import torch.nn as nn


class VaeFastPathGate:
    """Mutable flag shared by the wrappers installed on one VAE."""

    __slots__ = ("enabled",)

    def __init__(self) -> None:
        self.enabled = False


_VAE_FAST_PATH_GATES: WeakKeyDictionary[nn.Module, VaeFastPathGate] = (
    WeakKeyDictionary()
)


def register_vae_fast_path_gate(vae: nn.Module, gate: VaeFastPathGate) -> None:
    _VAE_FAST_PATH_GATES[vae] = gate


@contextmanager
def use_vae_fast_path(vae: nn.Module, enabled: bool):
    """Enable an installed VAE fast path for one decode and always reset it."""
    gate = _VAE_FAST_PATH_GATES.get(vae)
    if gate is None:
        yield
        return

    previous_enabled = gate.enabled
    gate.enabled = enabled
    try:
        yield
    finally:
        gate.enabled = previous_enabled
