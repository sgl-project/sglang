# SPDX-License-Identifier: Apache-2.0
"""Shared decode-scoped gate for optional VAE fast paths."""

from contextlib import contextmanager
from contextvars import ContextVar
from weakref import WeakKeyDictionary

import torch.nn as nn

from sglang.multimodal_gen.configs.sample.sampling_params import quality_allows

# Decode can also run on a streaming worker. Missing scope must be exact.
_allow_approximate = ContextVar("vae_allow_approximate", default=False)


def approximate_decode_enabled() -> bool:
    return _allow_approximate.get()


@contextmanager
def use_vae_decode_quality(vae: nn.Module, quality: str):
    token = _allow_approximate.set(quality_allows(quality, "high"))
    try:
        with use_vae_fast_path(vae, quality_allows(quality, "lossless")):
            yield
    finally:
        _allow_approximate.reset(token)


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
