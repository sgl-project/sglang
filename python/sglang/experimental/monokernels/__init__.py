# SPDX-License-Identifier: Apache-2.0
"""Monokernels for SGLang (RFC #43949): persistent kernels that run most of a decoder layer, including its
tensor-parallel all-reduce, in one or two launches. SGLang only calls these hooks; with
``--enable-monokernel`` off, none of them changes anything."""

from sglang.experimental.monokernels.guards import MonoRefused, handle_monokernel
from sglang.experimental.monokernels.runtime import (
    MonokernelError,
    attach,
    kernel,
    load_tensor,
    loaded,
    route,
)

__all__ = [
    "MonoRefused",
    "MonokernelError",
    "attach",
    "handle_monokernel",
    "kernel",
    "load_tensor",
    "loaded",
    "route",
]
