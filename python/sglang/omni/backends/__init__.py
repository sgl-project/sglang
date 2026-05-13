# SPDX-License-Identifier: Apache-2.0
"""Execution backends for omni orchestration.

Backends drive AR and multimodal generation engines. Model-specific token
grammar belongs in `sglang.omni.model_adapters`.
"""

from sglang.omni.backends.ar.base import UnsupportedARBackend
from sglang.omni.backends.mm_gen.base import UnsupportedGenerationBackend

__all__ = [
    "UnsupportedARBackend",
    "UnsupportedGenerationBackend",
]
