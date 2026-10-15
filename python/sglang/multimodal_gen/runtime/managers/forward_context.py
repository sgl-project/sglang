# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/forward_context.py
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.attention import AttentionMetadata
    from sglang.multimodal_gen.runtime.pipelines_core import Req


@dataclass
class ForwardContext:
    current_timestep: int
    attn_metadata: "AttentionMetadata"  # set dynamically for each forward pass
    forward_batch: Optional["Req"] = None


_forward_context: Optional["ForwardContext"] = None


def get_forward_context() -> "ForwardContext":
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


def get_forward_context_or_none() -> "ForwardContext | None":
    return _forward_context


@contextmanager
def set_forward_context(
    current_timestep, attn_metadata, forward_batch: Optional["Req"] = None
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global _forward_context
    prev_context = _forward_context
    _forward_context = ForwardContext(
        current_timestep=current_timestep,
        attn_metadata=attn_metadata,
        forward_batch=forward_batch,
    )

    try:
        yield
    finally:
        _forward_context = prev_context
