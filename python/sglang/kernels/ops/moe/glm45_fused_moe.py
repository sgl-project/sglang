"""H200 fast path for the TP=8 GLM-4.5-FP8 fused MoE layout.

The CUDA kernel is intentionally narrow. Callers must pass the production
GLM-4.5 tensor layout exactly; all other MoE configurations stay on the
existing backend. The implementation uses a fixed-capacity workspace so its
address remains stable for the lifetime of the worker process.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MAX_TOKENS = 8192
_NUM_EXPERTS = 161
_HIDDEN_SIZE = 5120
_GATE_UP_SIZE = 384
_INTERMEDIATE_SIZE = 192
_TOP_K = 9

_WORKSPACES: dict[int, torch.Tensor] = {}


@cache_once
def _jit_glm45_fused_moe_module() -> Module:
    return load_jit(
        "glm45_fused_moe_sm90",
        cuda_files=[
            "moe/glm45_fused_moe_sm90/entry.cu",
            "moe/glm45_fused_moe_sm90/kernel.cu",
        ],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        header_only=False,
    )


@functools.lru_cache(maxsize=16)
def _is_h200(device_index: int) -> bool:
    properties = torch.cuda.get_device_properties(device_index)
    return properties.major == 9 and properties.minor == 0 and "H200" in properties.name


def covered(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
) -> bool:
    """Return whether tensors match the specialized GLM-4.5 H200 ABI."""
    if not hidden_states.is_cuda or torch.cuda.is_current_stream_capturing():
        return False
    device_index = hidden_states.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if not _is_h200(device_index):
        return False
    if w1_scale is None or w2_scale is None:
        return False
    tokens = hidden_states.shape[0] if hidden_states.ndim == 2 else 0
    tensors = (hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale)
    return (
        1 <= tokens <= _MAX_TOKENS
        and hidden_states.shape == (tokens, _HIDDEN_SIZE)
        and w1.shape == (_NUM_EXPERTS, _GATE_UP_SIZE, _HIDDEN_SIZE)
        and w2.shape == (_NUM_EXPERTS, _HIDDEN_SIZE, _INTERMEDIATE_SIZE)
        and topk_weights.shape == (tokens, _TOP_K)
        and topk_ids.shape == (tokens, _TOP_K)
        and w1_scale.numel() == _NUM_EXPERTS * _GATE_UP_SIZE
        and w2_scale.numel() == _NUM_EXPERTS * _HIDDEN_SIZE
        and hidden_states.dtype == torch.bfloat16
        and w1.dtype == torch.float8_e4m3fn
        and w2.dtype == torch.float8_e4m3fn
        and topk_weights.dtype == torch.float32
        and topk_ids.dtype == torch.int32
        and w1_scale.dtype == torch.float32
        and w2_scale.dtype == torch.float32
        and all(t.device == hidden_states.device for t in tensors)
        and all(t.is_contiguous() for t in tensors)
    )


def _workspace(device: torch.device) -> torch.Tensor:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    workspace = _WORKSPACES.get(device_index)
    if workspace is None:
        size = int(_jit_glm45_fused_moe_module().workspace_size())
        workspace = torch.empty(size, dtype=torch.uint8, device=device)
        _WORKSPACES[device_index] = workspace
    return workspace


def glm45_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """Run the in-place fused MoE kernel after ``covered`` succeeds."""
    _jit_glm45_fused_moe_module().run(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        _workspace(hidden_states.device),
    )
    return hidden_states


__all__ = ["covered", "glm45_fused_moe"]
