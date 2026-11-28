"""Fail-closed adapter for AITER's FlyDSL GDR decode behind the GDN packed decode.

The Triton packed decode reads q, k and v out of the post-conv ``mixed_qkv``
row by offset. The FlyDSL kernel takes q, k and v as separate tensors but
accepts any row stride, so the same row is handed over as three strided views
and nothing is copied.
"""

from __future__ import annotations

import functools
import logging
import math

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip, print_warning_once

logger = logging.getLogger(__name__)

_AITER_MODULE = "aiter.ops.flydsl.linear_attention_kernels"
_DTYPES = (torch.bfloat16, torch.float16)
_STATE_DTYPES = (torch.float32, torch.bfloat16)
# Every tiling the launcher can pick, including the default it falls back to,
# divides a K that is a multiple of 64 and a V that is a multiple of 32.
_K_ALIGN = 64
_V_ALIGN = 32


def enabled() -> bool:
    return envs.SGLANG_USE_HIP_GDN_PACKED_DECODE_BACKEND.get().lower() == "aiter"


@functools.cache
def _module():
    """AITER's FlyDSL linear-attention module; None when it does not load."""
    import importlib

    try:
        module = importlib.import_module(_AITER_MODULE)
        module._is_supported_arch
        # Graph padding rows carry a negative slot and must come back zeroed.
        if not getattr(module.flydsl_gdr_decode, "zeroes_invalid_output", False):
            raise AttributeError("flydsl_gdr_decode does not zero padding rows")
    except (ImportError, AttributeError) as error:
        print_warning_once(
            f"AITER FlyDSL GDN decode unavailable ({error}); "
            "decode runs the Triton packed kernel"
        )
        return None
    return module


@functools.cache
def _armed(device: torch.device) -> bool:
    module = _module()
    if module is None or not module._is_supported_arch(device):
        return False
    logger.info("GDN packed decode armed: %s.flydsl_gdr_decode", _AITER_MODULE)
    return True


def available(device: torch.device) -> bool:
    """Whether the FlyDSL decode is selected and runs on ``device``."""
    if not is_hip() or not enabled() or not torch.cuda.is_available():
        return False
    return _armed(device)


def _head_layout(qkv_dim: int, num_v_heads: int, head_v_dim: int, head_k_dim: int):
    """The (q heads, q width) the packed row holds, or None if it does not split."""
    qk_dim = qkv_dim - num_v_heads * head_v_dim
    if qk_dim <= 0 or qk_dim % (2 * head_k_dim):
        return None
    num_k_heads = qk_dim // (2 * head_k_dim)
    if num_v_heads % num_k_heads:
        return None
    return num_k_heads, num_k_heads * head_k_dim


def covered(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    num_v_heads: int,
    head_v_dim: int,
) -> bool:
    """Whether the launch is one the FlyDSL kernel computes exactly as Triton does."""
    if mixed_qkv.ndim != 2 or ssm_states.ndim != 4:
        return False
    batch = mixed_qkv.shape[0]
    dtype = mixed_qkv.dtype
    head_k_dim = ssm_states.shape[-1]
    layout = _head_layout(mixed_qkv.shape[1], num_v_heads, head_v_dim, head_k_dim)
    return (
        batch > 0
        and layout is not None
        and dtype in _DTYPES
        and mixed_qkv.stride(-1) == 1
        and ssm_states.dtype in _STATE_DTYPES
        and ssm_states.shape[1:3] == (num_v_heads, head_v_dim)
        and ssm_states.stride(-1) == 1
        and ssm_states.data_ptr() % 16 == 0
        and head_k_dim % _K_ALIGN == 0
        and head_v_dim % _V_ALIGN == 0
        # The kernel bakes in head_k_dim ** -0.5.
        and math.isclose(scale, head_k_dim**-0.5, rel_tol=1e-6)
        and a.shape == (batch, num_v_heads)
        and b.shape == (batch, num_v_heads)
        and a.dtype == dtype
        and b.dtype == dtype
        and a.stride(-1) == 1
        and b.stride(-1) == 1
        # The kernel reads dt_bias in the activation dtype.
        and dt_bias.dtype == dtype
        and dt_bias.shape == (num_v_heads,)
        and dt_bias.stride(0) == 1
        and A_log.dtype in (torch.float32, torch.bfloat16)
        and A_log.shape == (num_v_heads,)
        and A_log.stride(0) == 1
        and cache_indices.shape == (batch,)
        and cache_indices.dtype == torch.int32
        and cache_indices.stride(0) == 1
    )


def run(
    *,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    num_v_heads: int,
    head_v_dim: int,
) -> torch.Tensor:
    """Packed decode through FlyDSL; output [B, 1, HV, V] like the Triton kernel."""
    module = _module()
    if module is None:
        raise RuntimeError("AITER FlyDSL GDN decode is unavailable")
    batch = mixed_qkv.shape[0]
    head_k_dim = ssm_states.shape[-1]
    num_k_heads, q_dim = _head_layout(
        mixed_qkv.shape[1], num_v_heads, head_v_dim, head_k_dim
    )
    query = mixed_qkv[:, :q_dim].unflatten(-1, (num_k_heads, head_k_dim))
    key = mixed_qkv[:, q_dim : 2 * q_dim].unflatten(-1, (num_k_heads, head_k_dim))
    value = mixed_qkv[:, 2 * q_dim :].unflatten(-1, (num_v_heads, head_v_dim))
    # Negative (graph padding) slots come back as zero rows, so empty is safe.
    out = mixed_qkv.new_empty(batch, 1, num_v_heads, head_v_dim)
    module.flydsl_gdr_decode(
        query.unsqueeze(1),
        key.unsqueeze(1),
        value.unsqueeze(1),
        a.unsqueeze(1),
        b.unsqueeze(1),
        dt_bias,
        A_log,
        cache_indices,
        ssm_states,
        out,
        use_qk_l2norm=True,
        need_shuffle_state=False,
    )
    return out
