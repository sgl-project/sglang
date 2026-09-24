"""gfx950 GLM-5.2 target-attention projection specializations."""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib import import_module
from typing import Optional, Tuple

import torch
from packaging.version import Version

logger = logging.getLogger(__name__)

_M = 4
_HIDDEN = 6144
_Q_LORA = 2048
_Q_OUT = 4096
_KV_LORA = 512
_ROPE = 64


@lru_cache(maxsize=1)
def is_target_projection_fusion_available() -> bool:
    """Return whether this runtime can JIT the gfx950 Gluon kernels."""
    try:
        triton = import_module("triton")
        if Version(Version(triton.__version__).base_version) < Version("3.5.0"):
            return False

        gl = import_module("triton.experimental.gluon.language")
        cdna4 = gl.amd.cdna4
        for obj, name in (
            (cdna4, "buffer_load"),
            (cdna4, "buffer_store"),
            (cdna4, "mfma"),
            (cdna4.async_copy, "buffer_load_to_shared"),
            (cdna4.async_copy, "commit_group"),
            (cdna4.async_copy, "wait_group"),
        ):
            getattr(obj, name)

        package = __package__
        import_module(f"{package}.target_qkv_a_norm_m4")
        import_module(f"{package}.target_q_b_gemm_m4")
        import_module(f"{package}.target_o_gemm_m4")
    except Exception as exc:
        logger.info("ROCm target projection fusion JIT is unavailable: %s", exc)
        return False
    return True


def _is_bf16_matrix(tensor: torch.Tensor, shape: Tuple[int, int]) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tuple(tensor.shape) == shape
        and tensor.dtype == torch.bfloat16
        and tensor.is_contiguous()
    )


def target_qkv_a_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    q_gamma: torch.Tensor,
    kv_gamma: torch.Tensor,
    *,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Fuse QKV-A projection with Q/K RMSNorm for the target M=4 shape."""
    if (
        not _is_bf16_matrix(hidden_states, (_M, _HIDDEN))
        or not _is_bf16_matrix(weight, (_Q_LORA + _KV_LORA + _ROPE, _HIDDEN))
        or tuple(q_gamma.shape) != (_Q_LORA,)
        or tuple(kv_gamma.shape) != (_KV_LORA,)
        or q_gamma.dtype != torch.bfloat16
        or kv_gamma.dtype != torch.bfloat16
        or not q_gamma.is_contiguous()
        or not kv_gamma.is_contiguous()
        or weight.device != hidden_states.device
        or q_gamma.device != hidden_states.device
        or kv_gamma.device != hidden_states.device
    ):
        return None

    from .target_qkv_a_norm_m4 import mla_qkv_a_norm

    return mla_qkv_a_norm(
        hidden_states,
        weight,
        q_gamma,
        kv_gamma,
        rope_dim=_ROPE,
        eps=eps,
    )


def target_q_b_proj(
    q_lora: torch.Tensor, weight: torch.Tensor
) -> Optional[torch.Tensor]:
    """Run the target Q-B M=4 specialization, or request native fallback."""
    if (
        not _is_bf16_matrix(q_lora, (_M, _Q_LORA))
        or not _is_bf16_matrix(weight, (_Q_OUT, _Q_LORA))
        or weight.device != q_lora.device
    ):
        return None

    from .target_q_b_gemm_m4 import bf16_gemm

    return bf16_gemm(q_lora, weight)


def target_o_proj(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> Optional[torch.Tensor]:
    """Run the target O-projection M=4 specialization, or request fallback."""
    if (
        not _is_bf16_matrix(hidden_states, (_M, _Q_OUT))
        or not _is_bf16_matrix(weight, (_HIDDEN, _Q_OUT))
        or weight.device != hidden_states.device
    ):
        return None

    from .target_o_gemm_m4 import bf16_gemm

    return bf16_gemm(hidden_states, weight)
