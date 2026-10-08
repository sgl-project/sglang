"""Kimi-K3's opt-in absorbed value projection plus BF16 output gate.

The kernel consumes the existing transposed BF16 weight directly. It preserves
the native BF16 rounding after BMM and sigmoid, before the final multiplication.
Gate projection, stream joins, output projection, and TP reduction stay with the
model. Static eligibility is checked after loading; runtime checks are metadata
only and are safe during decode graph capture.
"""

from __future__ import annotations

import os

import torch

from sglang.kernels.ops.attention.kda_whole_layer_gluon_hip import (
    _has_required_gluon_api,
    _rocm_arch,
)
from sglang.srt.utils import is_hip

_DECODE_ROWS = frozenset((1, 2, 4, 8, 16, 32, 64, 128, 256))


def enabled() -> bool:
    return os.environ.get("SGLANG_ROCM_K3_MLA_VC_FUSED_BACKEND", "").lower() == "gluon"


def entrypoint_name(rows: int) -> str | None:
    if rows in _DECODE_ROWS:
        return f"mla_vc_output_gate_m{rows}"
    if 1024 <= rows <= 8192:
        return "mla_vc_output_gate_m1024_8192"
    return None


def can_prepare(attn, parallel, server_args) -> bool:
    """Admit only the loaded TP8 BF16 K3 value/gate contract."""
    if not enabled() or not is_hip() or not _has_required_gluon_api():
        return False
    weight = getattr(attn, "w_vc", None)
    return (
        isinstance(weight, torch.Tensor)
        and weight.is_cuda
        and _rocm_arch(weight.device.index) == "gfx950"
        and attn.use_output_gate
        and not attn.use_dsa
        and not attn.use_deep_gemm_bmm
        and parallel.attn_tp_size == 8
        and not parallel.dcp_enabled
        and not getattr(server_args, "enable_lora", False)
        and not getattr(server_args, "speculative_algorithm", None)
        and type(attn.w_scale) in (float, int)
        and attn.w_scale == 1.0
        and weight.dtype
        == attn.w_kc.dtype
        == attn.o_proj.weight.dtype
        == torch.bfloat16
        and tuple(weight.shape) == (12, 512, 128)
        and weight.stride() == (65536, 1, 512)
        and weight.storage_offset() == 0
    )


def covered(attn, latent: torch.Tensor) -> bool:
    """Check mutable inputs after can_prepare has established device support."""
    hidden = attn._gate_hidden_states
    weight = attn.w_vc
    return (
        latent.ndim == 3
        and entrypoint_name(latent.shape[0]) is not None
        and tuple(latent.shape[1:]) == (12, 512)
        and latent.dtype == torch.bfloat16
        and latent.is_contiguous()
        and latent.storage_offset() == 0
        and latent.device == weight.device
        and tuple(weight.shape) == (12, 512, 128)
        and weight.dtype == torch.bfloat16
        and weight.stride() == (65536, 1, 512)
        and weight.storage_offset() == 0
        and isinstance(hidden, torch.Tensor)
        and tuple(hidden.shape) == (latent.shape[0], 7168)
        and hidden.dtype == torch.bfloat16
        and hidden.device == latent.device
    )


def run(latent: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor):
    from sglang.kernels.ops.attention.mla_gluon import kernels

    rows = latent.shape[0]
    name = entrypoint_name(rows)
    if name is None:
        raise ValueError(f"Unqualified Kimi-K3 MLA value/gate M={rows}")
    if (
        gate.dtype != torch.bfloat16
        or tuple(gate.shape) != (rows, 1536)
        or not gate.is_contiguous()
        or gate.device != latent.device
    ):
        raise RuntimeError("Kimi-K3 MLA output gate layout changed")
    return getattr(kernels, name)(latent, weight, gate)


def apply(attn, latent: torch.Tensor):
    """Consume gate state only after successful fused value/gate execution."""
    gate = attn._compute_output_gate(attn._gate_hidden_states)
    output = run(latent, attn.w_vc, gate)
    if (
        output.dtype != torch.bfloat16
        or tuple(output.shape) != (latent.shape[0], 1536)
        or not output.is_contiguous()
        or output.device != latent.device
        or output.untyped_storage().data_ptr()
        in {x.untyped_storage().data_ptr() for x in (latent, attn.w_vc, gate)}
    ):
        raise RuntimeError("Kimi-K3 MLA value/gate output ABI changed")
    # o_proj's existing gate wrapper observes this and does not gate twice.
    attn._gate_hidden_states = None
    return output
