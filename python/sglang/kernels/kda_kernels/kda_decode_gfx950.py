# SPDX-License-Identifier: Apache-2.0
"""HIP Kimi-K3 fused Kimi Delta Attention decode for gfx950 (MI350X / MI355X).

The extension is JIT-compiled with ``torch.utils.cpp_extension`` on first use
and cached under ``TORCH_EXTENSIONS_DIR``.
"""

from __future__ import annotations

import os

import torch

from sglang.kernels.kda_kernels import _cuda_source

_ext = None


def load_ext():
    """Build and cache the gfx950 HIP extension."""
    global _ext
    if _ext is None:
        from sglang.srt.utils.cpp_extension_loader import (
            load_extension_with_recovery,
        )

        # torch's JIT builds for every arch in PYTORCH_ROCM_ARCH
        previous_arch = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
        try:
            _ext = load_extension_with_recovery(
                name="kda_decode_gfx950",
                sources=[_cuda_source("attention/kda_decode_gfx950.cu")],
                extra_cuda_cflags=["-O3"],
                extra_cflags=["-O3"],
            )
        finally:
            if previous_arch is None:
                os.environ.pop("PYTORCH_ROCM_ARCH", None)
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = previous_arch
    return _ext


def kda_decode_gfx950(
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor,
    lower_bound: float,
    norm_eps: float,
) -> None:
    """Writes ``out``; conv_state and state at state_indices are updated in place."""
    load_ext().kda_decode_gfx950(
        f_a,
        f_b_weight,
        mixed_qkv,
        conv_weight,
        conv_state,
        raw_beta,
        A_log,
        dt_bias,
        state,
        state_indices,
        output_gate,
        norm_weight,
        out,
        lower_bound,
        norm_eps,
    )


__all__ = ["kda_decode_gfx950"]
