# SPDX-License-Identifier: Apache-2.0
"""HIP Kimi Delta Attention chunked prefill for gfx950 (MI350X / MI355X).

The extension is JIT-compiled with ``torch.utils.cpp_extension`` on first use
and cached under ``TORCH_EXTENSIONS_DIR``.
"""

from __future__ import annotations

import os

import torch

from sglang.kernels.kda_kernels import _cuda_source

_ext = None


def _load():
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
                name="kda_prefill_gfx950",
                sources=[_cuda_source("attention/kda_prefill_gfx950.cu")],
                extra_cuda_cflags=["-O3", "-mllvm", "-amdgpu-mfma-vgpr-form"],
                extra_cflags=["-O3"],
            )
        finally:
            if previous_arch is None:
                os.environ.pop("PYTORCH_ROCM_ARCH", None)
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = previous_arch
    return _ext


def kda_prefill_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
    track_state: torch.Tensor | None = None,
    track_chunk_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Returns the output; ssm_states[cache_indices] is updated in place."""
    empty = q.new_empty(0, dtype=torch.float32)
    return _load().kda_prefill_gfx950(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        cu_seqlens,
        max_seqlen,
        empty if track_state is None else track_state,
        empty.long() if track_chunk_idx is None else track_chunk_idx,
        scale,
    )


__all__ = ["kda_prefill_gfx950"]
