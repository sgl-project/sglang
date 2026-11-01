# SPDX-License-Identifier: Apache-2.0
"""HIP Kimi Delta Attention chunked prefill for gfx950 (MI350X / MI355X).

``TritonKDAKernel.extend`` tries :func:`extend` first on ROCm. It returns None for
anything the kernel does not cover, and the caller keeps the Triton ``chunk_kda``
path. On by default on gfx950; ``SGLANG_ROCM_KDA_HIP_PREFILL=0`` turns it off.

The kernel is ``attention.kda_prefill_gfx950`` (``KernelBackend.KDA``), implemented
in ``sglang.kernels.kda_kernels.kda_prefill_gfx950``.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import KernelBackend


@lru_cache(maxsize=1)
def available() -> bool:
    """gfx950 only: the kernel uses the 16x16x32 bf16 MFMA and up to 157 KiB of LDS."""
    if os.environ.get("SGLANG_ROCM_KDA_HIP_PREFILL", "1").lower() in ("0", "false"):
        return False
    if not torch.version.hip or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx950"


def extend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    beta_is_raw: bool = False,
    return_intermediate_states: bool = False,
    **kwargs,
):
    """``TritonKDAKernel.extend`` on the HIP kernel, or None when it is not covered.

    The kernel applies the gate (A_log, dt_bias, lower bound -5), the beta sigmoid
    and the q / k L2 norms itself, reads and overwrites ``ssm_states`` at
    ``cache_indices`` in place, and writes the optional fp32 ``track_state``
    snapshot. It cannot return the per-chunk states ``h``.
    """
    track_state = kwargs.get("track_state")
    track_chunk_idx = kwargs.get("track_chunk_idx")
    track_ssm_h_src = kwargs.get("track_ssm_h_src")
    if not (
        available()
        # spec-decode extends stay on the reference path
        and not kwargs.get("is_spec_decode", False)
        # breakable prefill CUDA graphs pass padded chunk tables
        and kwargs.get("chunk_indices") is None
        and (track_state is None) == (track_chunk_idx is None)
        # aligned-only tracking reads the final state, so h is not needed
        and not (
            return_intermediate_states
            and track_state is None
            and (track_ssm_h_src is None or track_ssm_h_src.numel() > 0)
        )
        and lower_bound == -5.0
        and beta_is_raw
        and A_log is not None
        and dt_bias is not None
        and all(x.dtype == torch.bfloat16 for x in (q, k, v, g, beta))
        and ssm_states.dtype == torch.float32
        and ssm_states.is_contiguous()
        and q.shape == k.shape == v.shape == g.shape
        and q.ndim == 4
        and q.shape[0] == 1
        and q.shape[-1] == 128
        and ssm_states.shape[-2:] == (128, 128)
    ):
        return None
    # the launch layout depends on the longest sequence; T bounds it without a sync
    seq_lens = kwargs.get("extend_seq_lens_cpu")
    output = get_kernel("attention.kda_prefill_gfx950", KernelBackend.KDA)(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        A_log.reshape(-1).float().contiguous(),
        dt_bias.reshape(-1).float().contiguous(),
        ssm_states,
        cache_indices.to(torch.int32).contiguous(),
        query_start_loc.to(torch.int32).contiguous(),
        max(seq_lens) if seq_lens else q.shape[1],
        q.shape[-1] ** -0.5,
        track_state=track_state,
        track_chunk_idx=(
            None
            if track_chunk_idx is None
            else track_chunk_idx.to(torch.int64).contiguous()
        ),
    )
    return (output, None) if return_intermediate_states else output
