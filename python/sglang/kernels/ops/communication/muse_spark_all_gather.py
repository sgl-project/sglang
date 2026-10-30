"""Multicast all-gather of bf16 column slices through the CustomAllReduceV2 push workspace.

``out[:, rank slice] = local`` (+ b) (+ c) for every rank's ``[M, N / world]``
slice: one kernel multicast-stores the slice into every peer's push workspace
and Lamport-spins the full rows together. Any TP world size; needs the plane's
multicast mapping. The column-parallel projection that produces the slices
lives in ``sglang.srt.layers.communication.muse_spark_gemm_ag``.

Unlike ``communication.gemm_ag`` (Kimi-K3), whose producer fuses the GEMV for
one fixed TP8 shape, this gathers slices an external GEMM already computed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

SMALL_PROJECTION_MAX_TOKENS = 8
MAX_TOKENS = 512
_VEC = 8


@cache_once
def _jit_module(n: int, world_size: int) -> Module:
    args = make_cpp_args(n, world_size, is_arch_support_pdl())
    return load_jit(
        "muse_spark_all_gather",
        *args,
        cuda_files=["muse_spark/all_gather.cuh"],
        cuda_wrappers=[("run", f"MuseSparkAllGather<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


# Pooler (pre-expert projection) at TP4: the local slice + gather beats the replicated projection
# up to the 8 MB push slot, measured on SM103 and SM107.
POOLER_TP4_MAX_TOKENS = 1024


def gemm_ag_supported(
    comm: Optional[CustomAllReduceV2],
    x: torch.Tensor,
    weight: torch.Tensor,
    pooler: bool = False,
) -> bool:
    """``pooler`` selects the pre-expert projection's rule: TP4/TP8 on SM103 or SM107 (the
    unpooler measured faster on the gather path on SM103 only) and up to POOLER_TP4_MAX_TOKENS at TP4."""
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    if (
        not isinstance(comm, CustomAllReduceV2)
        or comm.disabled
        or not comm.has_multicast
        or comm.world_size < 2
    ):
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or x.dim() != 2:
        return False
    m, k = x.shape
    n = weight.shape[0]
    archs = ((10, 3), (10, 7)) if pooler else ((10, 3),)
    if pooler and (
        comm.world_size not in (4, 8)
        or torch.cuda.get_device_capability(x.device) not in archs
    ):
        return False
    max_tokens = (
        POOLER_TP4_MAX_TOKENS if pooler and comm.world_size == 4 else MAX_TOKENS
    )
    return (
        1 <= m <= max_tokens
        and (
            m <= SMALL_PROJECTION_MAX_TOKENS
            or (
                comm.world_size in (4, 8)
                and n in (4096, 8192)
                and torch.cuda.get_device_capability(x.device) in archs
            )
        )
        and weight.shape[1] == k
        and n % (comm.world_size * _VEC) == 0
        and m * n * 2 <= comm.max_push_size
        and comm.config.num_push_blocks >= 2
        and weight.is_contiguous()
    )


def all_gather(
    comm: CustomAllReduceV2,
    local: torch.Tensor,
    out: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    c: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``out[:, rank slice] = local`` gathered from every rank, (+ b) (+ c)."""
    _jit_module(out.shape[1], comm.world_size).run(
        comm.obj, local.contiguous(), b, c, out
    )
    return out


# norm_fp4_quant's thread count at H = 4096: one 16-element block per thread.
_NORM_QUANT_THREADS = 256


@cache_once
def _norm_quant_module(n: int, world_size: int) -> Module:
    args = make_cpp_args(n, world_size, _NORM_QUANT_THREADS, is_arch_support_pdl())
    return load_jit(
        "muse_spark_all_gather_norm_fp4_quant",
        *args,
        cuda_files=["muse_spark/all_gather_norm_fp4_quant.cuh"],
        cuda_wrappers=[("run", f"MuseSparkAllGatherNormFp4Quant<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def all_gather_norm_fp4_quant_supported(local: torch.Tensor, n: int) -> bool:
    """Shape and dtype only; callers must also pass ``gemm_ag_supported(..., pooler=True)``,
    which checks the push plane, world size, architecture and token cap."""
    return n == 4096 and local.dtype == torch.bfloat16 and local.dim() == 2


def all_gather_norm_fp4_quant(
    comm: CustomAllReduceV2,
    local: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    global_scale: torch.Tensor,
    write_rows: bool = True,
) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Gathered ``[M, N]`` bf16 rows, then their RMS norm as static-scale NVFP4: ``(rows, fp4 [M, N/2], sf [M, N/16])``.

    fp4 / sf are bitwise ``fused_norm_fp4_quant_static`` of the gathered rows; ``rows`` is None unless ``write_rows``.
    """
    m, n_local = local.shape
    n = n_local * comm.world_size
    out = (
        torch.empty(m, n, dtype=torch.bfloat16, device=local.device)
        if write_rows
        else None
    )
    fp4 = torch.empty(m, n // 2, dtype=torch.uint8, device=local.device)
    sf = torch.empty(m, n // 16, dtype=torch.uint8, device=local.device)
    _norm_quant_module(n, comm.world_size).run(
        comm.obj,
        local.contiguous(),
        weight,
        global_scale.reshape(1),
        float(eps),
        out,
        fp4,
        sf,
    )
    return out, fp4, sf
