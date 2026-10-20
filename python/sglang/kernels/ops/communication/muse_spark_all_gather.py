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
# SM90 pays the [M, N] gather inline (no cluster boundary to defer it into);
# measured on H200 TP8, the 8192-wide gather outgrows the GEMM saving past 64 rows.
_SM90_MAX_TOKENS = {4096: MAX_TOKENS, 8192: 64}
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


def projection_max_tokens(world_size: int, n: int, device: torch.device) -> int:
    """Largest M that runs as slice GEMM + all-gather instead of a replicated GEMM."""
    if world_size in (4, 8) and n in (4096, 8192):
        capability = torch.cuda.get_device_capability(device)
        if capability == (10, 3):
            return MAX_TOKENS
        if capability == (9, 0):
            return _SM90_MAX_TOKENS[n]
    return SMALL_PROJECTION_MAX_TOKENS


def gemm_ag_supported(
    comm: Optional[CustomAllReduceV2], x: torch.Tensor, weight: torch.Tensor
) -> bool:
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
    return (
        1 <= m <= projection_max_tokens(comm.world_size, n, x.device)
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
