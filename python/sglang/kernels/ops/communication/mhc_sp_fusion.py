"""mHC sequence-parallel sublayer boundary, fused with its communication.

One kernel per boundary: reduce-scatter of the partial sublayer output, hc post +
next combine + RMSNorm on this rank's rows, all-gather of the normalized next
input; the statistics CTAs of the same launch produce the coefficients per tile.

Two transports, one interface: NVLS multicast from three peers up
(``mhc_sp_fusion_nvls.cuh``), peer load/store at two ranks
(``mhc_sp_fusion_p2p.cuh``), where multicast would cross NVLink twice per byte.

``weight=None`` drops the combine/norm/all-gather tail from either transport:
reduce-scatter + post only, for a seam whose next norm cannot fold.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from tvm_ffi import Module

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.communication.all_reduce import Communicator

TILE_ROWS = 64
PARTIAL_STRIDE = 28  # 24 mixes + 1 sum squares; pad for 16B vec op

# The largest k-split whose statistics shared memory stays under the 195 KiB where
# throughput steps down. Measured on B200; see sgl_kernel/dsv41/mhc.cuh.
BEST_SPLIT_K = {2: 16, 3: 20}


@cache_once
def _jit_mhc_sp_module(
    world_size: int,
    rows_per_cta: int,
    stats: bool,
    num_weight_parts: int,
    split_k: int,
    num_stat_mma_blocks: int,
    num_stat_reduce_blocks: int,
    fp8_ag: bool,
    epilogue: bool,
    use_nvls: bool,
) -> Module:
    args = make_cpp_args(
        world_size,
        rows_per_cta,
        stats,
        num_weight_parts,
        split_k,
        num_stat_mma_blocks,
        num_stat_reduce_blocks,
        fp8_ag,
        epilogue,
    )
    name = "mhc_sp_fusion_nvls" if use_nvls else "mhc_sp_fusion_p2p"
    cls = "MHCSPFusionNVLS" if use_nvls else "MHCSPFusionP2P"
    suffix = "nvls" if use_nvls else "p2p"
    return load_jit(
        name,
        *args,
        cuda_files=[f"deepseek_v4/mhc_sp_fusion_{suffix}.cuh"],
        cuda_wrappers=[("run", f"{cls}<{args}>::run")],
    )


# From custom all reduce v2; only need communicator
_COMM: Optional[Communicator] = None
# For local mega kernel event sync
_COUNTER: Optional[torch.Tensor] = None


def init_mhc_workspace(comm: Communicator, max_rows: int) -> None:
    global _COMM, _COUNTER
    _COMM = comm
    tiles = (max_rows + TILE_ROWS - 1) // TILE_ROWS
    _COUNTER = torch.zeros(2 * tiles, dtype=torch.int32, device="cuda")


def mhc_sp_fusion(
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    weight: Optional[torch.Tensor],  # the next norm's weight; None iff not epilogue
    *,
    # Multicast VAs of the caller's symmetric partial output and next input. Unless
    # `fp8_ag` is on they may be the same buffer; see the kernel's header for why.
    y_mc: int = 0,
    x_mc: int = 0,
    y_ptrs: Optional[Sequence[int]] = None,
    x_ptrs: Optional[Sequence[int]] = None,
    world_size: int,
    row_offset: int,
    total_rows: int,
    eps: float,
    comm_blocks: int,
    rows_per_cta: int = 4,
    fp8_ag: bool = False,
    # this boundary's own stats, computed in-kernel into post / comb / pre
    hc_w: Optional[torch.Tensor] = None,  # [parts, 24, 20480] bf16
    hc_scale: Optional[torch.Tensor] = None,
    hc_base: Optional[torch.Tensor] = None,
    split_k: Optional[int] = None,
    stat_mma_blocks: int = 0,
    stat_red_blocks: int = 4,
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-6,
) -> None:
    """``num_blocks`` CTAs run the communication; the statistics take
    ``stat_mma_blocks + stat_red_blocks`` more, and every one of them has to be resident
    at once or the kernel deadlocks. Pass ``hc_w`` to enable the statistics.
    """
    has_epilogue = weight is not None
    fuse_stats = hc_w is not None
    num_weight_parts = hc_w.shape[0] if fuse_stats else 3
    if split_k is None:
        split_k = BEST_SPLIT_K[num_weight_parts]
    if fuse_stats:
        num_tiles = (residual.shape[0] + TILE_ROWS - 1) // TILE_ROWS
        partial = torch.empty(
            num_tiles * split_k * TILE_ROWS * PARTIAL_STRIDE,
            dtype=torch.float32,
            device=residual.device,
        )
        counters = _COUNTER
    else:
        # The statistics trait is a kernel template parameter either way; give it a shape
        # that satisfies `kNumMMABlocks % SPLIT_K == 0`.
        stat_mma_blocks = split_k
        partial, counters = None, None
    use_nvls = world_size > 2
    if use_nvls:
        assert y_mc and (x_mc or not has_epilogue)
        peer_ptrs = (y_mc, x_mc)
    else:
        assert y_ptrs is not None and x_ptrs is not None
        peer_ptrs = (list(y_ptrs), list(x_ptrs))
    module = _jit_mhc_sp_module(
        world_size,
        rows_per_cta,
        fuse_stats,
        num_weight_parts,
        split_k,
        stat_mma_blocks,
        stat_red_blocks,
        fp8_ag,
        has_epilogue,
        use_nvls,
    )
    module.run(
        _COMM,
        *peer_ptrs,
        residual,
        residual_out,
        post,
        comb,
        pre,
        weight,
        row_offset,
        total_rows,
        eps,
        comm_blocks,
        hc_w,
        hc_scale,
        hc_base,
        partial,
        counters,
        rms_eps,
        hc_eps,
    )
