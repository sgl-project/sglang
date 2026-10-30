"""GEMM collectives for Blackwell prefill sequence parallelism.

Torch symmetric memory is used only to allocate/register NVLink-accessible
buffers. AG dispatches between a tile-gated CUTLASS GEMM and per-shard cuBLAS
GEMMs; RS uses a fused CUTLASS kernel.

The symmetric-memory workspaces are owned by
``sglang.srt.layers.communication.sp_collective_gemm_workspace``: each entry
point takes its getter as ``workspace`` and calls it with the tile-padded shape
``(kind, group, rows, width, x, row_align, flag_tile_mn)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from sglang.srt.layers.communication.sp_collective_gemm_workspace import (
        SPCollectiveGemmWorkspace,
    )

_TILE_M = 64
_TILE_N = 128
_AG_TILES = (
    (256, 256, True, (2, 1)),
    (128, 256, False, (1, 1)),
    (64, 128, False, (1, 1)),
)
_RS_TILES = ((128, 256, False, (1, 1)), (64, 128, False, (1, 1)))
_compiled = {}


def _use_ag_cublas(x: torch.Tensor, world: int) -> bool:
    return x.dtype == torch.bfloat16 and x.shape[0] * world >= 2048


def _pick_tile(tiles, n: int):
    """(mma_m, mma_n, use_2cta, cluster_mn) for an N that the tile must divide."""
    return next(t for t in tiles if n % t[1] == 0)


def _ag_tile(n: int):
    return _pick_tile(_AG_TILES, n)


def _rs_tile(n: int):
    return _pick_tile(_RS_TILES, n)


def _pad_shards(
    x: torch.Tensor, world_size: int, tile_m: int = _TILE_M
) -> torch.Tensor:
    """Pad INSIDE each rank's token range, never just at the end of the batch."""
    rows = x.shape[0] // world_size
    padded = (rows + tile_m - 1) // tile_m * tile_m
    if rows == padded:
        return x
    shards = x.reshape(world_size, rows, x.shape[1])
    return torch.nn.functional.pad(shards, (0, 0, 0, padded - rows)).reshape(
        world_size * padded, x.shape[1]
    )


def _unpad_shards(x: torch.Tensor, rows: int, world_size: int) -> torch.Tensor:
    return x.reshape(world_size, -1, x.shape[1])[:, :rows].reshape(
        world_size * rows, x.shape[1]
    )


def _tensor(x):
    from cutlass.cute.runtime import from_dlpack

    if x.ndim == 2:
        x = x.unsqueeze(-1)
    return from_dlpack(x, assumed_align=16).mark_layout_dynamic(
        leading_dim=1 if x.ndim == 3 else 0
    )


def _check_inputs(x, weight, group):
    if (
        not x.is_cuda
        or x.ndim != 2
        or weight.ndim != 2
        or x.dtype not in (torch.bfloat16, torch.float16)
        or weight.dtype != x.dtype
        or weight.device != x.device
        or weight.shape[1] != x.shape[1]
        or not weight.is_contiguous()
        or x.shape[1] % 8
        or weight.shape[0] % _TILE_N
    ):
        raise ValueError(
            "CUTLASS SP requires CUDA bf16/fp16 [M,K], contiguous [N,K], K%8=N%128=0"
        )
    if torch.cuda.get_device_capability(x.device)[0] != 10:
        raise ValueError("CUTLASS SP collectives require NVLink Blackwell GPUs")
    world = dist.get_world_size(group)
    if world not in (2, 4, 8):
        raise ValueError("CUTLASS SP collectives require 2, 4, or 8 ranks")
    return world, dist.get_rank(group)


def _all_gather_matmul_cublas(
    local: torch.Tensor,
    weight: torch.Tensor,
    ws: SPCollectiveGemmWorkspace,
    rank: int,
    rows: int,
    out: torch.Tensor,
) -> None:
    """Overlap peer copies with per-shard GEMMs using cuStreamWaitValue32."""
    import cuda.bindings.driver as cuda

    padded, k = local.shape
    gathered = ws.gathered[: padded * ws.world]
    main = torch.cuda.current_stream()
    # Protect the previous invocation's peer reads, then publish the new shard.
    ws.handle.barrier(channel=0)
    ws.storage[:padded].copy_(local)
    ws.flags.zero_()
    ws.handle.barrier(channel=0)
    # Capture the staging dependency before GEMMs wait on the copy stream.
    ws.copy_stream.wait_stream(main)
    size = padded * k * local.element_size()
    flag_base = ws.flags.data_ptr()
    flag_bytes = ws.flags.element_size()
    with torch.cuda.stream(ws.copy_stream):
        stream = cuda.CUstream(ws.copy_stream.cuda_stream)
        for step in range(1, ws.world):
            peer = (rank + step) % ws.world
            (status,) = cuda.cuMemcpyDtoDAsync(
                gathered.data_ptr() + peer * size,
                ws.handle.buffer_ptrs[peer],
                size,
                stream,
            )
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"SP cuBLAS peer copy failed: {status}")
            # Default flags fence the peer copy's writes before publishing readiness.
            (status,) = cuda.cuStreamWriteValue32(
                stream, cuda.CUdeviceptr(flag_base + peer * flag_bytes), 1, 0
            )
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"SP cuBLAS readiness flag set failed: {status}")
    weight_t = weight.t()
    main_stream = cuda.CUstream(main.cuda_stream)
    for step in range(ws.world):
        peer = (rank + step) % ws.world
        if step:
            (status,) = cuda.cuStreamWaitValue32(
                main_stream,
                cuda.CUdeviceptr(flag_base + peer * flag_bytes),
                1,
                cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
            )
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"SP cuBLAS readiness wait failed: {status}")
        # Keep communication padded, but write valid rows directly to compact output.
        # The local shard is already staged; only remote shards need gathering.
        torch.mm(
            ws.storage[:rows]
            if step == 0
            else gathered[peer * padded : peer * padded + rows],
            weight_t,
            out=out[peer * rows : (peer + 1) * rows],
        )
    main.wait_stream(ws.copy_stream)


def all_gather_matmul(
    x,
    weight,
    group,
    *,
    workspace: Callable[..., SPCollectiveGemmWorkspace],
    backend: str = "auto",
):
    """all_gather(x, dim=0) @ weight.T, with shape-based backend selection.

    Backend overrides must agree across ranks and are fixed at graph capture.
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from sglang.kernels.ops.communication.cutlass_all_gather_gemm import (
        PersistentDenseGemmKernel,
        ReleaseFlagSet,
    )

    if backend not in ("auto", "cutlass", "cublas"):
        raise ValueError(f"Unknown SP AG backend: {backend}")
    world, rank = _check_inputs(x, weight, group)
    rows, k = x.shape
    n = weight.shape[0]
    if rows == 0:
        return x.new_empty((0, n))
    if backend == "auto":
        backend = "cublas" if _use_ag_cublas(x, world) else "cutlass"
    mma_m, mma_n, use_2cta, cluster = _ag_tile(n)
    local = _pad_shards(x.contiguous(), 1, mma_m)
    padded = local.shape[0]
    ws = workspace("ag", group, padded, k, x, mma_m)
    gathered = ws.gathered[: padded * world]
    if backend == "cublas":
        out = x.new_empty((rows * world, n))
        _all_gather_matmul_cublas(local, weight, ws, rank, rows, out)
        return out

    out = x.new_empty((padded * world, n))
    a, b, c = _tensor(gathered), _tensor(weight), _tensor(out)
    flags = _tensor(ws.flags)
    main = torch.cuda.current_stream()
    stream = cuda.CUstream(main.cuda_stream)
    key = ("ag", x.device, x.dtype, k, n, world)
    if key not in _compiled:
        gemm = PersistentDenseGemmKernel(
            cutlass.Float32,
            use_2cta,
            (mma_m, mma_n),
            cluster,
            True,
            gated_a_load=True,
        )
        # Keep SMs available for the copy-stream flag publication kernel.
        fn = cute.compile(
            gemm,
            a,
            b,
            c,
            (ws.sms - 2) // (cluster[0] * cluster[1]),
            stream,
            rows_per_rank=cutlass.Int32(padded),
            gate_a_flags=flags,
        )
        flag_fn = cute.compile(ReleaseFlagSet(), flags, cutlass.Int32(0), stream)
        _compiled[key] = fn, flag_fn
    fn, flag_fn = _compiled[key]

    # First barrier protects the previous invocation's peer reads; the second
    # publishes the new local shard. Never reset flags while a peer uses them.
    ws.handle.barrier(channel=0)
    ws.storage[:padded].copy_(local)
    ws.flags.zero_()
    ws.handle.barrier(channel=0)
    gathered[rank * padded : (rank + 1) * padded].copy_(local)
    flag_fn(flags, cutlass.Int32(rank), stream)
    # Record this dependency BEFORE launching a GEMM that waits on copy flags.
    ws.copy_stream.wait_stream(main)
    fn(a, b, c, stream, rows_per_rank=cutlass.Int32(padded), gate_a_flags=flags)
    with torch.cuda.stream(ws.copy_stream):
        copy_stream = cuda.CUstream(ws.copy_stream.cuda_stream)
        size = padded * k * x.element_size()
        for step in range(1, world):
            peer = (rank + step) % world
            (status,) = cuda.cuMemcpyDtoDAsync(
                gathered.data_ptr() + peer * size,
                ws.handle.buffer_ptrs[peer],
                size,
                copy_stream,
            )
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"CUTLASS SP peer copy failed: {status}")
            flag_fn(flags, cutlass.Int32(peer), copy_stream)
    main.wait_stream(ws.copy_stream)
    return _unpad_shards(out, rows, world)


def matmul_reduce_scatter(
    x, weight, group, *, workspace: Callable[..., SPCollectiveGemmWorkspace]
):
    """reduce_scatter(x @ weight.T, SUM, dim=0), one fused GEMM-RS kernel."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from sglang.kernels.ops.communication.cutlass_gemm_reduce_scatter import (
        Sm100PersistentDenseGemmReduceScatterLDMCKernel,
    )

    world, rank = _check_inputs(x, weight, group)
    if x.shape[0] % world:
        raise ValueError("GEMM-RS input rows must be padded to the TP world size")
    rows = x.shape[0] // world
    n, k = weight.shape
    if rows == 0:
        return x.new_empty((0, n))
    mma_m, mma_n, use_2cta, cluster = _rs_tile(n)
    cta_m = mma_m // (2 if use_2cta else 1)
    padded_input = _pad_shards(x.contiguous(), world, cta_m)
    m = padded_input.shape[0]
    ws = workspace("rs", group, m, n, x, cta_m, (cta_m, mma_n))
    out = x.new_empty((m // world, n))
    args = dict(
        a=_tensor(padded_input),
        b=_tensor(weight),
        c=_tensor(ws.storage[:m]),
        comm_in_multicast_tensor=_tensor(ws.multicast[:m]),
        comm_out_unicast_tensor=_tensor(out),
        barrier_flag_unicast=_tensor(ws.flags),
        barrier_flag_multicast=_tensor(ws.flag_multicast),
        barrier_flag_unicast_per_peer=tuple(_tensor(t) for t in ws.peer_flags),
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    key = ("rs", x.device, x.dtype, k, n, world, rank)
    if key not in _compiled:
        dtype = cutlass.BFloat16 if x.dtype == torch.bfloat16 else cutlass.Float16
        gemm = Sm100PersistentDenseGemmReduceScatterLDMCKernel(
            cutlass.Float32,
            dtype,
            use_2cta,
            (mma_m, mma_n),
            cluster,
            True,
            rank,
            world,
        )
        gemm.can_implement((m, n, k, 1), dtype, dtype, "k", "k", "n")
        _compiled[key] = cute.compile(
            gemm,
            **args,
            max_active_clusters=ws.sms // (cluster[0] * cluster[1]),
        )
    # The kernel's exit-flag offsets depend on M. Even after this rank exits,
    # a peer may still be resetting its previous exit slots. A changed M can
    # reuse those slots as producer flags, so fence ALL peers before reuse.
    ws.handle.barrier(channel=0)
    _compiled[key](**args)
    return out[:rows]
