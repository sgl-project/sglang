"""K3 fused o_proj GEMM + all-reduce for decode (bf16, TP row-parallel).

One entry point over ``csrc/kimi_k3/comm/gemm_ar.cuh``: a single kernel per
rank computes the local ``x_r [M, K] @ W_r [7168, K]^T`` partial AND the
cross-rank sum — the epilogue pushes finished tiles straight into a
peer-mapped P2P comm region, one flag boundary, then a tile-local reduce
writes the fully reduced ``out [M, 7168]`` on every rank. Replaces the
o_proj GEMM + NCCL all-reduce pair with one launch (see GEMM_AR_README.md).

Contracts:

* bf16 only; ``out = sum_r bf16(x_r @ W_r^T)`` (partials round to bf16
  pre-sum — same numerics as the unfused bf16 GEMM + ring AR).
* M in [1, 512]; internally rounded up to a tuned cell {8, 16, 32, 64,
  128, 256, 512}. ``out`` is allocated with ``cell`` rows and sliced.
* SM100+ with full NVLink P2P (fabric/MNNVL across nodes); perf-tuned on
  GB300 (sm_103a).
* CUDA-graph compatible: the per-cell launch epoch lives in device memory
  (read at kernel entry, bumped by a trailing kernel), so replays advance
  it naturally. Each dispatch cell owns its own flag-ring family — no
  host-side ring reset, ever.
* All TP ranks must call :func:`o_proj_gemm_ar` with the same M in
  lockstep (same stream order of cells on every rank).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

N = 7168  # K3 hidden size (OPROJ_N compile-time default in gemm_ar.cuh)
MAX_TOKENS = 512  # kMMax


@cache_once
def _jit_module(k: int, world_size: int) -> Module:
    args = make_cpp_args(
        k,
        world_size,
        is_arch_support_pdl(),
    )
    cls = f"GemmArKernel<{args}>"
    return load_jit(
        "kimi_k3_gemm_ar",
        *args,
        cuda_files=["kimi_k3/comm/gemm_ar.cuh"],
        cuda_wrappers=[
            ("run", f"{cls}::run"),
            ("set_bases", f"{cls}::set_bases"),
            ("region_nbytes", f"{cls}::region_nbytes"),
            ("gather_words", f"{cls}::gather_words"),
            ("num_fams", f"{cls}::num_fams"),
        ],
        extra_cuda_cflags=["-O3"],
        extra_dependencies=["cutlass"],
    )


class _State(NamedTuple):
    world_size: int
    rank: int
    region: tuple  # (slab tensor, peer buffer views) — keeps mappings alive
    uc_bases: torch.Tensor  # [R] int64 CPU: per-rank UC VAs of the region
    gather: torch.Tensor  # [kFams * 2 * kRing] int32 CUDA, device-local
    epochs: torch.Tensor  # [kFams] int32 CUDA: device-resident CTA ticket counters


_STATE: Optional[_State] = None


def init(
    *,
    world_size: int,
    rank: int,
    group: torch.distributed.ProcessGroup,
    k: int,
) -> None:
    """Allocate + rendezvous the P2P comm region (collective; call once from
    every TP rank, BEFORE any CUDA-graph capture). ``group`` is the TP CPU
    (gloo) group used for the symm-mem rendezvous."""
    global _STATE
    if _STATE is not None:
        return
    # the empty_strided_p2p + get_buffer path is the one that exchanges
    # fabric handles and maps every peer (incl. cross-node MNNVL) into this
    # process — the mem-pool rendezvous(tensor, group_name) API leaves
    # remote-node (and sometimes even local) peers unmapped.
    from torch._C._distributed_c10d import _SymmetricMemory

    mod = _jit_module(k, world_size)
    nbytes = int(mod.region_nbytes())
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.__version__ < "2.11.0":
        import torch.distributed._symmetric_memory as torch_symm_mem

        torch_symm_mem.enable_symm_mem_for_group(group.group_name)
    region = _SymmetricMemory.empty_strided_p2p(
        (nbytes,), [1], torch.uint8, device, group.group_name
    )
    symm = _SymmetricMemory.rendezvous(region)
    region.zero_()
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    # keep the peer buffer tensors alive alongside the region
    peer_bufs = [symm.get_buffer(r, [nbytes], torch.uint8) for r in range(world_size)]
    ptrs = [t.data_ptr() for t in peer_bufs]
    import logging

    logging.getLogger(__name__).info(
        "gemm_ar comm region: rank=%d nbytes=%d uc_bases=%s",
        rank,
        nbytes,
        [hex(p) for p in ptrs],
    )
    assert all(p != 0 for p in ptrs), f"gemm_ar: null peer pointers {ptrs}"
    # explicit cpu: model build may run under a cuda default-device context,
    # and a silently-cuda tensor here means the host-side deref in set_bases
    # reads a device pointer (segfault)
    uc_bases = torch.tensor(ptrs, dtype=torch.int64, device="cpu")
    gather = torch.zeros(int(mod.gather_words()), dtype=torch.int32, device=device)
    epochs = torch.zeros(int(mod.num_fams()), dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    _STATE = _State(
        world_size=world_size,
        rank=rank,
        region=(region, peer_bufs),
        uc_bases=uc_bases,
        gather=gather,
        epochs=epochs,
    )


def initialized() -> bool:
    return _STATE is not None


@cache_once
def _module_with_bases(k: int, world_size: int) -> Module:
    """The per-K JIT module with the comm-region base addresses stashed
    host-side (per-call CPU-tensor derefs from inside the custom op segfault
    under the sglang scheduler, so the module holds them in a static)."""
    state = _STATE
    assert state is not None
    mod = _jit_module(k, world_size)
    mod.set_bases(state.uc_bases)
    return mod


def fits(x: torch.Tensor) -> bool:
    """Whether this o_proj input can take the fused GEMM+AR path."""
    return (
        _STATE is not None
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and 0 < x.shape[0] <= MAX_TOKENS
        and x.stride(1) == 1
        and x.stride(0) == x.shape[1]
    )


@register_custom_op(mutates_args=["out", "epochs"])
def _gemm_ar_op(
    k: int,
    world_size: int,
    out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    gather: torch.Tensor,
    epochs: torch.Tensor,
    my_rank: int,
) -> None:
    _module_with_bases(k, world_size).run(out, x, weight, gather, epochs, my_rank)


def _cell_of(m: int) -> int:
    for c in (8, 16, 32, 64, 128, 256, 512):
        if m <= c:
            return c
    raise ValueError(f"gemm_ar: M={m} outside [1, {MAX_TOKENS}]")


def o_proj_gemm_ar(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fully reduced ``sum_r x_r @ weight_r^T`` on every rank, one kernel.

    ``x`` is the TP-local [M, K] o_proj input, ``weight`` the TP-local
    [7168, K] o_proj weight shard. Caller checked :func:`fits`; all ranks
    call in lockstep with the same M.
    """
    state = _STATE
    assert state is not None
    m = x.shape[0]
    cell = _cell_of(m)
    out = torch.empty((cell, N), dtype=torch.bfloat16, device=x.device)
    _gemm_ar_op(
        weight.shape[1],
        state.world_size,
        out,
        x,
        weight,
        state.gather,
        state.epochs,
        state.rank,
    )
    return out[:m]
