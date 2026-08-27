from __future__ import annotations

import atexit
import os
from typing import Dict, Tuple

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import multigpu_bench_main
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import nvlink_comm as nvl
from sglang.srt.distributed.parallel_state import GroupCoordinator

DTYPE = torch.bfloat16
HIDDEN = 7168
PUSH_SLOT_MB = 32
PULL_MB = 4  # nvlink pull borrows only the semaphores, never this buffer
# Narrow the sweep while iterating, e.g. BENCH_PROVIDERS=nvlink-pull,nccl
PROVIDERS = [
    "nccl",
    "nccl-symm",
    "v2",
    "nvlink-push",
    "nvlink-pull",
]
OPS = nvl.SUPPORTED_OPS


@cache_once
def _init_groups():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    torch.cuda.set_stream(torch.cuda.Stream())
    device = torch.device(f"cuda:{local_rank}")
    gpu_group = dist.new_group(backend="nccl", device_id=device)
    # A second coordinator, this one with pynccl, so NCCL symmetric memory has a
    # communicator to register its windows against.
    pynccl_coord = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=local_rank,
        torch_distributed_backend="nccl",
        use_pynccl=True,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="nvlink_bench_pynccl",
    )
    # This file owns the process group, so it owns tearing it down; NCCL warns
    # on exit otherwise.
    atexit.register(lambda: dist.is_initialized() and dist.destroy_process_group())
    return coord.cpu_group, gpu_group, pynccl_coord, device


@cache_once
def _init_comms():
    """Two communicators: one for the nvlink planes, one as the v2 baseline."""
    from sglang.kernels.ops.communication.mp import register_comm_cleanup
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    cpu_group, _, _, device = _init_groups()
    # The nvlink pull path needs the plane's semaphores, so the pull half has to
    # exist even though its buffer goes unused.
    nvlink = CustomAllReduceV2(
        cpu_group,
        device,
        max_push_size=PUSH_SLOT_MB << 20,
        max_pull_size=PULL_MB << 20,
    )
    v2 = CustomAllReduceV2(cpu_group, device, max_size=PUSH_SLOT_MB << 20)
    for c in (nvlink, v2):
        assert not c.disabled, "CustomAllReduceV2 is disabled on this system"
        register_comm_cleanup(c)
    assert nvlink.has_multicast, "the nvlink collectives need a multicast plane"
    return nvlink, v2


# `empty_strided_p2p` is collective, is not cached, and its allocations are
# never returned, so the sweep reuses one buffer per shape instead of minting a
# fresh pair for every (op, size, provider) cell.
_SYMM_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}


def _symm(shape: Tuple[int, int], group) -> torch.Tensor:
    """A tensor whose allocation carries a multicast alias."""
    from torch._C._distributed_c10d import _SymmetricMemory

    cached = _SYMM_CACHE.get(shape)
    if cached is not None:
        return cached
    if torch.__version__ < "2.11.0":
        import torch.distributed._symmetric_memory as torch_symm_mem

        torch_symm_mem.enable_symm_mem_for_group(group.group_name)
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    t = _SymmetricMemory.empty_strided_p2p(
        (shape[0] * shape[1],), [1], DTYPE, device, group.group_name
    )
    _SymmetricMemory.rendezvous(t)
    _SYMM_CACHE[shape] = view = t.view(shape)
    return view


def _nccl_pool_tensor(shape: Tuple[int, ...], pynccl_coord) -> torch.Tensor:
    """A tensor NCCL can serve from its symmetric-memory fast path."""
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        SymmetricMemoryContext,
    )

    with SymmetricMemoryContext(pynccl_coord):
        return torch.zeros(
            shape, dtype=DTYPE, device=f"cuda:{os.environ['LOCAL_RANK']}"
        )


def _shapes(op: str, tokens: int, world_size: int) -> Tuple[Tuple[int, int], ...]:
    if op == "all_gather":
        return (tokens, HIDDEN), (tokens * world_size, HIDDEN)
    if op == "reduce_scatter":
        return (tokens * world_size, HIDDEN), (tokens, HIDDEN)
    return (tokens, HIDDEN), (tokens, HIDDEN)


@marker.parametrize("residual", [False, True])
@marker.parametrize("op", OPS)
@marker.parametrize("tokens", [2**n for n in range(15)])
@marker.benchmark("provider", PROVIDERS)
def benchmark(op: str, tokens: int, residual: bool, provider: str):
    cpu_group, gpu_group, pynccl_coord, device = _init_groups()
    world_size = dist.get_world_size(cpu_group)
    nvlink, v2 = _init_comms()

    in_shape, out_shape = _shapes(op, tokens, world_size)
    in_bytes = in_shape[0] * in_shape[1] * DTYPE.itemsize
    out_bytes = out_shape[0] * out_shape[1] * DTYPE.itemsize

    if provider == "v2" and op != "all_reduce":
        marker.skip("v2 is an all-reduce baseline only")
    if provider.startswith("nvlink-push"):
        # Only the gather spans the plane; all-reduce and the scatter each put a
        # sender's whole contribution into a single slot.
        cap = nvlink.max_push_size * (world_size if op == "all_gather" else 1)
        if out_bytes > cap:
            marker.skip(f"{out_bytes} B exceeds the {cap} B push plane")
    if provider == "v2" and in_bytes > v2.max_size:
        marker.skip(f"{in_bytes} B exceeds the v2 workspace")

    sym_in = _symm(in_shape, cpu_group)
    sym_out = _symm(out_shape, cpu_group)
    sym_in.normal_()

    # The nvlink kernels fold the residual into the reduction; every other
    # provider has to pay for a separate pass, which is what this row compares.
    # The gather's residual lands on this rank's shard before it is broadcast,
    # so it is the input that gets the extra pass there, not the output.
    res = None
    if residual:
        res_shape = in_shape if op == "all_gather" else out_shape
        res = torch.randn(res_shape, dtype=DTYPE, device=device)

    def unfused(collective, target):
        """The collective plus the separate residual pass it cannot absorb."""
        if res is None:
            return collective
        if op == "all_gather":
            return lambda: (target.add_(res), collective())
        return lambda: (collective(), target.add_(res))

    if provider == "nccl":
        plain_in = torch.randn(in_shape, dtype=DTYPE, device=device)
        plain_out = torch.empty(out_shape, dtype=DTYPE, device=device)
        if op == "all_reduce":
            fn = unfused(lambda: dist.all_reduce(plain_in, group=gpu_group), plain_in)
        elif op == "all_gather":
            fn = unfused(
                lambda: dist.all_gather_single(plain_out, plain_in, group=gpu_group),
                plain_in,
            )
        else:
            fn = unfused(
                lambda: dist.reduce_scatter_single(
                    plain_out, plain_in, group=gpu_group
                ),
                plain_out,
            )
    elif provider == "nccl-symm":
        pool_in = _nccl_pool_tensor(in_shape, pynccl_coord)
        pool_out = _nccl_pool_tensor(out_shape, pynccl_coord)
        comm = pynccl_coord.pynccl_comm
        assert comm is not None and comm.available, "pynccl is unavailable"
        # The communicator ships disabled; enable it for the whole measurement.
        ctx = comm.change_state(enable=True)
        ctx.__enter__()
        # All three run in place on registered buffers, so this stays a
        # zero-copy baseline, matching nvlink-pull.
        if op == "all_reduce":
            fn = unfused(lambda: comm.all_reduce(pool_in), pool_in)
        elif op == "all_gather":
            fn = unfused(lambda: comm.all_gather(pool_out, pool_in), pool_in)
        else:
            fn = unfused(lambda: comm.reduce_scatter(pool_out, pool_in), pool_out)
    elif provider == "v2":
        plain_in = torch.randn(in_shape, dtype=DTYPE, device=device)
        fn = unfused(lambda: v2.custom_all_reduce(plain_in), plain_in)
    else:
        op_fn = {
            ("all_reduce", "push"): nvl.all_reduce_push,
            ("all_gather", "push"): nvl.all_gather_push,
            ("reduce_scatter", "push"): nvl.reduce_scatter_push,
            ("all_reduce", "pull"): nvl.all_reduce_pull,
            ("all_gather", "pull"): nvl.all_gather_pull,
            ("reduce_scatter", "pull"): nvl.reduce_scatter_pull,
        }[(op, "push" if provider.startswith("nvlink-push") else "pull")]
        fn = lambda: op_fn(nvlink.obj, sym_in, sym_out, res)

    # Bandwidth-equivalent bytes a ring collective moves per rank.
    payload = max(in_bytes, out_bytes)
    factor = (
        2 * (world_size - 1) / world_size
        if op == "all_reduce"
        else (world_size - 1) / world_size
    )

    return marker.do_bench(
        fn,
        # The symmetric buffers must keep their addresses, so nothing is cloned
        # between replays; the L2 effect that cloning avoids is dwarfed by the
        # cross-GPU traffic here anyway.
        graph_clone_args=None,
        graph_clone_kwargs=None,
        sync_multigpu_fn=lambda: dist.barrier(gpu_group),  # type: ignore
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=int(payload * factor),
    )


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=[2, 4, 8],
        main_fn=benchmark.run,
    )
