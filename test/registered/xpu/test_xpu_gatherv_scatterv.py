"""2-rank XPU test for `GroupCoordinator.all_gatherv` / `reduce_scatterv` on
the xccl backend. Before the XPU branch existed, both crashed with
`AttributeError: 'NoneType' object has no attribute 'change_state'` because
pynccl_comm is None on XPU.

Spawns 2 XPU workers via `torch.multiprocessing.spawn` and checks:
    1. all_gatherv uneven sizes: values match reference, output not reallocated.
    2. all_gatherv equal + sizes=None fast paths.
    3. all_gatherv list input raises NotImplementedError (CUDA-only path).
    4. reduce_scatterv uneven: reduced chunk correct, output not reallocated.
    5. reduce_scatterv does not mutate input_ (regression for review comment #3).
    6. reduce_scatterv equal chunks.
"""

import unittest
import torch

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    update_environment_variables,
)
from sglang.srt.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=120, suite="stage-a-test-2-gpu-xpu")

NUM_XPUS = 2
H = 8  # hidden width of the fake token rows


def _block(r: int, nrows: int, device) -> torch.Tensor:
    """Rank-tagged [nrows, H] block: row i of rank r holds (r+1)*1000 + i, so any
    misplaced/uninitialized row is detectable in a reference comparison."""
    base = torch.arange(nrows, device=device, dtype=torch.float32).reshape(-1, 1)
    return (base + (r + 1) * 1000.0).repeat(1, H)


@unittest.skipIf(
    not is_xpu() or torch.xpu.device_count() < NUM_XPUS,
    f"This test requires at least {NUM_XPUS} XPU devices",
)
class TestXpuGatherv(CustomTestCase):
    def test_two_rank(self):
        """Spawn NUM_XPUS workers running the gatherv/reduce_scatterv suite."""
        torch.multiprocessing.spawn(_worker_main, args=(NUM_XPUS,), nprocs=NUM_XPUS)


def _worker_main(local_rank: int, world_size: int):
    device = torch.device(f"xpu:{local_rank}")
    torch.xpu.set_device(device)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12361",  # Distinct from other tests' ports.
        }
    )
    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        backend="xccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    tp = get_tp_group()
    rank = tp.rank_in_group

    # ---- all_gatherv: uneven sizes into a pre-allocated output buffer ----
    sizes = [3, 5] if world_size == 2 else [world_size - i for i in range(world_size)]
    total = sum(sizes)
    my_in = _block(rank, sizes[rank], device)
    ref = torch.cat([_block(r, sizes[r], device) for r in range(world_size)], dim=0)
    out_buf = torch.empty((total, H), device=device, dtype=torch.float32)
    ptr_before = out_buf.data_ptr()
    res = tp.all_gatherv(my_in, sizes=sizes, output=out_buf)
    torch.xpu.synchronize()
    assert torch.equal(out_buf, ref), "all_gatherv uneven: gathered values != reference"
    assert (
        res[0].data_ptr() == ptr_before
    ), "all_gatherv must write into the pre-allocated output (no realloc)"

    # ---- all_gatherv: equal sizes + sizes=None (single-collective fast path) ----
    eq = 4
    my_eq = _block(rank, eq, device)
    ref_eq = torch.cat([_block(r, eq, device) for r in range(world_size)], dim=0)
    assert torch.equal(tp.all_gatherv(my_eq, sizes=[eq] * world_size)[0], ref_eq)
    assert torch.equal(tp.all_gatherv(my_eq, sizes=None)[0], ref_eq)

    # ---- all_gatherv: list input is CUDA/flashinfer-only, must fail loud on XPU ----
    raised = False
    try:
        tp.all_gatherv([my_eq, my_eq], sizes=[eq] * world_size)
    except NotImplementedError:
        raised = True
    assert raised, "all_gatherv list input on XPU should raise NotImplementedError"

    # ---- reduce_scatterv uneven: every rank contributes (rank+1); reduced
    # chunk should equal world*(world+1)/2 everywhere. ----
    exp_val = float(world_size * (world_size + 1) // 2)
    rs_in = torch.full((total, H), float(rank + 1), device=device, dtype=torch.float32)
    rs_out = torch.empty((sizes[rank], H), device=device, dtype=torch.float32)
    rs_ptr = rs_out.data_ptr()
    r = tp.reduce_scatterv(rs_in, output=rs_out, sizes=sizes)
    torch.xpu.synchronize()
    ok = torch.equal(rs_out, torch.full((sizes[rank], H), exp_val, device=device))
    flag = torch.tensor([1 if ok else 0], device=device)
    torch.distributed.all_reduce(flag)
    assert int(flag.item()) == world_size, "reduce_scatterv uneven: wrong reduced chunk"
    assert r.data_ptr() == rs_ptr, "reduce_scatterv must write into pre-allocated output"

    # ---- reduce_scatterv: verify input_ is not mutated by the call. ----
    mut_in = torch.arange(
        total * H, dtype=torch.float32, device=device
    ).reshape(total, H) + float(rank + 1) * 10_000.0
    mut_out = torch.empty((sizes[rank], H), device=device, dtype=torch.float32)
    snapshot = mut_in.clone()
    tp.reduce_scatterv(mut_in, output=mut_out, sizes=sizes)
    torch.xpu.synchronize()
    unchanged = torch.equal(mut_in, snapshot)
    flag_mut = torch.tensor([1 if unchanged else 0], device=device)
    torch.distributed.all_reduce(flag_mut)
    assert int(flag_mut.item()) == world_size, (
        f"reduce_scatterv mutated input_ on rank {rank} — "
        f"max|diff|={(mut_in - snapshot).abs().max().item()}"
    )

    # ---- reduce_scatterv: equal chunks (single native reduce_scatter_tensor) ----
    eqrs_in = torch.full(
        (eq * world_size, H), float(rank + 1), device=device, dtype=torch.float32
    )
    r2 = tp.reduce_scatterv(eqrs_in, sizes=[eq] * world_size)
    torch.xpu.synchronize()
    ok2 = torch.equal(r2, torch.full((eq, H), exp_val, device=device))
    flag2 = torch.tensor([1 if ok2 else 0], device=device)
    torch.distributed.all_reduce(flag2)
    assert int(flag2.item()) == world_size, "reduce_scatterv equal: wrong reduced chunk"

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
