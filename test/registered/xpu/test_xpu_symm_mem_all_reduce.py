"""Correctness test for the Intel XPU symmetric-memory one-shot all-reduce.

The sweep lives in one test function on purpose: every rank must issue the same
sequence of collectives, and a shuffled test order (pytest-randomly seeds per
process) would pair ranks on different payloads.

Usage::

    # Runs itself under torchrun on 2 XPUs:
    python test/registered/xpu/test_xpu_symm_mem_all_reduce.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.xpu_symm_mem import (
    XpuSymmMemCommunicator,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import is_in_ci

register_xpu_ci(est_time=180, suite="nightly-xpu-kernel-main-2-gpu", nightly=True)

_WORLD_SIZE = 2
_MAX_BYTES = 512 * 1024
_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# 128 exercises a masked block tail; 131072 fp32 fills the 512 KiB buffer.
_NUMELS = [128, 3584, 131072]

_comm_cache = {}


def _get_comm() -> XpuSymmMemCommunicator:
    """One communicator per worker; rendezvous is collective, so cache it."""
    if "comm" not in _comm_cache:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.xpu.set_device(local_rank)
        dist.init_process_group(backend="gloo")
        _comm_cache["comm"] = XpuSymmMemCommunicator(
            group=dist.group.WORLD, device=torch.device(f"xpu:{local_rank}")
        )
    comm = _comm_cache["comm"]
    if comm.disabled:
        # The CI container grants SYS_PTRACE for rendezvous, so a disabled
        # communicator there is a broken setup rather than a missing feature.
        if is_in_ci():
            pytest.fail("XPU symmetric memory unavailable in CI")
        pytest.skip("XPU symmetric memory unavailable")
    return comm


def _shards(numel: int, dtype: torch.dtype, world_size: int) -> list[torch.Tensor]:
    """Per-rank inputs, seeded so every rank can build the reference locally."""
    return [
        torch.randn(
            numel,
            generator=torch.Generator().manual_seed(1234 + rank),
            dtype=torch.float32,
        ).to(dtype)
        for rank in range(world_size)
    ]


def _reference(shards: list[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    """Accumulate in fp32 in ascending rank order, exactly as the kernel does."""
    acc = shards[0].float()
    for shard in shards[1:]:
        acc = acc + shard.float()
    return acc.to(dtype)


@torch.inference_mode()
def test_one_shot_all_reduce_matches_ordered_fp32_sum() -> None:
    comm = _get_comm()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    for dtype in _DTYPES:
        for numel in _NUMELS:
            if numel * torch.finfo(dtype).bits // 8 > comm.max_bytes:
                continue
            shards = _shards(numel, dtype, world_size)
            expected = _reference(shards, dtype)
            inp = shards[rank].to(comm.device)

            out = comm.all_reduce(inp)
            assert out is not None, f"{dtype}/{numel} rejected by the fast path"
            torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)

            # In-place form, as GroupCoordinator._all_reduce_in_place calls it.
            inplace = inp.clone()
            assert comm.all_reduce(inplace, out=inplace) is not None
            torch.testing.assert_close(inplace.cpu(), expected, atol=0, rtol=0)


@torch.inference_mode()
def test_ineligible_payloads_fall_back() -> None:
    """Rejected inputs must return None so the caller falls back."""
    comm = _get_comm()

    max_bf16 = comm.max_bytes // 2
    too_large = torch.zeros(max_bf16 + 1, dtype=torch.bfloat16, device=comm.device)
    assert not comm.should_torch_symm_mem_allreduce(too_large)
    assert comm.all_reduce(too_large) is None

    exactly_max = torch.zeros(max_bf16, dtype=torch.bfloat16, device=comm.device)
    assert comm.should_torch_symm_mem_allreduce(exactly_max)

    unsupported_dtype = torch.zeros(128, dtype=torch.float64, device=comm.device)
    assert not comm.should_torch_symm_mem_allreduce(unsupported_dtype)

    non_contiguous = torch.zeros(128, 8, dtype=torch.bfloat16, device=comm.device)[
        :, ::2
    ]
    assert not comm.should_torch_symm_mem_allreduce(non_contiguous)

    on_cpu = torch.zeros(128, dtype=torch.bfloat16)
    assert not comm.should_torch_symm_mem_allreduce(on_cpu)


@torch.inference_mode()
def test_staging_is_not_overwritten_before_a_slow_peer_reads_it() -> None:
    """With one barrier per call, a peer that lags between the barrier and its
    read must still see this call's inputs, not the next call's."""
    comm = _get_comm()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    calls, numel = 8, 4096
    lag = torch.randn(4096, 4096, device=comm.device)
    workspace_barrier = comm.workspace.barrier

    def barrier_then_lag() -> None:
        workspace_barrier()
        lag @ lag  # queued after the barrier, before this rank's peer reads

    inp = torch.empty(numel, dtype=torch.float32, device=comm.device)
    results = torch.empty(calls, numel, dtype=torch.float32, device=comm.device)
    with patch.object(
        comm.workspace,
        "barrier",
        barrier_then_lag if rank == 0 else workspace_barrier,
    ):
        for call in range(calls):
            inp.fill_(float(call * world_size + rank))
            assert comm.all_reduce(inp, out=results[call]) is not None
    expected = torch.tensor(
        [
            float(sum(call * world_size + r for r in range(world_size)))
            for call in range(calls)
        ]
    )
    torch.testing.assert_close(
        results.cpu(), expected[:, None].expand(calls, numel), atol=0, rtol=0
    )


@torch.inference_mode()
def test_group_coordinator_takes_one_shot_only_under_graph_capture() -> None:
    """GroupCoordinator.all_reduce keeps eager calls on oneCCL and routes a
    captured eligible call to the one-shot path."""
    comm = _get_comm()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    group = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=int(os.environ["LOCAL_RANK"]),
        torch_distributed_backend="xccl",
        use_pynccl=False,
        use_mscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=True,
        use_hpu_communicator=False,
        use_xpu_communicator=True,
        use_npu_communicator=False,
        group_name="xpu_symm_mem_test",
    )
    one_shot = group.xpu_symm_mem_comm
    assert one_shot is not None and not one_shot.disabled
    taken = []
    one_shot_all_reduce = one_shot.all_reduce

    def record(inp, *, out=None):
        taken.append(inp.numel())
        return one_shot_all_reduce(inp, out=out)

    x = torch.full((4096,), float(rank + 1), dtype=torch.bfloat16, device=comm.device)
    expected = float(world_size * (world_size + 1) // 2)
    with patch.object(one_shot, "all_reduce", record):
        group.all_reduce(x)
        torch.xpu.synchronize()
        assert taken == []
        assert bool((x == expected).all())

        x.fill_(float(rank + 1))
        graph, stream = torch.xpu.XPUGraph(), torch.xpu.Stream()
        stream.wait_stream(torch.xpu.current_stream())
        with torch.xpu.graph(xpu_graph=graph, stream=stream):
            group.all_reduce(x)
        assert taken == [4096]
        graph.replay()
        torch.xpu.synchronize()
        assert bool((x == expected).all())


def _main() -> int:
    if torch.xpu.device_count() < _WORLD_SIZE:
        print(f"needs {_WORLD_SIZE} XPUs, found {torch.xpu.device_count()}; skipping")
        # A CI runner without the cards is a broken runner, not a skip.
        return 1 if is_in_ci() else 0
    with envs.SGLANG_XPU_SYMM_MEM_MAX_BYTES.override(_MAX_BYTES):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                # A free port: a fixed default collides with other jobs' torchrun.
                "--standalone",
                f"--nproc-per-node={_WORLD_SIZE}",
                __file__,
                *sys.argv[1:],
            ],
            env={**os.environ, "_XPU_SYMM_MEM_TEST_WORKER": "1"},
        ).returncode


if __name__ == "__main__":
    if os.environ.get("_XPU_SYMM_MEM_TEST_WORKER"):
        args = ["-x" if a == "-f" else a for a in sys.argv[1:]]
        # no:randomly — a per-process shuffle would put ranks on different
        # collectives; the tests below assume a fixed order across ranks.
        sys.exit(pytest.main([__file__, "-p", "no:randomly", *args]))
    sys.exit(_main())
