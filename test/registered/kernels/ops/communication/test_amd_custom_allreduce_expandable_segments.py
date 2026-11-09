"""SGLang's ROCm custom all-reduce with `expandable_segments:True`.

The caching allocator then hands out VMM-backed memory, which
hipIpcGetMemHandle cannot export: the staging buffer has to come from regular
segments, and graph-captured inputs have to be copied in instead of registered.

Usage:
    python3 test_amd_custom_allreduce_expandable_segments.py
"""

import multiprocessing as mp
import socket
import unittest

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=90, stage="sgl-kernel-unit", runner_config="2-gpu-amd")

WORLD_SIZE = 2
NUMEL = 64 * 1024
# A staging buffer smaller than the input, so the deterministic path reduces it
# in pieces; 2.5 buffers of fp32 input also leaves a partial piece at the end.
SMALL_MAX_SIZE = 256 * 1024
OVERSIZED_NUMEL = SMALL_MAX_SIZE // 4 * 5 // 2


def _get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, port, deterministic, numel, max_size):
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    envs.SGLANG_USE_1STAGE_ALLREDUCE.set("1" if deterministic else "0")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )

    from sglang.srt.distributed.device_communicators import custom_all_reduce

    def expandable_segments():
        return custom_all_reduce._allocator_settings()["expandable_segments"]

    assert expandable_segments()
    ar_group = dist.new_group(backend="gloo")
    kwargs = {} if max_size is None else {"max_size": max_size}
    custom_ar = custom_all_reduce.CustomAllreduce(
        group=ar_group, device=device, **kwargs
    )
    assert not custom_ar.disabled, "custom all-reduce is disabled on this topology"
    assert custom_ar.use_amd_deterministic_impl == deterministic
    assert expandable_segments(), "allocator setting not restored"

    # Small integers, exact in fp32 for any summation order, repeating with a
    # period that does not divide the piece size, so a misplaced piece shows.
    pattern = (torch.arange(numel, device=device, dtype=torch.float32) % 1021) + 1
    expected = pattern * (WORLD_SIZE * (WORLD_SIZE + 1) // 2)

    def check(result, what):
        assert result is not None, f"rank {rank}: custom all-reduce declined {what}"
        torch.cuda.synchronize()
        wrong = int((result != expected).sum())
        assert wrong == 0, f"rank {rank}: {what} is wrong in {wrong} places"

    inp = pattern * (rank + 1)
    check(custom_ar.custom_all_reduce(inp), "eager all-reduce")

    static_inp = torch.zeros_like(inp)
    graph = torch.cuda.CUDAGraph()
    with custom_ar.capture():
        with torch.cuda.graph(graph):
            static_out = custom_ar.custom_all_reduce(static_inp)
    assert static_out is not None, f"rank {rank}: custom all-reduce declined capture"
    # Replay twice so the graph is shown to read the input in place.
    for scale in (1, 2):
        static_inp.copy_(inp * scale)
        graph.replay()
        check(static_out / scale, f"graph replay with scale {scale}")

    dist.barrier()
    custom_ar.close()
    dist.destroy_process_group()


def _run_workers(deterministic, numel=NUMEL, max_size=None):
    ctx = mp.get_context("spawn")
    port = _get_open_port()
    procs = [
        ctx.Process(target=_worker, args=(rank, port, deterministic, numel, max_size))
        for rank in range(WORLD_SIZE)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    failed = {rank: p.exitcode for rank, p in enumerate(procs) if p.exitcode != 0}
    if failed:
        raise AssertionError(f"worker processes failed, exit codes by rank: {failed}")


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    f"requires {WORLD_SIZE} GPUs",
)
class TestCustomAllreduceExpandableSegments(CustomTestCase):
    def test_deterministic(self):
        _run_workers(deterministic=True)

    def test_deterministic_input_larger_than_the_staging_buffer(self):
        _run_workers(deterministic=True, numel=OVERSIZED_NUMEL, max_size=SMALL_MAX_SIZE)

    def test_non_deterministic(self):
        _run_workers(deterministic=False)


if __name__ == "__main__":
    unittest.main()
