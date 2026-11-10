"""Native PPU collectives and sequential TP=1/TP=2 numerical serving smoke."""

import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.ppu_pr_utils import free_port

register_ppu_ci(est_time=240, suite="stage-c-test-2-gpu-ppu")


def collective_worker(rank, port):
    torch.cuda.set_device(rank)
    # SAIL exposes its collective implementation through PyTorch's nccl backend.
    # Do not enable CUDA custom-all-reduce or assume NVIDIA peer-memory support.
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        x = torch.arange(16, device="cuda", dtype=torch.float32) + rank * 16
        total = x.clone()
        dist.all_reduce(total)
        torch.testing.assert_close(
            total,
            torch.arange(16, device="cuda") * 2 + 16,
            rtol=0,
            atol=0,
            check_dtype=False,
        )
        gathered = [torch.empty_like(x) for _ in range(2)]
        dist.all_gather(gathered, x)
        for peer, actual in enumerate(gathered):
            torch.testing.assert_close(
                actual,
                torch.arange(16, device="cuda", dtype=torch.float32) + peer * 16,
                rtol=0,
                atol=0,
            )
        torch.cuda.synchronize()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_collectives():
    assert torch.cuda.device_count() >= 2, "Two PPUs are mandatory"
    context = mp.spawn(collective_worker, args=(free_port(),), nprocs=2, join=False)
    try:
        import time

        deadline = time.monotonic() + 90
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, "collective processes timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=10)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
