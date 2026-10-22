"""Native PPU collectives and sequential TP=1/TP=2 numerical serving smoke."""

import math
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.ppu_pr_utils import free_port, generate, prompt_ids, server

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


def logprobs(result):
    values = result["meta_info"]["input_token_logprobs"]
    pairs = [(item[1], item[0]) for item in values if item[0] is not None]
    assert len(pairs) >= 16
    assert all(math.isfinite(value) for _, value in pairs)
    return pairs


def test_tp2_against_tp1():
    ids = prompt_ids(128)
    with server(tp=1) as base:
        reference = logprobs(
            generate(base, ids, return_logprob=True, logprob_start_len=0)
        )
    # No overlapping model/server lifetimes: TP=2 gets both devices.
    with server(tp=2) as base:
        single = generate(base, ids, return_logprob=True, logprob_start_len=0)
        actual = logprobs(single)
        assert [token for token, _ in reference] == [token for token, _ in actual]
        # BF16 reductions may change low bits. Compare probabilities, not decoded text.
        torch.testing.assert_close(
            torch.tensor([value for _, value in actual]),
            torch.tensor([value for _, value in reference]),
            atol=0.15,
            rtol=0.02,
        )
        batch = generate(base, [ids] * 4)
        assert len(batch) == 4
        for item in [single, *batch]:
            assert isinstance(item["text"], str)
            assert 0 < item["meta_info"]["completion_tokens"] <= 16


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
