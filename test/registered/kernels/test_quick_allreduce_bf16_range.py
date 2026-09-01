import multiprocessing
import os
import socket
import time
import unittest

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    QuickAllReduce,
    qr_rocm_arch_available,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-c-test-4-gpu-amd")
register_amd_ci(est_time=30, suite="stage-c-test-large-8-gpu-amd-mi35x")


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _run_bf16_range_test(rank: int, world_size: int, port: int) -> None:
    os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1"

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        numel = 1 << 20
        cases = [
            ("low", 2**-8, False),
            ("ordinary", 100.0, False),
            ("sum_above_fp16", 20_000.0, False),
            ("input_above_fp16", 80_000.0, False),
            ("negative_sum_above_fp16", -20_000.0, False),
            ("mixed_input_above_fp16", 1.0, True),
        ]
        for quant_mode in ("FP", "INT8", "INT6", "INT4"):
            os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = quant_mode
            quick_all_reduce = QuickAllReduce(group=dist.group.WORLD, device=device)
            assert not quick_all_reduce.disabled
            assert quick_all_reduce.use_fp16_kernels
            try:
                for case_name, value, mixed in cases:
                    inp = torch.full(
                        (numel,), value, dtype=torch.bfloat16, device=device
                    )
                    if mixed:
                        inp[::32] = 80_000.0
                    expected = (inp.float() * world_size).to(torch.bfloat16)

                    dist.barrier()
                    out = quick_all_reduce.quick_all_reduce(inp)
                    torch.cuda.synchronize()
                    assert (
                        torch.isfinite(out).all().item()
                    ), f"{quant_mode=} {case_name=} produced non-finite output"
                    if quant_mode == "FP" or case_name in ("low", "ordinary"):
                        torch.testing.assert_close(
                            out,
                            expected,
                            rtol=0,
                            atol=0,
                            msg=lambda msg: f"{quant_mode=} {case_name=}\n{msg}",
                        )
            finally:
                quick_all_reduce.close()
    finally:
        dist.destroy_process_group()


class TestQuickAllReduceBf16Range(CustomTestCase):
    @unittest.skipUnless(
        qr_rocm_arch_available() and torch.cuda.device_count() >= 4,
        "QuickReduce range test requires at least four supported ROCm GPUs",
    )
    def test_bf16_range(self):
        world_size = 4
        port = _get_open_port()
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(
                target=_run_bf16_range_test,
                args=(rank, world_size, port),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()

        deadline = time.monotonic() + 120
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))

        if any(process.is_alive() for process in processes):
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            self.fail("QuickReduce bf16 range test timed out")

        for rank, process in enumerate(processes):
            self.assertEqual(
                process.exitcode,
                0,
                f"QuickReduce bf16 range test failed on rank {rank}",
            )


if __name__ == "__main__":
    unittest.main()
