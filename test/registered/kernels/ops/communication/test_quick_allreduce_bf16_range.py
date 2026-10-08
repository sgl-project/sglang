import multiprocessing
import os
import socket
import time
import unittest

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    QuickAllReduce,
    create_quick_allreduce,
    qr_rocm_arch_available,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-c-test-4-gpu-amd")
register_amd_ci(est_time=30, suite="stage-c-test-large-8-gpu-amd-mi35x")

_RELATIVE_L2_LIMITS = {
    "FP": 0.05,
    "INT8": 0.15,
    "INT6": 0.20,
    "INT4": 0.35,
}


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _run_bf16_range_test(rank: int, world_size: int, port: int) -> None:
    os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1"
    os.environ["AITER_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1"

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
        for backend_name in ("bundled", "aiter"):
            os.environ["SGLANG_USE_AITER"] = str(int(backend_name == "aiter"))
            for quant_mode in ("FP", "INT8", "INT6", "INT4"):
                os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = quant_mode
                os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = (
                    "FP8" if quant_mode == "INT8" else quant_mode
                )
                quick_all_reduce = create_quick_allreduce(
                    group=dist.group.WORLD, device=device
                )
                assert quick_all_reduce is not None
                assert not quick_all_reduce.disabled
                communicator = quick_all_reduce._communicator
                if backend_name == "bundled":
                    assert isinstance(communicator, QuickAllReduce)
                    assert communicator.use_fp16_kernels
                else:
                    assert type(communicator).__module__.startswith("aiter.")
                try:
                    # AITER's optional BF16-to-FP16 cast does not provide the
                    # bundled kernel's overflow handling. Exercise its real
                    # factory/kernel path within FP16 range, while retaining
                    # the full range regression coverage for the bundled
                    # implementation.
                    backend_cases = cases if backend_name == "bundled" else cases[:2]
                    for case_name, value, mixed in backend_cases:
                        inp = torch.full(
                            (numel,), value, dtype=torch.bfloat16, device=device
                        )
                        if mixed:
                            inp[::32] = 80_000.0
                        expected = (inp.float() * world_size).to(torch.bfloat16)

                        dist.barrier()
                        out = quick_all_reduce.quick_all_reduce(inp)
                        torch.cuda.synchronize()
                        context = f"{backend_name=} {quant_mode=} {case_name=}"
                        assert torch.isfinite(out).all().item(), (
                            f"{context} produced non-finite output"
                        )
                        assert torch.count_nonzero(out).item() > 0, (
                            f"{context} produced an all-zero output"
                        )
                        if quant_mode == "FP" or case_name in ("low", "ordinary"):
                            relative_l2 = (
                                torch.linalg.vector_norm(out.float() - expected.float())
                                / torch.linalg.vector_norm(expected.float())
                            ).item()
                            relative_l2_limit = _RELATIVE_L2_LIMITS[quant_mode]
                            assert relative_l2 <= relative_l2_limit, (
                                f"{context} relative L2 error {relative_l2:.6f} "
                                f"exceeds {relative_l2_limit:.6f}"
                            )
                        if backend_name == "bundled" and (
                            quant_mode == "FP" or case_name in ("low", "ordinary")
                        ):
                            torch.testing.assert_close(
                                out,
                                expected,
                                rtol=0,
                                atol=0,
                                msg=lambda msg: f"{context}\n{msg}",
                            )
                finally:
                    quick_all_reduce.close()
    finally:
        dist.destroy_process_group()


class TestQuickAllReduceBf16Range(CustomTestCase):
    @unittest.skipUnless(
        qr_rocm_arch_available() and torch.cuda.device_count() >= 2,
        "QuickReduce range test requires at least two supported ROCm GPUs",
    )
    def test_bf16_range(self):
        for world_size in (2, 4, 8):
            if world_size > torch.cuda.device_count():
                continue
            with self.subTest(world_size=world_size):
                self._run_world_size(world_size)

    def _run_world_size(self, world_size: int):
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
                f"QuickReduce bf16 range test failed for {world_size=} on rank {rank}",
            )


if __name__ == "__main__":
    unittest.main()
