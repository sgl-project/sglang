import multiprocessing
import os
import socket
import time
import unittest

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    QuickAllReduce,
    QuickReduceRegime,
    qr_rocm_arch_available,
)
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-c-test-4-gpu-amd")
register_amd_ci(est_time=120, suite="stage-c-test-large-8-gpu-amd-mi35x")


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
                    assert torch.isfinite(out).all().item(), (
                        f"{quant_mode=} {case_name=} produced non-finite output"
                    )
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


_INT8 = QuickReduceRegime.INT8.value
_INT6 = QuickReduceRegime.INT6.value
_INT4 = QuickReduceRegime.INT4.value


def _open_comms(rank: int, world_size: int) -> tuple[int, int]:
    import sgl_kernel.allreduce as aot

    from sglang.kernels.ops.communication import quick_all_reduce_hip as jit

    def open_comm(init, get_handle, open_handles):
        handle = init(rank, world_size, 128 << 20)
        handles = [None] * world_size
        dist.all_gather_object(handles, get_handle(handle))
        open_handles(handle, handles)
        return handle

    # sgl_kernel.allreduce takes (world_size, rank), the JIT module the reverse.
    aot_comm = open_comm(
        lambda r, w, m: aot.init_custom_qr(w, r, m),
        aot.qr_get_handle,
        aot.qr_open_handles,
    )
    jit_comm = open_comm(jit.init_custom_qr, jit.qr_get_handle, jit.qr_open_handles)
    return aot_comm, jit_comm


def _run_jit_bitwise_test(rank: int, world_size: int, port: int) -> None:
    import sgl_kernel.allreduce as aot

    from sglang.kernels.ops.communication import quick_all_reduce_hip as jit

    os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = "INT4"
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
        aot_comm, jit_comm = _open_comms(rank, world_size)
        generator = torch.Generator(device=device).manual_seed(rank)
        # (requested level, level it runs): asked for INT4, the JIT build runs INT8 below 64 MiB;
        # at 32 KB a tile, the sizes past the first have more tiles than blocks (grid-stride loop)
        for numel, level, runs in (
            (384 * 6144, _INT4, _INT8),
            (4096 * 6144, _INT4, _INT8),
            (6144 * 6144, _INT4, _INT4),
            (6144 * 6144, _INT8, _INT8),
        ):
            inp = torch.randn(numel, device=device, generator=generator).bfloat16()
            inp[::4099] *= 64
            expected, out = torch.empty_like(inp), torch.empty_like(inp)
            for _ in range(2):
                aot.qr_all_reduce(aot_comm, inp, expected, runs, True)
                jit.qr_all_reduce(jit_comm, inp, out, level, True)
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), (
                f"{numel=} {level=}: JIT quick all-reduce differs from sgl-kernel's"
            )
        try:
            jit.qr_all_reduce(jit_comm, inp, out, _INT6, True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("the INT4 / INT8 JIT build ran INT6")
        jit.qr_destroy(jit_comm)
        aot.qr_destroy(aot_comm)
    finally:
        dist.destroy_process_group()


def _run_jit_fallback_test(rank: int, world_size: int, port: int) -> None:
    import sgl_kernel.allreduce as aot

    from sglang.kernels.ops.communication import quick_all_reduce_hip as jit

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        generator = torch.Generator(device=device).manual_seed(rank)
        inp = torch.randn(384 * 6144, device=device, generator=generator).bfloat16()
        # (level, cast) the JIT build does not serve: INT6, INT4 at TP2, no cast at TP4
        cases = [("INT6", "1", _INT6, True)]
        if world_size == 4:
            cases.append(("INT8", "0", _INT8, False))
        else:
            cases.append(("INT4", "1", _INT4, True))
        for level_name, cast, level, cast_bf2half in cases:
            os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = level_name
            os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = cast
            aot_comm, jit_comm = _open_comms(rank, world_size)
            expected, out = torch.empty_like(inp), torch.empty_like(inp)
            aot.qr_all_reduce(aot_comm, inp, expected, level, cast_bf2half)
            jit.qr_all_reduce(jit_comm, inp, out, level, cast_bf2half)
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), (
                f"{world_size=} {level_name=} {cast=}: the fallback is not sgl-kernel's"
            )
            jit.qr_destroy(jit_comm)
            aot.qr_destroy(aot_comm)
    finally:
        dist.destroy_process_group()


def _spawn(test_case, target, world_size, name):
    port = _get_open_port()
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=target, args=(rank, world_size, port))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 300
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    if any(process.is_alive() for process in processes):
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        test_case.fail(f"{name} timed out")
    for rank, process in enumerate(processes):
        test_case.assertEqual(process.exitcode, 0, f"{name} failed on rank {rank}")


class TestQuickAllReduceBf16Range(CustomTestCase):
    @unittest.skipUnless(
        qr_rocm_arch_available() and torch.cuda.device_count() >= 4,
        "QuickReduce range test requires at least four supported ROCm GPUs",
    )
    def test_bf16_range(self):
        _spawn(self, _run_bf16_range_test, 4, "QuickReduce bf16 range test")

    @unittest.skipUnless(
        is_gfx95_supported() and torch.cuda.device_count() >= 4,
        "the JIT quick all-reduce is built and validated on four or more gfx950 GPUs",
    )
    def test_jit_matches_sgl_kernel(self):
        """On bf16 cast to fp16 the JIT build must reduce bit-identically to sgl-kernel's at
        the level it runs (INT8 below 64 MiB when asked for INT4) and refuse any level but
        INT4 and INT8."""
        _spawn(self, _run_jit_bitwise_test, 4, "QuickReduce JIT bitwise test")

    @unittest.skipUnless(
        is_gfx95_supported() and torch.cuda.device_count() >= 4,
        "the JIT quick all-reduce is built and validated on four or more gfx950 GPUs",
    )
    def test_jit_falls_back_to_sgl_kernel(self):
        """The JIT module stands in for every group's quick all-reduce, so a group it does not
        serve (TP2, INT6, no bf16 -> fp16 cast) must run sgl-kernel's instead of raising."""
        for world_size in (2, 4):
            _spawn(
                self,
                _run_jit_fallback_test,
                world_size,
                f"QuickReduce JIT fallback test (TP{world_size})",
            )


if __name__ == "__main__":
    unittest.main()
