"""INT4 startup reserves HIP IPC, and FlyDSL IPC beside it, without a model.

Two ranks construct QuickAllReduce for real. With AITER on, gfx950 creates
the HIP workspace and the FlyDSL workspace. With AITER off, the same ranks
create only the HIP workspace. A bf16 payload is reduced through FlyDSL and
an fp16 payload through HIP. Driver free memory is read with
torch.cuda.mem_get_info, which counts HIP allocations outside the PyTorch
caching allocator.
"""

import multiprocessing
import os
import socket
import time
import traceback
import unittest

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    QuickAllReduce,
    _flydsl_int4_available,
    qr_rocm_arch_available,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=300, suite="stage-c-test-4-gpu-amd")
register_amd_ci(est_time=300, suite="stage-c-test-large-8-gpu-amd-mi35x")

_MIB = 1024 * 1024


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _vram_used(device: int) -> int:
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return int(total_bytes - free_bytes)


def _workspace_report(quick_reduce: QuickAllReduce, before: int, after: int) -> dict:
    flydsl = quick_reduce._flydsl_int4
    flydsl_bytes = 0
    if flydsl is not None:
        flydsl_bytes = sum(int(eng.buf_bytes) for eng in flydsl._by_st.values())
    return {
        "has_flydsl": flydsl is not None,
        "ptr": int(quick_reduce._ptr or 0),
        "disabled": bool(quick_reduce.disabled),
        "qr_max_size": int(quick_reduce.qr_max_size),
        "flydsl_bytes": flydsl_bytes,
        "before": before,
        "after": after,
        "delta": after - before,
    }


def _one_workspace(rank: int, device: torch.device) -> dict:
    before = _vram_used(rank)
    quick_reduce = QuickAllReduce(group=dist.group.WORLD, device=device)
    after = _vram_used(rank)
    report = _workspace_report(quick_reduce, before, after)
    bf16 = torch.ones(1 << 20, dtype=torch.bfloat16, device=device)
    fp16 = torch.ones(1 << 20, dtype=torch.float16, device=device)
    bf16_out = quick_reduce.quick_all_reduce(bf16)
    fp16_out = quick_reduce.quick_all_reduce(fp16)
    torch.cuda.synchronize(device)
    report["bf16_finite"] = bool(torch.isfinite(bf16_out).all().item())
    report["fp16_finite"] = bool(torch.isfinite(fp16_out).all().item())
    # Both ranks must drop their peer mappings before the local free shows up.
    dist.barrier()
    quick_reduce.close()
    dist.barrier()
    report["released_to"] = _vram_used(rank)
    return report


def _run_ipc_switch(rank: int, world_size: int, port: int, queue) -> None:
    os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = "INT4"
    os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1"
    os.environ.pop("ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", None)
    os.environ["SGLANG_USE_AITER"] = "1"

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.empty(1024, device=device)
        flydsl_can_run = _flydsl_int4_available()
        reports = {"rank": rank, "flydsl_available": flydsl_can_run}
        if flydsl_can_run:
            reports["flydsl"] = _one_workspace(rank, device)
            os.environ.pop("SGLANG_USE_AITER", None)
        reports["hip"] = _one_workspace(rank, device)
        queue.put(reports)
    except Exception:
        queue.put({"rank": rank, "error": traceback.format_exc()})
    finally:
        dist.destroy_process_group()


class TestFlyDSLInt4IpcWorkspace(CustomTestCase):
    @unittest.skipUnless(
        qr_rocm_arch_available() and torch.cuda.device_count() >= 2,
        "IPC workspace test needs two ROCm GPUs that support QuickReduce",
    )
    def test_int4_startup_keeps_hip_and_adds_flydsl(self):
        world_size = 2
        port = _get_open_port()
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        processes = [
            context.Process(
                target=_run_ipc_switch,
                args=(rank, world_size, port, queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()

        deadline = time.monotonic() + 600
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        if any(process.is_alive() for process in processes):
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            self.fail("INT4 IPC workspace test timed out")

        reports = []
        while not queue.empty():
            reports.append(queue.get())
        for rank, process in enumerate(processes):
            self.assertEqual(
                process.exitcode,
                0,
                f"rank {rank} exited {process.exitcode}; reports={reports}",
            )

        self.assertEqual(len(reports), world_size, reports)
        by_rank = {item["rank"]: item for item in reports}
        for rank in range(world_size):
            item = by_rank[rank]
            self.assertNotIn("error", item, item.get("error"))
            flydsl = item.get("flydsl")
            hip = item["hip"]
            print(
                f"rank {rank}: flydsl_delta={None if flydsl is None else flydsl['delta']}"
                f" hip_delta={hip['delta']} qr_max_size={hip['qr_max_size']}",
                flush=True,
            )
            self._assert_workspaces(item)

    def _assert_workspaces(self, item: dict) -> None:
        hip = item["hip"]
        self.assertFalse(hip["disabled"], hip)
        self.assertFalse(hip["has_flydsl"], hip)
        self.assertGreater(hip["ptr"], 0, hip)
        self.assertGreater(hip["delta"], hip["qr_max_size"], hip)
        self.assertLess(hip["delta"], 2 * hip["qr_max_size"] + 512 * _MIB, hip)
        self.assertTrue(hip["bf16_finite"], hip)
        self.assertTrue(hip["fp16_finite"], hip)
        self.assertLess(abs(hip["released_to"] - hip["before"]), 512 * _MIB, hip)

        if not item["flydsl_available"]:
            return

        # gfx950 reserves the HIP workspace and the FlyDSL workspace. The
        # extra memory is the FlyDSL heaps plus the peer IPC mappings, which
        # is far below a second HIP reservation.
        flydsl = item["flydsl"]
        self.assertFalse(flydsl["disabled"], flydsl)
        self.assertTrue(flydsl["has_flydsl"], flydsl)
        self.assertGreater(flydsl["ptr"], 0, flydsl)
        self.assertGreater(flydsl["flydsl_bytes"], 0, flydsl)
        self.assertGreater(flydsl["delta"], hip["delta"], flydsl)
        self.assertLess(flydsl["delta"], hip["delta"] + 1024 * _MIB, flydsl)
        self.assertTrue(flydsl["bf16_finite"], flydsl)
        self.assertTrue(flydsl["fp16_finite"], flydsl)
        self.assertLess(
            abs(flydsl["released_to"] - flydsl["before"]), 512 * _MIB, flydsl
        )


if __name__ == "__main__":
    unittest.main()
