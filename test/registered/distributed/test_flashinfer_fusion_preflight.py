"""Distributed tests for FlashInfer allreduce-fusion workspace preflight."""

import multiprocessing as mp
import os
import socket
import unittest

import torch

from sglang.srt.utils import get_cuda_driver_bindings, is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-2-gpu-large")

WORLD_SIZE = 2


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_rank(rank, world_size, port, scenario, result_q):
    held = None
    cuda_driver = None
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        torch.cuda.set_device(rank)

        import torch.distributed as dist

        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
        )
        cpu_group = dist.group.WORLD

        from sglang.srt.layers.flashinfer_comm_fusion import (
            _make_flashinfer_workspace_allocation_prop,
            _preflight_check_workspace_memory,
        )

        probe_kwargs = dict(
            world_size=8,
            max_token_num=2048,
            hidden_dim=12288,
            dtype=torch.bfloat16,
            cpu_group=cpu_group,
        )

        if scenario == "rank0_starved" and rank == 0:
            cuda_driver = get_cuda_driver_bindings()
            prop = _make_flashinfer_workspace_allocation_prop(cuda_driver)

            free, _total = torch.cuda.mem_get_info(rank)
            target = max(free - (1 << 30), 0)
            granularity_flag = (
                cuda_driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
            )
            err, gran = cuda_driver.cuMemGetAllocationGranularity(
                prop,
                granularity_flag,
            )
            assert err == cuda_driver.CUresult.CUDA_SUCCESS, err
            aligned = (target // gran) * gran
            assert aligned > 0, "not enough free memory to starve the preflight"
            err, held = cuda_driver.cuMemCreate(aligned, prop, 0)
            assert err == cuda_driver.CUresult.CUDA_SUCCESS, (err, aligned)

        decision = _preflight_check_workspace_memory(**probe_kwargs)
        result_q.put((rank, "ok", bool(decision)))
    except Exception as e:  # pragma: no cover - debug path
        result_q.put((rank, "err", repr(e)))
    finally:
        if held is not None:
            cuda_driver.cuMemRelease(held)
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def _spawn_and_collect(scenario, world_size=WORLD_SIZE):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _get_free_port()
    procs = []
    for rank in range(world_size):
        proc = ctx.Process(
            target=_run_rank,
            args=(rank, world_size, port, scenario, q),
        )
        proc.start()
        procs.append(proc)

    try:
        results = {}
        for _ in range(world_size):
            rank, status, payload = q.get(timeout=300)
            results[rank] = (status, payload)

        for proc in procs:
            proc.join(timeout=60)
            assert proc.exitcode == 0, f"rank exited with {proc.exitcode}"
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)

    return results


class TestFlashInferPreflightDistributed(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
            raise unittest.SkipTest(
                f"Need {WORLD_SIZE} CUDA devices, got {torch.cuda.device_count()}"
            )
        if not is_flashinfer_available():
            raise unittest.SkipTest("FlashInfer is not available")
        try:
            from sglang.srt.layers.flashinfer_comm_fusion import (
                _make_flashinfer_workspace_allocation_prop,
            )

            cuda_driver = get_cuda_driver_bindings()
            _make_flashinfer_workspace_allocation_prop(cuda_driver)
        except Exception as e:
            raise unittest.SkipTest(
                f"FlashInfer preflight dependencies unavailable: {e}"
            )

    def test_happy_path_votes_proceed(self):
        results = _spawn_and_collect("normal")
        for rank, (status, payload) in results.items():
            self.assertEqual(status, "ok", f"rank {rank}: {payload}")
            self.assertTrue(payload, f"rank {rank} voted SKIP unexpectedly")

    def test_starved_rank_broadcasts_skip(self):
        results = _spawn_and_collect("rank0_starved")
        for rank, (status, payload) in results.items():
            self.assertEqual(status, "ok", f"rank {rank}: {payload}")
            self.assertFalse(
                payload,
                f"rank {rank} voted PROCEED but rank 0 was starved",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
