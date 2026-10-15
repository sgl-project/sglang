"""Actual CPU mappings and Gloo failures for the opt-in shared host allocator."""

from __future__ import annotations

import gc
import json
import os
import tempfile
import time
import unittest
import weakref
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.mem_cache.pool_host import base as pool_base
from sglang.srt.mem_cache.pool_host import mla as mla_pool
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_host_register,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.shared import SharedHostTensorAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=35, suite="base-a-test-cpu")


class _Cudart:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.fail_unregister = False
        self.registered = []
        self.unregistered = []

    def cudaHostRegister(self, pointer, size, flags):
        self.registered.append((pointer, size, flags))
        return int(self.fail)

    def cudaHostUnregister(self, pointer):
        self.unregistered.append(pointer)
        return int(self.fail_unregister)

    def cudaGetErrorString(self, code):
        return "injected registration failure"


def _expect_failure(function, message):
    try:
        function()
    except (RuntimeError, ValueError) as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"expected rejection containing {message!r}")


def _worker(rank, rendezvous, output, scenario):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    allocator = SharedHostTensorAllocator(dist.group.WORLD)
    names = []

    def make_name(kind):
        name = f"sgl_shared_allocator_test_{os.getpid()}_{len(names)}"
        names.append(name)
        return name

    try:
        with mock.patch(
            "sglang.srt.mem_cache.pool_host.shared.make_shm_name", side_effect=make_name
        ):
            if scenario == "mapping":
                buffers = []
                for iteration in range(3):
                    buffer = allocator.allocate((2, 16), torch.int32, "cpu")
                    buffers.append(buffer)
                    if rank == 0:
                        buffer.copy_(torch.arange(32).reshape(2, 16) + iteration)
                    dist.barrier()
                    assert torch.equal(
                        buffer, torch.arange(32).reshape(2, 16) + iteration
                    )
                    if rank == 1:
                        buffer[1, 15] = 100 + iteration
                    dist.barrier()
                    assert buffer[1, 15].item() == 100 + iteration
                assert allocator.owner_rank == 0
                assert allocator.rank == rank and allocator.world_size == 2
                assert allocator.is_owner == (rank == 0)
                assert len({buffer.data_ptr() for buffer in buffers}) == 3
            elif scenario == "spec":
                for dims, dtype in (
                    ((2 + rank, 8), torch.float32),
                    ((2, 8), torch.float32 if rank == 0 else torch.int32),
                ):
                    _expect_failure(
                        lambda: allocator.allocate(dims, dtype, "cpu"),
                        "shape/dtype differs",
                    )
                _expect_failure(
                    lambda: allocator.allocate(
                        (2, 8), torch.float32, "cuda" if rank else "cpu"
                    ),
                    "must be allocated on CPU",
                )
                _expect_failure(
                    lambda: allocator.synchronize_error(
                        ValueError() if rank else None, operation="empty-message error"
                    ),
                    "rank 1:",
                )
            elif scenario == "space":
                with mock.patch(
                    "sglang.srt.mem_cache.pool_host.shared.os.statvfs",
                    return_value=SimpleNamespace(f_bavail=0, f_frsize=4096),
                ):
                    _expect_failure(
                        lambda: allocator.allocate((2, 8), torch.float32, "cpu"),
                        "insufficient space",
                    )
            elif scenario == "attachment":
                with mock.patch.object(
                    allocator,
                    "_open_mapping",
                    side_effect=OSError("peer cannot open mapping"),
                ):
                    _expect_failure(
                        lambda: allocator.allocate((2, 8), torch.float32, "cpu"),
                        "peer cannot open mapping",
                    )
            elif scenario == "registration":
                runtime = _Cudart(fail=rank == 1)
                with mock.patch.object(torch.cuda, "cudart", return_value=runtime):
                    _expect_failure(
                        lambda: alloc_with_host_register(
                            (2, 8), torch.float32, "cpu", True, allocator
                        ),
                        "CUDA host registration",
                    )
                assert [flags for _, _, flags in runtime.registered] == [1]
                assert len(runtime.unregistered) == (1 if rank == 0 else 0)
                assert not allocator._mappings
            elif scenario == "metadata":
                runtime = _Cudart()
                created_allocators = []
                real_factory = pool_base.get_allocator_from_storage

                def create_allocator(*args, **kwargs):
                    result = real_factory(*args, **kwargs)
                    created_allocators.append(result)
                    return result

                def create_pointer_table(buffers, *args, **kwargs):
                    if rank == 1:
                        raise RuntimeError("peer GPU pointer metadata failed")
                    return torch.tensor(
                        [buffer.data_ptr() for buffer in buffers], dtype=torch.uint64
                    )

                device_pool = SimpleNamespace(
                    device="cuda",
                    store_dtype=torch.float32,
                    size=8,
                    start_layer=0,
                    end_layer=2,
                    layer_num=2,
                    layer_shard_enabled=False,
                    kv_lora_rank=8,
                    qk_rope_head_dim=2,
                    kv_buffer=[torch.empty(10, 1, 10) for _ in range(2)],
                )
                with (
                    mock.patch.object(torch.cuda, "cudart", return_value=runtime),
                    mock.patch.object(pool_base, "_is_cuda", True),
                    mock.patch.object(
                        pool_base, "host_memory_budget_bytes", return_value=1024**3
                    ),
                    mock.patch.object(
                        pool_base,
                        "get_allocator_from_storage",
                        side_effect=create_allocator,
                    ),
                    mock.patch.object(
                        mla_pool, "can_use_hicache_jit_kernel", return_value=False
                    ),
                    mock.patch.object(
                        mla_pool,
                        "make_kernel_ptr_table",
                        side_effect=create_pointer_table,
                    ),
                    mock.patch.dict(
                        ALLOC_MEMORY_FUNCS, {"cuda": alloc_with_host_register}
                    ),
                ):
                    _expect_failure(
                        lambda: MLATokenToKVPoolHost(
                            device_pool,
                            2,
                            0,
                            2,
                            "page_first_direct",
                            shared_memory_group=dist.group.WORLD,
                        ),
                        "peer GPU pointer metadata failed",
                    )
                assert len(runtime.unregistered) == 1
                assert len(created_allocators) == 1
                assert not created_allocators[0]._mappings
                assert created_allocators[0]._closed
            elif scenario in ("destroy", "unregister"):
                runtime = _Cudart()
                with (
                    mock.patch.object(torch.cuda, "cudart", return_value=runtime),
                    mock.patch.object(pool_base, "_is_cuda", True),
                ):
                    buffer = alloc_with_host_register(
                        (2, 8), torch.float32, "cpu", True, allocator
                    )
                    buffer.fill_(17)
                    mapping_ref = weakref.ref(next(iter(allocator._mappings.values())))
                    external_view = buffer[0]
                    pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
                    pool.allocator = allocator
                    pool.pin_memory = True
                    pool.kv_buffer = buffer
                    pool.data_refs = [buffer[0], buffer[1]]
                    pool.data_ptrs = None
                    buffer = None
                    if scenario == "unregister":
                        runtime.fail_unregister = True
                        _expect_failure(pool.destroy, "unregister failed")
                        assert pool.kv_buffer is not None
                        assert allocator._mappings
                        runtime.fail_unregister = False
                    pool.destroy()
                    pool.destroy()
                    assert len(runtime.unregistered) == (
                        2 if scenario == "unregister" else 1
                    )
                    assert not allocator._mappings and not pool.data_refs
                    assert mapping_ref() is not None
                    assert external_view.tolist() == [17.0] * 8
                    external_view = None
                    gc.collect()
                    assert mapping_ref() is None, (
                        "mapping outlived its final tensor storage"
                    )
                # This allocator is intentionally closed by pool.destroy().
                allocator = SharedHostTensorAllocator(dist.group.WORLD)
            else:
                raise AssertionError(scenario)

            # Any failed allocation must leave the same group usable. Also
            # verify no startup name survives either success or rollback.
            for name in names:
                assert not Path("/dev/shm", name).exists(), name
            followup = allocator.allocate((2, 4), torch.int64, "cpu")
            if rank == 0:
                followup.fill_(91)
            dist.barrier()
            assert followup.tolist() == [[91] * 4] * 2
            Path(output, f"{rank}.json").write_text(json.dumps({"passed": True}))
    finally:
        allocator.close()
        dist.destroy_process_group()


@unittest.skipUnless(
    os.path.isdir("/dev/shm")
    and hasattr(os, "posix_fallocate")
    and dist.is_gloo_available(),
    "requires Linux tmpfs and Gloo",
)
class TestHiCacheSharedAllocator(CustomTestCase):
    def _run(self, scenario):
        with tempfile.TemporaryDirectory(prefix="hicache-shared-test-") as output:
            rendezvous = os.path.join(output, "rendezvous")
            context = mp.get_context("spawn")
            processes = [
                context.Process(
                    target=_worker, args=(rank, rendezvous, output, scenario)
                )
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                deadline = time.monotonic() + 40
                for process in processes:
                    process.join(max(0, deadline - time.monotonic()))
                for rank, process in enumerate(processes):
                    self.assertFalse(process.is_alive(), f"rank {rank} timed out")
                    self.assertEqual(process.exitcode, 0, f"rank {rank} failed")
                    self.assertTrue(
                        json.loads(Path(output, f"{rank}.json").read_text())["passed"]
                    )
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    if process.pid is not None:
                        process.join(5)
                        if process.is_alive():
                            process.kill()
                            process.join(5)

    def test_one_backing_is_visible_to_both_ranks(self):
        self._run("mapping")

    def test_shape_dtype_and_device_mismatch_reject_on_every_rank(self):
        self._run("spec")

    def test_owner_space_error_rejects_every_rank(self):
        self._run("space")

    def test_peer_attachment_error_rolls_back_and_unlinks(self):
        self._run("attachment")

    def test_peer_registration_error_rolls_back_portable_registration(self):
        self._run("registration")

    def test_destroy_unregisters_before_releasing_views_and_keeps_external_view_safe(
        self,
    ):
        self._run("destroy")

    def test_unregister_error_keeps_pinned_mapping_until_successful_retry(self):
        self._run("unregister")

    def test_peer_constructor_metadata_error_cleans_up_every_pool(self):
        self._run("metadata")


if __name__ == "__main__":
    unittest.main()
