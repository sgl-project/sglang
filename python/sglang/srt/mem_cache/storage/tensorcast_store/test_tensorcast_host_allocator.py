# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_fresh_python(source: str) -> subprocess.CompletedProcess[str]:
    """Run a scenario with pristine import and process-lifecycle state.

    TensorCast SDK modules may already be cached by pytest collection, and the
    adapter's process-scoped Session registry is intentionally not resettable.
    A newly executed interpreter preserves those production invariants while
    keeping import-boundary and lifecycle tests independent of test order.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
    )


# Attach, claim, fail, or terminate the process-scoped Session registry.
# Each complete lifecycle should belong to a fresh interpreter.
def test_allocator_delegates_exact_tensors_in_subprocess() -> None:
    result = _run_fresh_python("""
        import torch

        from sglang.srt.mem_cache.storage.tensorcast_store import host_allocator

        class FakeSession:
            def __init__(self):
                self.calls = []
                self.results = [
                    torch.empty((2, 3), dtype=torch.float32),
                    torch.empty((5,), dtype=torch.uint8),
                    torch.empty((1,), dtype=torch.int64),
                ]

            def allocate_host_tensor(self, shape, dtype, *, name):
                self.calls.append((shape, dtype, name))
                return self.results[len(self.calls) - 1]

            def terminate_process_session(self):
                raise AssertionError("allocator terminated the Session")

        session = FakeSession()
        attach_calls = []

        def fake_attach(options):
            attach_calls.append(options)
            return session

        host_allocator._attach_process_session = fake_attach
        source = {"tensorcast": {"daemon_address": "127.0.0.1:8073"}}
        first_allocator = host_allocator.create_tensorcast_host_allocator(
            source,
            host_memory_mode="cache",
            host_layout="page_first",
            io_backend="kernel",
            platform_name="linux",
            is_cuda_backend=True,
            world_rank=0,
            world_size=1,
        )
        second_allocator = host_allocator.create_tensorcast_host_allocator(
            source,
            host_memory_mode="cache",
            host_layout="page_first",
            io_backend="kernel",
            platform_name="linux",
            is_cuda_backend=True,
            world_rank=0,
            world_size=1,
        )

        assert isinstance(
            first_allocator, host_allocator.TensorcastHostTensorAllocator
        )
        assert isinstance(
            second_allocator, host_allocator.TensorcastHostTensorAllocator
        )
        assert first_allocator.session is session
        assert second_allocator.session is session
        assert len(attach_calls) == 1

        first = first_allocator.allocate((2, 3), torch.float32, "cpu")
        second = first_allocator.allocate((5,), torch.uint8, "cpu")
        third = second_allocator.allocate((1,), torch.int64, "cpu")
        assert first is session.results[0]
        assert second is session.results[1]
        assert third is session.results[2]
        assert session.calls == [
            ((2, 3), torch.float32, "host-pool-1"),
            ((5,), torch.uint8, "host-pool-2"),
            ((1,), torch.int64, "host-pool-1"),
        ]
        assert first_allocator.dims == (5,)
        assert first_allocator.dtype is torch.uint8

        try:
            first_allocator.allocate((1,), torch.float32, "cuda")
        except ValueError as exc:
            assert "requires CPU memory" in str(exc)
        else:
            raise AssertionError("non-CPU allocation was accepted")
        assert len(session.calls) == 3
        """)
    assert result.returncode == 0, result.stderr


def test_registry_equal_conflicting_and_store_claims_in_subprocess() -> None:
    result = _run_fresh_python("""
        from sglang.srt.mem_cache.storage.tensorcast_store import host_allocator

        class FakeSession:
            def __init__(self):
                self.terminate_calls = 0

            def terminate_process_session(self):
                self.terminate_calls += 1

        session = FakeSession()
        attach_calls = []

        def fake_attach(options):
            attach_calls.append(options)
            return session

        host_allocator._attach_process_session = fake_attach
        first_config = host_allocator.TensorcastConfig(
            daemon_address="127.0.0.1:8073"
        )
        first_options = host_allocator.build_tensorcast_session_options(
            first_config, world_rank=0, world_size=1
        )
        conflicting_options = host_allocator.build_tensorcast_session_options(
            host_allocator.TensorcastConfig(
                daemon_address="127.0.0.1:8073", exists_timeout_s=31.0
            ),
            world_rank=0,
            world_size=1,
        )

        assert host_allocator.attach_early_process_session(first_options) is session
        assert host_allocator.attach_early_process_session(first_options) is session
        assert len(attach_calls) == 1
        try:
            host_allocator.attach_early_process_session(conflicting_options)
        except host_allocator.TensorcastSessionRegistryError as exc:
            assert "conflicts" in str(exc)
        else:
            raise AssertionError("conflicting early attach was accepted")

        owner = object()
        other_owner = object()
        assert (
            host_allocator.claim_tensorcast_store_session(
                first_options, owner=owner
            )
            is session
        )
        assert (
            host_allocator.claim_tensorcast_store_session(
                first_options, owner=owner
            )
            is session
        )
        try:
            host_allocator.claim_tensorcast_store_session(
                first_options, owner=other_owner
            )
        except host_allocator.TensorcastSessionRegistryError as exc:
            assert "already owns" in str(exc)
        else:
            raise AssertionError("a second Store owner was accepted")

        assert host_allocator.terminate_tensorcast_store_session(owner=owner) is True
        assert host_allocator.terminate_tensorcast_store_session(owner=owner) is False
        assert session.terminate_calls == 1
        for operation in (
            lambda: host_allocator.attach_early_process_session(first_options),
            lambda: host_allocator.claim_tensorcast_store_session(
                first_options, owner=owner
            ),
        ):
            try:
                operation()
            except host_allocator.TensorcastSessionRegistryError as exc:
                assert "terminal" in str(exc)
            else:
                raise AssertionError("terminal registry admitted an operation")
        """)
    assert result.returncode == 0, result.stderr


def test_store_close_lifecycle_is_terminal_and_retains_allocator_roots_in_subprocess() -> (
    None
):
    result = _run_fresh_python("""
        from types import SimpleNamespace

        import torch

        import sglang.srt.runtime_context as runtime_context
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
        from sglang.srt.mem_cache.storage.tensorcast_store import host_allocator
        from sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store import (
            TensorcastStore,
        )

        runtime_context.get_parallel = lambda: SimpleNamespace(
            world_rank=0,
            world_size=1,
        )

        class FakeSession:
            def __init__(self):
                self.terminate_calls = 0

            def allocate_host_tensor(self, shape, dtype, *, name):
                del name
                return torch.empty(shape, dtype=dtype)

            def terminate_process_session(self):
                self.terminate_calls += 1

        session = FakeSession()
        host_allocator._attach_process_session = lambda options: session
        source = {
            "tensorcast": {
                "daemon_address": "127.0.0.1:8073",
                "model_id": "close-lifecycle-model",
            }
        }
        allocator = host_allocator.create_tensorcast_host_allocator(
            source,
            host_memory_mode="cache",
            host_layout="page_first_direct",
            io_backend="direct",
            platform_name="linux",
            is_cuda_backend=True,
            world_rank=0,
            world_size=1,
        )
        root = allocator.allocate((8,), torch.float32, "cpu")
        root.fill_(7)
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="unused",
            extra_config=source,
        )
        store = TensorcastStore(config)
        store.close()
        store.close()

        assert session.terminate_calls == 1
        assert store.batch_exists(["after-close"]) == 0
        indices = torch.tensor([0, 1], dtype=torch.int64)
        assert store.batch_get_v1(["after-close"], indices) == [False]
        assert store.batch_set_v1(["after-close"], indices) == [False]
        assert torch.equal(root, torch.full((8,), 7.0))
        root.add_(1)
        assert torch.equal(root, torch.full((8,), 8.0))

        try:
            TensorcastStore(config)
        except host_allocator.TensorcastSessionRegistryError as exc:
            assert "terminal" in str(exc)
        else:
            raise AssertionError("terminal registry admitted a second Store")

        try:
            host_allocator.create_tensorcast_host_allocator(
                source,
                host_memory_mode="cache",
                host_layout="page_first_direct",
                io_backend="direct",
                platform_name="linux",
                is_cuda_backend=True,
                world_rank=0,
                world_size=1,
            )
        except host_allocator.TensorcastSessionRegistryError as exc:
            assert "terminal" in str(exc)
        else:
            raise AssertionError("terminal registry admitted allocator reclaim")
        """)
    assert result.returncode == 0, result.stderr
