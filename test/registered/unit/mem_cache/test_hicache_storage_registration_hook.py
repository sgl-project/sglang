# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The storage backend's view of the initial host-pool registration burst.

``on_host_pools_registered`` fires exactly once per attach, immediately after
the last call of the initial burst: the anchor pool on the base controller
(its whole burst), after the per-pool v2 loop on the hybrid controller (which
continues the burst after super() returns). Pools registered later (dynamic
sidecars) arrive through v2 alone and do not re-fire the hook.

    python3 -m pytest test/registered/unit/mem_cache/test_hicache_storage_registration_hook.py -v
"""

import threading
import unittest
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest import mock

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.storage import StorageBackendFactory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 16


class RecordingBackend:
    """A backend that records the registration calls it receives."""

    def __init__(self, order: List[str], fail_on_complete: bool = False):
        self.order = order
        self.fail_on_complete = fail_on_complete
        self.closed = False

    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        self.order.append("v1")

    def register_mem_host_pool_v2(self, host_pool: Any, host_pool_name: Any) -> None:
        self.order.append(f"v2:{host_pool_name}")

    def on_host_pools_registered(self) -> None:
        self.order.append("complete")
        if self.fail_on_complete:
            raise RuntimeError("registration failed in the backend")

    def close(self) -> None:
        self.closed = True


def _fake_pool(name: str) -> SimpleNamespace:
    """The HostPoolGroup surface, without CUDA."""
    return SimpleNamespace(
        name=name,
        layout="page_first",
        page_size=PAGE,
        device="cpu",
        size=128,
        logical_size=128,
        can_use_write_back_jit=True,
    )


def _entry(name: PoolName, pool: Any, is_anchor: bool = False) -> PoolEntry:
    return PoolEntry(
        name=name,
        host_pool=pool,
        device_pool=SimpleNamespace(),
        layer_mapper=lambda layer: layer,
        is_primary_index_anchor=is_anchor,
    )


def _group() -> HostPoolGroup:
    return HostPoolGroup(
        [
            _entry(PoolName.KV, _fake_pool("kv"), is_anchor=True),
            _entry(PoolName.MAMBA, _fake_pool("mamba")),
        ]
    )


def _controller(cls: type, storage_host_pool: Any, mem_pool_host: Any) -> Any:
    """Only the attributes ``attach_storage_backend`` reads."""
    controller = object.__new__(cls)
    controller.enable_storage = False
    controller.storage_stop_event = threading.Event()
    controller.storage_host_pool = storage_host_pool
    controller.mem_pool_host = mem_pool_host
    controller.page_size = PAGE
    controller.host_memory_mode = "cache"
    controller.tp_group = None
    controller.attn_cp_group = None
    controller.attn_tp_group = None
    controller.pp_group = None
    controller.prefetch_hits_sync_groups = []
    controller.prefetch_completion_sync_groups = []
    if cls is HybridCacheController:
        controller.pp_prefetch_command_thread = None
        controller.pp_prefetch_command_group = None
        controller.extra_host_mem_release_queues = {}
    return controller


def _attach(
    controller: Any,
    backend: RecordingBackend,
    order: List[str],
    host_pools: Optional[List[PoolEntry]] = None,
) -> None:
    """Run ``attach_storage_backend`` with the heavy pieces stubbed out."""
    storage_config = SimpleNamespace(is_mla_model=False, extra_config={})
    # Only the hybrid attach accepts host_pools; the base attach's burst is
    # the anchor pool alone.
    kwargs = {"host_pools": host_pools} if host_pools is not None else {}
    with (
        mock.patch(
            "sglang.srt.managers.cache_controller.get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=1),
        ),
        mock.patch.object(
            StorageBackendFactory, "create_backend", return_value=backend
        ),
        mock.patch.object(
            controller, "_generate_storage_config", return_value=storage_config
        ),
        mock.patch.object(
            controller,
            "_start_storage_threads",
            side_effect=lambda: order.append("threads"),
        ),
    ):
        controller.attach_storage_backend(storage_backend="dynamic", **kwargs)


class TestRegistrationBurstOrder(CustomTestCase):
    def test_hybrid_hook_fires_after_its_v2_burst(self):
        # super() completes (threads included, as before), then the hybrid
        # burst continues and the hook fires at its end -- once.
        order: List[str] = []
        group = _group()
        controller = _controller(
            HybridCacheController, group.anchor_entry.host_pool, group
        )

        _attach(controller, RecordingBackend(order), order, host_pools=group.entries)

        self.assertEqual(order, ["v1", "threads", "v2:kv", "v2:mamba", "complete"])

    def test_base_hook_fires_right_after_the_anchor_pool(self):
        # Without a host pool group the burst is the anchor pool alone, so
        # the hook fires inside the base attach.
        order: List[str] = []
        pool = _fake_pool("kv")
        controller = _controller(HiCacheController, pool, pool)

        _attach(controller, RecordingBackend(order), order)

        self.assertEqual(order, ["v1", "complete", "threads"])

    def test_base_hook_failure_rolls_the_attach_back(self):
        # On the base path the hook runs inside the attach's guarded section,
        # so a backend failure rolls the backend back through the existing
        # error path.
        order: List[str] = []
        pool = _fake_pool("kv")
        controller = _controller(HiCacheController, pool, pool)
        backend = RecordingBackend(order, fail_on_complete=True)

        with self.assertRaisesRegex(RuntimeError, "registration failed in the backend"):
            _attach(controller, backend, order)

        self.assertEqual(order, ["v1", "complete"])
        self.assertTrue(backend.closed)
        self.assertFalse(controller.enable_storage)
        self.assertIsNone(controller.storage_backend)

    def test_hybrid_hook_failure_propagates(self):
        # On the hybrid path the burst continues after super() has completed;
        # a backend failure there propagates out of attach_storage_backend,
        # failing the attach (the post-super section has no rollback of its
        # own, exactly as for the v2 calls themselves).
        order: List[str] = []
        group = _group()
        controller = _controller(
            HybridCacheController, group.anchor_entry.host_pool, group
        )
        backend = RecordingBackend(order, fail_on_complete=True)

        with self.assertRaisesRegex(RuntimeError, "registration failed in the backend"):
            _attach(controller, backend, order, host_pools=group.entries)

        self.assertEqual(order, ["v1", "threads", "v2:kv", "v2:mamba", "complete"])

    def test_dynamic_sidecar_does_not_re_fire_the_hook(self):
        order: List[str] = []
        group = _group()
        controller = _controller(
            HybridCacheController, group.anchor_entry.host_pool, group
        )
        backend = RecordingBackend(order)

        _attach(controller, backend, order, host_pools=group.entries)
        controller.register_host_pool_entry(_entry(PoolName.SWA, _fake_pool("swa")))

        self.assertEqual(
            order,
            ["v1", "threads", "v2:kv", "v2:mamba", "complete", "v2:swa"],
        )


if __name__ == "__main__":
    unittest.main()
