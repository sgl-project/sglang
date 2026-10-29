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
"""Attach-time rejection of v1-only backends on multi-pool cache stacks.

Hybrid cache stacks (Mamba/SWA/DSA sidecars, draft pools, retraction backup
pools) route page transfers through the v2 multi-pool API
(``batch_exists_v2``/``batch_get_v2``/``batch_set_v2``). Backends that do not
override those methods inherit base-class stubs that raise
``NotImplementedError``. Before this gate, the combination failed mid-serving
on the first storage transfer; now ``attach_storage_backend`` and
``register_host_pool_entry`` reject it up front with an actionable error.

    python3 -m pytest test/registered/unit/mem_cache/test_hicache_sidecar_backend_gate.py -v
"""

import threading
import unittest
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest import mock

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.storage import StorageBackendFactory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 16


class V1OnlyBackend(HiCacheStorage):
    """A backend that only implements the v1 surface (like aibrix/simm/eic).

    Inherits the v2 stubs from HiCacheStorage without overriding them: the
    capability probe must report "no v2" through inheritance alone.
    """

    def get(
        self,
        key: str,
        target_location: Any = None,
        target_sizes: Any = None,
    ):
        return None

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        return []

    def set(
        self,
        key: str,
        value: Any = None,
        target_location: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        return True

    def batch_set(
        self, keys, values=None, target_locations=None, target_sizes=None
    ) -> bool:
        return True

    def exists(self, key: str) -> bool:
        return False

    # Registration surface used by attach_storage_backend.
    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        pass

    def register_mem_host_pool_v2(self, host_pool: Any, host_pool_name: Any) -> None:
        pass

    def on_host_pools_registered(self) -> None:
        pass

    def close(self) -> None:
        pass


class V2Backend(V1OnlyBackend):
    """A backend that overrides the full v2 surface (like mooncake/hf3fs)."""

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        return None  # content irrelevant to the gate

    def batch_get_v2(self, transfers, extra_info=None):
        return None

    def batch_set_v2(self, transfers, extra_info=None):
        return None


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


def _controller(group: HostPoolGroup) -> HybridCacheController:
    """Only the attributes ``attach_storage_backend`` reads."""
    controller = object.__new__(HybridCacheController)
    controller.enable_storage = False
    controller.storage_stop_event = threading.Event()
    controller.storage_host_pool = group.anchor_entry.host_pool
    controller.mem_pool_host = group
    controller.page_size = PAGE
    controller.host_memory_mode = "cache"
    controller.tp_group = None
    controller.attn_cp_group = None
    controller.attn_tp_group = None
    controller.pp_group = None
    controller.prefetch_hits_sync_groups = []
    controller.prefetch_completion_sync_groups = []
    controller.pp_prefetch_command_thread = None
    controller.pp_prefetch_command_group = None
    controller.pp_prefetch_states = {}
    controller.pp_prefetch_decisions = {}
    controller.pp_prefetch_state_lock = threading.Lock()
    controller.extra_host_mem_release_queues = {}
    return controller


def _attach(
    controller: Any, backend: Any, host_pools: Optional[List[PoolEntry]]
) -> None:
    """Run ``attach_storage_backend`` with the heavy pieces stubbed out."""
    storage_config = SimpleNamespace(is_mla_model=False, extra_config={})
    with (
        mock.patch.object(
            StorageBackendFactory, "create_backend", return_value=backend
        ),
        mock.patch.object(
            controller, "_generate_storage_config", return_value=storage_config
        ),
        mock.patch.object(controller, "_start_storage_threads"),
    ):
        controller.attach_storage_backend(
            storage_backend="dynamic", host_pools=host_pools
        )


class TestSupportsPoolTransfersProbe(CustomTestCase):
    def test_v1_only_backend_reports_no_support(self):
        self.assertFalse(V1OnlyBackend.supports_pool_transfers())

    def test_v2_backend_reports_support(self):
        self.assertTrue(V2Backend.supports_pool_transfers())

    def test_base_class_reports_no_support(self):
        self.assertFalse(HiCacheStorage.supports_pool_transfers())

    def test_partial_override_reports_no_support(self):
        # Overriding only one of the three v2 methods must NOT advertise
        # support: the other two calls would still hit the raising stub.

        class PartialBackend(V1OnlyBackend):
            def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
                return None

        self.assertFalse(PartialBackend.supports_pool_transfers())

    def test_real_v1_backends_report_no_support(self):
        # The three backends known to lack the v2 surface today. Their vendor
        # SDKs are optional dependencies, so verify from source instead of
        # importing: the class must not define any of the three v2 methods.
        import ast
        import inspect
        from pathlib import Path

        mem_cache_dir = Path(inspect.getsourcefile(HiCacheStorage) or ".").parent
        backend_files = {
            "AibrixKVCacheStorage": "storage/aibrix_kvcache/aibrix_kvcache_storage.py",
            "EICStorage": "storage/eic/eic_storage.py",
            "HiCacheSiMM": "storage/simm/hicache_simm.py",
        }
        for class_name, rel_path in backend_files.items():
            tree = ast.parse((mem_cache_dir / rel_path).read_text())
            classes = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ]
            self.assertEqual(len(classes), 1, f"{class_name} not found in {rel_path}")
            methods = {
                item.name
                for item in classes[0].body
                if isinstance(item, ast.FunctionDef)
            }
            v2_methods = {"batch_exists_v2", "batch_get_v2", "batch_set_v2"}
            overridden = methods & v2_methods
            self.assertFalse(
                overridden,
                f"{class_name} appears to override {sorted(overridden)}; "
                f"update the gate expectations",
            )

    def test_real_v2_backends_report_support(self):
        # HiCacheFile is stdlib-only, so probe it directly; hf3fs/nixl need
        # optional vendor SDKs, so verify from source the same way.
        import ast
        import inspect
        from pathlib import Path

        from sglang.srt.mem_cache.hicache_storage import HiCacheFile

        self.assertTrue(HiCacheFile.supports_pool_transfers())

        mem_cache_dir = Path(inspect.getsourcefile(HiCacheFile) or ".").parent
        backend_files = {
            "HiCacheHF3FS": "storage/hf3fs/storage_hf3fs.py",
            "HiCacheNixl": "storage/nixl/hicache_nixl.py",
        }
        v2_methods = {"batch_exists_v2", "batch_get_v2", "batch_set_v2"}
        for class_name, rel_path in backend_files.items():
            tree = ast.parse((mem_cache_dir / rel_path).read_text())
            classes = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ]
            self.assertEqual(len(classes), 1, f"{class_name} not found in {rel_path}")
            methods = {
                item.name
                for item in classes[0].body
                if isinstance(item, ast.FunctionDef)
            }
            self.assertTrue(
                v2_methods <= methods,
                f"{class_name} is missing {sorted(v2_methods - methods)}; "
                f"update the gate expectations",
            )


class TestAttachTimeGate(CustomTestCase):
    def setUp(self):
        super().setUp()
        # The attach path resolves derived parallel widths (e.g. attn_dcp_size)
        # from the published runtime context, so a case states its topology the
        # way a process does, by publishing one.
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="test")

    def test_v1_only_backend_rejected_with_sidecar_pools(self):
        group = _group()
        controller = _controller(group)

        with self.assertRaises(ValueError) as ctx:
            _attach(controller, V1OnlyBackend(), host_pools=group.entries)

        self.assertIn("batch_exists_v2", str(ctx.exception))
        self.assertIn("mamba", str(ctx.exception))

    def test_v2_backend_attaches_with_sidecar_pools(self):
        group = _group()
        controller = _controller(group)

        _attach(controller, V2Backend(), host_pools=group.entries)

        self.assertTrue(controller.enable_storage)

    def test_v1_only_backend_attaches_anchor_only(self):
        # A flat KV stack (anchor entry only) never issues v2 transfers, so
        # v1-only backends stay fully usable for plain MHA/MLA models.
        group = HostPoolGroup([_entry(PoolName.KV, _fake_pool("kv"), is_anchor=True)])
        controller = _controller(group)

        _attach(controller, V1OnlyBackend(), host_pools=group.entries)

        self.assertTrue(controller.enable_storage)

    def test_gate_failure_rolls_the_attach_back(self):
        group = _group()
        controller = _controller(group)
        backend = V1OnlyBackend()

        with self.assertRaises(ValueError):
            _attach(controller, backend, host_pools=group.entries)

        # The base attach's guarded section already rolled the backend back.
        self.assertFalse(controller.enable_storage)
        self.assertIsNone(controller.storage_backend)


class TestDynamicSidecarGate(CustomTestCase):
    def setUp(self):
        super().setUp()
        # The attach path resolves derived parallel widths (e.g. attn_dcp_size)
        # from the published runtime context, so a case states its topology the
        # way a process does, by publishing one.
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="test")

    def test_dynamic_sidecar_on_v1_backend_rejected(self):
        # Sidecars registered after attach (retraction backup pools, draft
        # sidecars) go through register_host_pool_entry; a v1-only backend
        # must be rejected there too, before the entry lands in the group.
        group = HostPoolGroup([_entry(PoolName.KV, _fake_pool("kv"), is_anchor=True)])
        controller = _controller(group)

        _attach(controller, V1OnlyBackend(), host_pools=group.entries)

        with self.assertRaises(ValueError) as ctx:
            controller.register_host_pool_entry(_entry(PoolName.SWA, _fake_pool("swa")))

        self.assertIn("batch_get_v2", str(ctx.exception))
        # The entry must not have been added despite the late rejection.
        self.assertNotIn(PoolName.SWA, group.entry_map)

    def test_dynamic_sidecar_on_v2_backend_accepted(self):
        group = HostPoolGroup([_entry(PoolName.KV, _fake_pool("kv"), is_anchor=True)])
        controller = _controller(group)

        _attach(controller, V2Backend(), host_pools=group.entries)
        controller.register_host_pool_entry(_entry(PoolName.SWA, _fake_pool("swa")))

        self.assertIn(PoolName.SWA, group.entry_map)


if __name__ == "__main__":
    unittest.main()
