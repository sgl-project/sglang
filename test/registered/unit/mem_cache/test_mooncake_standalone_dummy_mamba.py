import os
import types
import unittest
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _fake_mooncake_modules(fake_store_cls):
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = fake_store_cls
    return {
        "mooncake": mooncake,
        "mooncake.store": mooncake_store,
    }


class TestMooncakeStandaloneDummyMamba(CustomTestCase):
    @staticmethod
    def _linker_config(local_buffer_size, **extra_config):
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig

        return HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test",
            extra_config={
                "standalone_storage": True,
                "client_server_address": "127.0.0.1:50052",
                "local_buffer_size": local_buffer_size,
                **extra_config,
            },
        )

    def test_setup_dummy_without_host_pool(self):
        """The GPU direct linker initializes storage before registering its pools."""
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        for local_buffer_size in (16 << 20, 64 << 20, 0):
            with self.subTest(local_buffer_size=local_buffer_size):
                backend = Mock()
                backend.setup_dummy.return_value = 0
                with (
                    patch.object(
                        MooncakeStore,
                        "_import_mooncake_store",
                        return_value=lambda: backend,
                    ),
                    patch.object(
                        MooncakeStore,
                        "_import_mooncake_group_semantics",
                        return_value=(None, False),
                    ),
                    patch.object(MooncakeStore, "warmup"),
                ):
                    storage = MooncakeStore(
                        self._linker_config(local_buffer_size), mem_pool=None
                    )

                self.assertIs(storage.store, backend)
                backend.setup_dummy.assert_called_once_with(
                    0, local_buffer_size, "127.0.0.1:50052"
                )
                backend.setup.assert_not_called()

    def test_explicit_persistent_registration_is_forwarded_without_env_changes(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        for enabled, env_value in ((False, "1"), (True, "0")):
            with self.subTest(enabled=enabled):
                backend = Mock()
                backend.setup_dummy.return_value = 0
                with (
                    patch.dict(
                        os.environ,
                        {"MC_CUDA_IPC_PERSISTENT_REGISTRATION": env_value},
                    ),
                    patch.object(
                        MooncakeStore,
                        "_import_mooncake_store",
                        return_value=lambda: backend,
                    ),
                    patch.object(
                        MooncakeStore,
                        "_import_mooncake_group_semantics",
                        return_value=(None, False),
                    ),
                    patch.object(MooncakeStore, "warmup"),
                ):
                    MooncakeStore(
                        self._linker_config(
                            16 << 20, cuda_ipc_persistent_registration=enabled
                        ),
                        mem_pool=None,
                    )
                    self.assertEqual(
                        os.environ["MC_CUDA_IPC_PERSISTENT_REGISTRATION"], env_value
                    )
                backend.setup_dummy.assert_called_once_with(
                    0,
                    16 << 20,
                    "127.0.0.1:50052",
                    cuda_ipc_persistent_registration=enabled,
                )

    def test_legacy_backend_requires_upgrade_for_explicit_configuration(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
            MooncakeStoreConfig,
        )

        class LegacyStore:
            def setup_dummy(self, required_bytes, local_buffer_bytes, addr):
                return 0

        for enabled in (None, False, True):
            with self.subTest(enabled=enabled):
                storage_config = self._linker_config(
                    16 << 20, cuda_ipc_persistent_registration=enabled
                )
                resolved_config = MooncakeStoreConfig.load(storage_config.extra_config)
                with (
                    patch.object(
                        MooncakeStore,
                        "_import_mooncake_store",
                        return_value=LegacyStore,
                    ),
                    patch.object(
                        MooncakeStore,
                        "_import_mooncake_group_semantics",
                        return_value=(None, False),
                    ),
                    patch.object(MooncakeStore, "warmup"),
                    patch.object(
                        MooncakeStore,
                        "_load_config",
                        side_effect=AssertionError("config must not be loaded twice"),
                    ),
                ):
                    if enabled is None:
                        MooncakeStore(storage_config, mooncake_config=resolved_config)
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError, "Please upgrade Mooncake"
                        ):
                            MooncakeStore(
                                storage_config, mooncake_config=resolved_config
                            )

    def test_standalone_host_pool_still_requires_mooncake_allocator(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        backend = Mock()
        with (
            patch.object(
                MooncakeStore, "_import_mooncake_store", return_value=lambda: backend
            ),
            patch.object(
                MooncakeStore,
                "_import_mooncake_group_semantics",
                return_value=(None, False),
            ),
            self.assertRaisesRegex(
                RuntimeError, "requires MooncakeHostTensorAllocator"
            ),
        ):
            MooncakeStore(
                self._linker_config(16 << 20),
                mem_pool=types.SimpleNamespace(allocator=object()),
            )
        backend.setup_dummy.assert_not_called()

    def test_setup_dummy_includes_hybrid_buffers(self):
        """Standalone(dummy) must size shared mapping for KV + Mamba buffers."""
        import torch

        captured = {}

        class FakeMooncakeDistributedStore:
            def setup_dummy(self, required_bytes, local_buffer_bytes, addr):
                captured["required_bytes"] = int(required_bytes)
                captured["local_buffer_bytes"] = int(local_buffer_bytes)
                captured["addr"] = addr
                return 0

            def setup(self, *args, **kwargs):
                raise AssertionError("should not call setup() in standalone mode")

            def register_buffer(self, ptr, size):
                return 0

            def put(self, *args, **kwargs):
                return 0

            def is_exist(self, *args, **kwargs):
                return 1

            def get(self, *args, **kwargs):
                return bytes(4 * 1024)

        with patch.dict(
            "sys.modules",
            _fake_mooncake_modules(FakeMooncakeDistributedStore),
        ):
            from sglang.srt.mem_cache.hicache_storage import (
                HiCacheStorageConfig,
                PoolName,
            )
            from sglang.srt.mem_cache.storage.mooncake_store import (
                mooncake_store as mc_mod,
            )
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            class FakeAllocator:
                pass

            class FakeKVPool:
                def __init__(self):
                    # KV buffer (anchor).
                    self.kv_buffer = torch.empty((128,), dtype=torch.uint8)
                    self.size = 128
                    self.size_per_token = 1
                    self.allocator = FakeAllocator()

            class FakeMambaPool:
                def __init__(self):
                    self.temporal_buffer = torch.empty((64,), dtype=torch.uint8)
                    self.conv_buffer = [torch.empty((32,), dtype=torch.uint8)]

                def get_hybrid_pool_buffer(self):
                    return [self.temporal_buffer, *self.conv_buffer]

            class FakeEntry:
                def __init__(self, name, host_pool):
                    self.name = name
                    self.host_pool = host_pool

            class FakeHostPoolGroup:
                def __init__(self):
                    self.kv = FakeKVPool()
                    self.mamba = FakeMambaPool()
                    self.entries = [
                        FakeEntry(PoolName.KV, self.kv),
                        FakeEntry(PoolName.MAMBA, self.mamba),
                    ]

                # Anchor-like fields accessed by MooncakeStore.
                @property
                def kv_buffer(self):
                    return self.kv.kv_buffer

                @property
                def allocator(self):
                    return self.kv.allocator

                @property
                def size(self):
                    return self.kv.size

                @property
                def size_per_token(self):
                    return self.kv.size_per_token

            mem_pool = FakeHostPoolGroup()
            cfg = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                attn_cp_rank=0,
                attn_cp_size=1,
                is_mla_model=False,
                enable_storage_metrics=False,
                is_page_first_layout=True,
                model_name="test",
                extra_config={
                    "standalone_storage": True,
                    "client_server_address": "127.0.0.1:50052",
                },
            )

            with patch.object(mc_mod, "MooncakeHostTensorAllocator", FakeAllocator):
                MooncakeStore(cfg, mem_pool)

            expected = (
                mem_pool.kv.kv_buffer.numel() * mem_pool.kv.kv_buffer.element_size()
                + mem_pool.mamba.temporal_buffer.numel()
                * mem_pool.mamba.temporal_buffer.element_size()
                + mem_pool.mamba.conv_buffer[0].numel()
                * mem_pool.mamba.conv_buffer[0].element_size()
            )
            self.assertEqual(captured["required_bytes"], expected)


if __name__ == "__main__":
    unittest.main(verbosity=3)
