import types
import unittest
from unittest.mock import patch

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
