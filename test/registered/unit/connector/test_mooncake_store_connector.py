import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "0")
sys.modules.setdefault("deep_gemm", ModuleType("deep_gemm"))

from sglang.srt.connector.mooncake_store import (
    DEFAULT_LOCAL_BUFFER_SIZE,
    MooncakeStoreConnector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class FakeStore:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key, b"")

    def put(self, key, value, config=None):
        self.data[key] = value
        return 0

    def is_exist(self, key):
        return int(key in self.data)

    def batch_put_from(self, keys, tensor_ptrs, tensor_sizes, config):
        raise AssertionError("batch_put_from should not be used in this test")

    def batch_get_into(self, keys, tensor_ptrs, tensor_sizes):
        raise AssertionError("batch_get_into should not be used in this test")


class FakeInitStore:
    def __init__(self):
        self.setup_args = None
        self.setup_dummy_args = None

    def setup(self, *args):
        self.setup_args = args
        return 0

    def setup_dummy(self, *args):
        self.setup_dummy_args = args
        return 0


class TestMooncakeStoreConnector(unittest.TestCase):
    def setUp(self):
        self.connector = object.__new__(MooncakeStoreConnector)
        self.connector.closed = True
        self.connector.model_name = "model"
        self.connector.store = FakeStore()
        self.connector._rep_config = object()

    def test_getstr_missing_key_returns_none(self):
        self.assertIsNone(self.connector.getstr("missing"))

    def test_getstr_existing_empty_value_returns_empty_string(self):
        self.connector.store.put("empty", b"")
        self.assertEqual(self.connector.getstr("empty"), "")

    def test_setstr_registers_index_for_existing_file(self):
        self.connector.setstr("model/files/config.json", '{"model_type":"opt"}')
        self.connector.setstr("model/files/tokenizer_config.json", "{}")

        self.assertEqual(
            self.connector.list("model/files/"),
            [
                "model/files/config.json",
                "model/files/tokenizer_config.json",
            ],
        )

    def test_init_uses_setup_dummy_for_standalone_storage(self):
        fake_store_module = ModuleType("mooncake.store")
        fake_store_module.MooncakeDistributedStore = FakeInitStore
        fake_store_module.ReplicateConfig = type("ReplicateConfig", (), {})

        fake_config = SimpleNamespace(
            local_hostname="127.0.0.1",
            metadata_server="http://127.0.0.1:18080/metadata",
            global_segment_size=1234,
            protocol="tcp",
            device_name="",
            master_server_address="127.0.0.1:50081",
            standalone_storage=True,
            client_server_address="127.0.0.1:50052",
        )
        fake_config_module = ModuleType(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store"
        )
        fake_config_module.MooncakeStoreConfig = SimpleNamespace(
            load_from_env=mock.Mock(return_value=fake_config)
        )

        with mock.patch.dict(
            sys.modules,
            {
                "mooncake.store": fake_store_module,
                "sglang.srt.mem_cache": ModuleType("sglang.srt.mem_cache"),
                "sglang.srt.mem_cache.storage": ModuleType(
                    "sglang.srt.mem_cache.storage"
                ),
                "sglang.srt.mem_cache.storage.mooncake_store": ModuleType(
                    "sglang.srt.mem_cache.storage.mooncake_store"
                ),
                "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store": (
                    fake_config_module
                ),
            },
        ):
            connector = MooncakeStoreConnector("mooncake:///model")

        self.assertIsNone(connector.store.setup_args)
        self.assertEqual(
            connector.store.setup_dummy_args,
            (1234, DEFAULT_LOCAL_BUFFER_SIZE, "127.0.0.1:50052"),
        )

    def test_standalone_storage_put_and_get_use_byte_fallback(self):
        self.connector.config = SimpleNamespace(standalone_storage=True)
        tensor = torch.arange(8, dtype=torch.float32)

        with mock.patch("sglang.srt.connector.mooncake_store.STANDALONE_CHUNK_SIZE", 8):
            self.connector.batch_put_from(["model/keys/rank_0/test.weight"], [tensor])

            loaded = torch.empty_like(tensor)
            self.connector.batch_get_into(["keys/rank_0/test.weight"], [loaded])

        self.assertTrue(torch.equal(loaded, tensor))
        self.assertIn(
            "model/keys/rank_0/test.weight.__chunk_manifest__",
            self.connector.store.data,
        )


if __name__ == "__main__":
    unittest.main()
