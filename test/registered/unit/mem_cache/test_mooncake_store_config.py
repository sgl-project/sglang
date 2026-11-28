"""Unit tests for MooncakeStoreConfig local_hostname and protocol resolution.

Regression for sgl-project/sglang#23457: runtime attach must resolve
per-node local_hostname from process environment instead of a shared default.
"""

import unittest
from unittest import mock

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeStore,
    MooncakeStoreConfig,
)
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestMooncakeStoreConfigLocalHostname(CustomTestCase):
    _BASE_EXTRA_CONFIG = {"master_server_address": "127.0.0.1:50051"}

    def _load_from_extra_config(self, extra_config=None):
        config = {**self._BASE_EXTRA_CONFIG, **(extra_config or {})}
        return MooncakeStoreConfig.load_from_extra_config(config)

    def test_load_from_extra_config_uses_mooncake_env_when_omitted(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME="10.0.0.2"):
            cfg = self._load_from_extra_config()
        self.assertEqual(cfg.local_hostname, "10.0.0.2")

    def test_load_from_extra_config_prefers_env_over_broadcast_override(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME="10.0.0.4"):
            cfg = self._load_from_extra_config({"local_hostname": "10.0.0.1"})
        self.assertEqual(cfg.local_hostname, "10.0.0.4")


class TestMooncakeStoreConfigProtocol(CustomTestCase):
    _BASE_CONFIG = {"master_server_address": "127.0.0.1:50051"}

    def _load_from_extra_config(self, extra_config=None):
        config = {**self._BASE_CONFIG, **(extra_config or {})}
        return MooncakeStoreConfig.load_from_extra_config(config)

    def test_load_from_extra_config_uses_protocol_env_when_omitted(self):
        with temp_set_env(MOONCAKE_PROTOCOL="efa"):
            cfg = self._load_from_extra_config()
        self.assertEqual(cfg.protocol, "efa")

    def test_load_from_extra_config_prefers_explicit_protocol_over_env(self):
        with temp_set_env(MOONCAKE_PROTOCOL="efa"):
            cfg = self._load_from_extra_config({"protocol": "rdma"})
        self.assertEqual(cfg.protocol, "rdma")

    def test_inherited_nvlink_protocol_warns(self):
        with temp_set_env(MOONCAKE_PROTOCOL="nvlink"):
            with self.assertLogs(level="WARNING") as logs:
                cfg = self._load_from_extra_config()
        self.assertEqual(cfg.protocol, "nvlink")
        self.assertIn("MOONCAKE_PROTOCOL=nvlink", "\n".join(logs.output))

    def test_explicit_protocol_does_not_warn_under_nvlink_env(self):
        with temp_set_env(MOONCAKE_PROTOCOL="nvlink"):
            with self.assertNoLogs(level="WARNING"):
                cfg = self._load_from_extra_config({"protocol": "rdma"})
        self.assertEqual(cfg.protocol, "rdma")

    def test_env_only_nvlink_protocol_points_to_store_config(self):
        with temp_set_env(
            MOONCAKE_MASTER="127.0.0.1:50051", MOONCAKE_PROTOCOL="nvlink"
        ):
            with self.assertLogs(level="WARNING") as logs:
                cfg = MooncakeStoreConfig.load_from_env()
        self.assertEqual(cfg.protocol, "nvlink")
        out = "\n".join(logs.output)
        self.assertIn("environment variables", out)
        self.assertIn("--hicache-storage-backend-extra-config", out)
        # A protocol-only extra config is ignored, so the hint must name the address key.
        self.assertIn("master_server_address", out)
        self.assertNotIn("config omits 'protocol'", out)


class TestMooncakeStoreWarmup(CustomTestCase):
    """A failed warmup read must raise a descriptive error, not an AssertionError."""

    def _store(self, *, exists: int, value: bytes) -> MooncakeStore:
        store = MooncakeStore.__new__(MooncakeStore)
        store.local_rank = 0
        store.config = MooncakeStoreConfig.load_from_extra_config(
            {"master_server_address": "127.0.0.1:50051", "protocol": "rdma"}
        )
        store.store = mock.Mock()
        store.store.put.return_value = 0
        store.store.is_exist.return_value = exists
        store.store.get.return_value = value
        return store

    def test_warmup_succeeds_when_value_round_trips(self):
        store = self._store(exists=1, value=b"")
        store.store.get.side_effect = lambda key: store.store.put.call_args[0][1]
        store.warmup()

    def test_warmup_get_mismatch_raises_runtime_error(self):
        store = self._store(exists=1, value=b"")
        with self.assertRaisesRegex(RuntimeError, "MC_ENABLE_DEST_DEVICE_AFFINITY"):
            store.warmup()

    def test_warmup_missing_key_raises_runtime_error(self):
        store = self._store(exists=0, value=b"")
        with self.assertRaisesRegex(RuntimeError, "master_server_address"):
            store.warmup()


class TestMooncakeStoreConfigLocalBufferSize(CustomTestCase):
    _BASE_CONFIG = {"master_server_address": "127.0.0.1:50051"}

    def test_local_buffer_size_comes_from_extra_config(self):
        load = MooncakeStoreConfig.load_from_extra_config
        self.assertEqual(load(self._BASE_CONFIG).local_buffer_size, 16 * 1024**2)
        cfg = load({**self._BASE_CONFIG, "local_buffer_size": "4294967296"})
        self.assertEqual(cfg.local_buffer_size, 4 * 1024**3)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            load({**self._BASE_CONFIG, "local_buffer_size": -1})


if __name__ == "__main__":
    unittest.main()
