"""Unit tests for MooncakeStoreConfig local_hostname and protocol resolution.

Regression for sgl-project/sglang#23457: runtime attach must resolve
per-node local_hostname from process environment instead of a shared default.
"""

import unittest

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeStoreConfig,
)
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()
