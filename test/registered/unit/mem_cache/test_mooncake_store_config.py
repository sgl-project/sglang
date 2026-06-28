import json
import os
import tempfile
import unittest

from sglang.srt.environ import temp_set_env
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeStoreConfig,
)


class TestMooncakeStoreConfigLocalHostname(unittest.TestCase):
    _BASE_EXTRA_CONFIG = {"master_server_address": "127.0.0.1:50051"}

    def _load_from_extra_config(self, extra_config=None):
        config = {**self._BASE_EXTRA_CONFIG, **(extra_config or {})}
        return MooncakeStoreConfig.load_from_extra_config(config)

    def test_load_from_extra_config_uses_mooncake_env_when_omitted(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME="10.0.0.2"):
            cfg = self._load_from_extra_config()
        self.assertEqual(cfg.local_hostname, "10.0.0.2")

    def test_load_from_extra_config_uses_local_hostname_env_when_omitted(self):
        with temp_set_env(LOCAL_HOSTNAME="10.0.0.3"):
            cfg = self._load_from_extra_config()
        self.assertEqual(cfg.local_hostname, "10.0.0.3")

    def test_load_from_extra_config_defaults_to_localhost_without_env(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME=None, LOCAL_HOSTNAME=None):
            cfg = self._load_from_extra_config()
        self.assertEqual(cfg.local_hostname, "localhost")

    def test_load_from_extra_config_uses_explicit_override_without_env(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME=None, LOCAL_HOSTNAME=None):
            cfg = self._load_from_extra_config({"local_hostname": "10.0.0.9"})
        self.assertEqual(cfg.local_hostname, "10.0.0.9")

    def test_load_from_extra_config_prefers_env_over_broadcast_override(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME="10.0.0.4"):
            cfg = self._load_from_extra_config({"local_hostname": "10.0.0.1"})
        self.assertEqual(cfg.local_hostname, "10.0.0.4")

    def test_load_from_env_uses_mooncake_env(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME="10.0.0.5", MOONCAKE_MASTER="127.0.0.1:50051"):
            cfg = MooncakeStoreConfig.load_from_env()
        self.assertEqual(cfg.local_hostname, "10.0.0.5")

    def test_from_file_uses_process_env_when_local_hostname_omitted(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as fin:
            json.dump(
                {
                    "master_server_address": "127.0.0.1:50051",
                    "metadata_server": "P2PHANDSHAKE",
                },
                fin,
            )
            config_path = fin.name

        try:
            with temp_set_env(
                allow_sglang=True,
                SGLANG_HICACHE_MOONCAKE_CONFIG_PATH=config_path,
                MOONCAKE_LOCAL_HOSTNAME="10.0.0.6",
            ):
                cfg = MooncakeStoreConfig.from_file()
            self.assertEqual(cfg.local_hostname, "10.0.0.6")
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    unittest.main()
