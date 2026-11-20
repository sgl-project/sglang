"""Unit tests for MooncakeStoreConfig local_hostname and protocol resolution.

Regression for sgl-project/sglang#23457: runtime attach must resolve
per-node local_hostname from process environment instead of a shared default.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
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


class TestMooncakeStoreConfigLocalBufferSize(CustomTestCase):
    _BASE_CONFIG = {"master_server_address": "127.0.0.1:50051"}

    def test_local_buffer_size_comes_from_extra_config(self):
        load = MooncakeStoreConfig.load_from_extra_config
        self.assertEqual(load(self._BASE_CONFIG).local_buffer_size, 16 * 1024**2)
        cfg = load({**self._BASE_CONFIG, "local_buffer_size": "4294967296"})
        self.assertEqual(cfg.local_buffer_size, 4 * 1024**3)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            load({**self._BASE_CONFIG, "local_buffer_size": -1})


class TestMooncakeStoreConfigPersistentRegistration(CustomTestCase):
    _BASE_CONFIG = {"client_server_address": "127.0.0.1:50052"}

    def _load(self, **overrides):
        return MooncakeStoreConfig.load_from_extra_config(
            {**self._BASE_CONFIG, **overrides}
        )

    def test_default_is_unspecified_and_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            config = self._load()
            self.assertIsNone(config.cuda_ipc_persistent_registration)
            self.assertFalse(config.cuda_ipc_persistent_registration_enabled)

    def test_unspecified_uses_exact_legacy_env_value(self):
        for value in ("0", "1", "true", "false", "", " 1"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"MC_CUDA_IPC_PERSISTENT_REGISTRATION": value}),
            ):
                config = self._load(cuda_ipc_persistent_registration=None)
                self.assertIsNone(config.cuda_ipc_persistent_registration)
                self.assertEqual(
                    config.cuda_ipc_persistent_registration_enabled, value == "1"
                )

    def test_explicit_bool_overrides_environment_without_mutating_it(self):
        for enabled, env_value in ((False, "1"), (True, "0")):
            with (
                self.subTest(enabled=enabled),
                patch.dict(
                    os.environ, {"MC_CUDA_IPC_PERSISTENT_REGISTRATION": env_value}
                ),
            ):
                config = self._load(cuda_ipc_persistent_registration=enabled)
                self.assertIs(config.cuda_ipc_persistent_registration, enabled)
                self.assertIs(config.cuda_ipc_persistent_registration_enabled, enabled)
                self.assertEqual(
                    os.environ["MC_CUDA_IPC_PERSISTENT_REGISTRATION"], env_value
                )

    def test_json_file_accepts_bool_and_null(self):
        for value in (True, False, None):
            with (
                self.subTest(value=value),
                tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file,
            ):
                json.dump(
                    {**self._BASE_CONFIG, "cuda_ipc_persistent_registration": value},
                    config_file,
                )
                config_file.flush()
                with envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.override(
                    config_file.name
                ):
                    config = MooncakeStoreConfig.from_file()
                self.assertIs(config.cuda_ipc_persistent_registration, value)

    def test_rejects_non_bool_values_in_extra_config_and_json(self):
        for value in ("false", "true", 0, 1, [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "boolean or null"):
                    self._load(cuda_ipc_persistent_registration=value)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json"
                ) as config_file:
                    json.dump(
                        {
                            **self._BASE_CONFIG,
                            "cuda_ipc_persistent_registration": value,
                        },
                        config_file,
                    )
                    config_file.flush()
                    with (
                        envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.override(
                            config_file.name
                        ),
                        self.assertRaisesRegex(ValueError, "boolean or null"),
                    ):
                        MooncakeStoreConfig.from_file()

    def test_option_only_extra_config_overrides_file_or_env_source(self):
        for source in ("file", "env"):
            for value in (True, False, None, "false"):
                with self.subTest(source=source, value=value):
                    config = self._load(
                        cuda_ipc_persistent_registration=source == "file"
                    )
                    with (
                        temp_set_env(MC_CUDA_IPC_PERSISTENT_REGISTRATION="1"),
                        envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.override(
                            "unused.json"
                        ),
                        patch.object(
                            MooncakeStoreConfig, "from_file", return_value=config
                        ) as load_file,
                        patch.object(
                            MooncakeStoreConfig, "load_from_env", return_value=config
                        ) as load_env,
                    ):
                        if source == "env":
                            envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.clear()
                        extra_config = {"cuda_ipc_persistent_registration": value}
                        if value == "false":
                            with self.assertRaisesRegex(ValueError, "boolean or null"):
                                MooncakeStoreConfig.load(extra_config)
                        else:
                            resolved = MooncakeStoreConfig.load(extra_config)
                            self.assertIs(
                                resolved.cuda_ipc_persistent_registration, value
                            )
                            self.assertIs(
                                resolved.cuda_ipc_persistent_registration_enabled,
                                True if value is None else value,
                            )
                            self.assertEqual(
                                resolved.client_server_address,
                                config.client_server_address,
                            )
                            self.assertIsNot(resolved, config)
                        (
                            load_file if source == "file" else load_env
                        ).assert_called_once()
                        (
                            load_env if source == "file" else load_file
                        ).assert_not_called()


if __name__ == "__main__":
    unittest.main()
