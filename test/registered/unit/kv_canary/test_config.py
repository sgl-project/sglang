"""Tests for sglang.srt.kv_canary.config: CanaryMode enum and CanaryConfig.from_env."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.test.test_utils import CustomTestCase

_ENV_RING = "SGLANG_KV_CANARY_RING_CAPACITY"
_ENV_WRITE = "SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT"
_ENV_VERIFY = "SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT"
_ENV_STATS = "SGLANG_KV_CANARY_STATS_PRINT_EVERY_N_STEPS"

_ALL_ENV_KEYS = (_ENV_RING, _ENV_WRITE, _ENV_VERIFY, _ENV_STATS)


def _make_server_args(mode="none", real_data="PARTIAL", sweep_interval=0):
    return SimpleNamespace(
        kv_canary=mode,
        kv_canary_real_data=real_data,
        kv_canary_sweep_interval=sweep_interval,
    )


def _clean_env():
    """Return a copy of os.environ with all kv_canary keys removed."""
    return {k: v for k, v in os.environ.items() if k not in _ALL_ENV_KEYS}


class TestCanaryModeEnum(CustomTestCase):
    def test_none_value(self):
        self.assertEqual(CanaryMode.NONE.value, "none")

    def test_log_value(self):
        self.assertEqual(CanaryMode.LOG.value, "log")

    def test_raise_value(self):
        self.assertEqual(CanaryMode.RAISE.value, "raise")

    def test_members_count(self):
        self.assertEqual(len(list(CanaryMode)), 3)

    def test_is_str_subclass(self):
        self.assertIsInstance(CanaryMode.NONE, str)


class TestCanaryConfigFromEnvMode(CustomTestCase):
    def test_mode_none(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            cfg = CanaryConfig.from_env(_make_server_args(mode="none"))
        self.assertEqual(cfg.mode, CanaryMode.NONE)

    def test_mode_log(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            cfg = CanaryConfig.from_env(_make_server_args(mode="log"))
        self.assertEqual(cfg.mode, CanaryMode.LOG)

    def test_mode_raise(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            cfg = CanaryConfig.from_env(_make_server_args(mode="raise"))
        self.assertEqual(cfg.mode, CanaryMode.RAISE)

    def test_mode_uppercase_is_accepted(self):
        # from_env calls .lower(), so "NONE" becomes "none" and is valid.
        with patch.dict(os.environ, _clean_env(), clear=True):
            cfg = CanaryConfig.from_env(_make_server_args(mode="NONE"))
        self.assertEqual(cfg.mode, CanaryMode.NONE)

    def test_mode_with_whitespace_stripped(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            cfg = CanaryConfig.from_env(_make_server_args(mode="  log  "))
        self.assertEqual(cfg.mode, CanaryMode.LOG)

    def test_invalid_mode_raises_value_error(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            with self.assertRaises(ValueError) as ctx:
                CanaryConfig.from_env(_make_server_args(mode="disabled"))
        self.assertIn("disabled", str(ctx.exception))

    def test_invalid_mode_warn_raises(self):
        # "warn" is not a valid mode (valid: none/log/raise).
        with patch.dict(os.environ, _clean_env(), clear=True):
            with self.assertRaises(ValueError):
                CanaryConfig.from_env(_make_server_args(mode="warn"))

    def test_invalid_mode_health_check_raises(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            with self.assertRaises(ValueError):
                CanaryConfig.from_env(_make_server_args(mode="health_check"))


class TestCanaryConfigFromEnvDefaults(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._patched = patch.dict(os.environ, _clean_env(), clear=True)
        self._patched.start()

    def tearDown(self):
        self._patched.stop()
        super().tearDown()

    def _cfg(self, **kwargs):
        return CanaryConfig.from_env(_make_server_args(**kwargs))

    def test_default_ring_capacity(self):
        cfg = self._cfg()
        self.assertEqual(cfg.ring_capacity, 1024)

    def test_default_enable_write_input_assert(self):
        cfg = self._cfg()
        self.assertFalse(cfg.enable_write_input_assert)

    def test_default_enable_verify_token_assert(self):
        cfg = self._cfg()
        self.assertFalse(cfg.enable_verify_token_assert)

    def test_default_stats_print_every_n_steps(self):
        cfg = self._cfg()
        self.assertEqual(cfg.stats_print_every_n_steps, 100)

    def test_sweep_interval_from_server_args(self):
        cfg = self._cfg(sweep_interval=7)
        self.assertEqual(cfg.sweep_interval, 7)

    def test_real_kv_hash_mode_partial(self):
        cfg = self._cfg(real_data="PARTIAL")
        self.assertEqual(cfg.real_kv_hash_mode, RealKvHashMode.PARTIAL)

    def test_real_kv_hash_mode_none(self):
        cfg = self._cfg(real_data="NONE")
        self.assertEqual(cfg.real_kv_hash_mode, RealKvHashMode.NONE)

    def test_real_kv_hash_mode_all(self):
        cfg = self._cfg(real_data="ALL")
        self.assertEqual(cfg.real_kv_hash_mode, RealKvHashMode.ALL)


class TestCanaryConfigFromEnvOverrides(CustomTestCase):
    def test_ring_capacity_from_env(self):
        env = {**_clean_env(), _ENV_RING: "512"}
        with patch.dict(os.environ, env, clear=True):
            cfg = CanaryConfig.from_env(_make_server_args())
        self.assertEqual(cfg.ring_capacity, 512)

    def test_enable_write_input_assert_from_env(self):
        env = {**_clean_env(), _ENV_WRITE: "1"}
        with patch.dict(os.environ, env, clear=True):
            cfg = CanaryConfig.from_env(_make_server_args())
        self.assertTrue(cfg.enable_write_input_assert)

    def test_enable_verify_token_assert_from_env(self):
        env = {**_clean_env(), _ENV_VERIFY: "1"}
        with patch.dict(os.environ, env, clear=True):
            cfg = CanaryConfig.from_env(_make_server_args())
        self.assertTrue(cfg.enable_verify_token_assert)

    def test_stats_print_every_n_steps_from_env(self):
        env = {**_clean_env(), _ENV_STATS: "50"}
        with patch.dict(os.environ, env, clear=True):
            cfg = CanaryConfig.from_env(_make_server_args())
        self.assertEqual(cfg.stats_print_every_n_steps, 50)


class TestCanaryConfigFieldAccess(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._patched = patch.dict(os.environ, _clean_env(), clear=True)
        self._patched.start()
        self._cfg = CanaryConfig.from_env(
            _make_server_args(mode="log", sweep_interval=3)
        )

    def tearDown(self):
        self._patched.stop()
        super().tearDown()

    def test_all_fields_accessible(self):
        cfg = self._cfg
        _ = cfg.mode
        _ = cfg.ring_capacity
        _ = cfg.sweep_interval
        _ = cfg.real_kv_hash_mode
        _ = cfg.enable_write_input_assert
        _ = cfg.enable_verify_token_assert
        _ = cfg.stats_print_every_n_steps

    def test_frozen_dataclass_rejects_mutation(self):
        with self.assertRaises((AttributeError, TypeError)):
            self._cfg.mode = CanaryMode.NONE


if __name__ == "__main__":
    unittest.main(verbosity=3)
