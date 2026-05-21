from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = "1"

from sglang.srt.kv_canary.token_oracle.install import install_token_oracle_from_env
from sglang.srt.layers.sampler import _CUSTOM_SAMPLER_FACTORIES
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")


def _make_server_args(*, sampling_backend: str) -> SimpleNamespace:
    return SimpleNamespace(sampling_backend=sampling_backend)


class TestInstallTokenOracleFromEnv(CustomTestCase):
    def test_install_token_oracle_from_env_disabled_returns_none(self) -> None:
        """Verify disabled token oracle installation returns no sampler hook."""
        server_args = _make_server_args(sampling_backend="auto")
        hook = install_token_oracle_from_env(server_args=server_args, vocab_size=1000)
        self.assertIsNone(hook)

    def test_install_token_oracle_from_env_enabled_registers_oracle_backend(
        self,
    ) -> None:
        """Verify enabled token oracle installation registers the oracle backend."""
        server_args = _make_server_args(sampling_backend="token_oracle")
        hook = install_token_oracle_from_env(server_args=server_args, vocab_size=512)
        self.assertIsNotNone(hook)
        self.assertIn("token_oracle", _CUSTOM_SAMPLER_FACTORIES)

    def test_install_token_oracle_from_env_enabled_returns_hook_with_hash_oracle(
        self,
    ) -> None:
        """Verify enabled token oracle installation returns a hook with HashOracle."""
        from sglang.srt.kv_canary.token_oracle.oracle import HashOracle

        server_args = _make_server_args(sampling_backend="token_oracle")
        hook = install_token_oracle_from_env(server_args=server_args, vocab_size=256)
        self.assertIsNotNone(hook)
        self.assertIsInstance(hook.oracle, HashOracle)
        self.assertEqual(hook.oracle.vocab_size, 256)


if __name__ == "__main__":
    unittest.main()
