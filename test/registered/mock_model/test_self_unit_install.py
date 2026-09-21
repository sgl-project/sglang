from __future__ import annotations

import os
import unittest

os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = "1"

from sglang.srt.kv_canary.token_oracle.install import install_token_oracle_from_env
from sglang.srt.kv_canary.token_oracle.oracle import HashOracle
from sglang.srt.layers.sampler import _CUSTOM_SAMPLER_FACTORIES
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=60, suite="extra-a-test-1-gpu-small-amd")


def _publish(case, *, sampling_backend: str) -> None:
    """The gate reads the published config, so the test publishes one."""
    override = get_context().override_server_args(sampling_backend=sampling_backend)
    override.install()
    case.addCleanup(override.restore)


class TestInstallTokenOracleFromEnv(CustomTestCase):
    def test_install_token_oracle_from_env_disabled_returns_none(self) -> None:
        """Verify server-arg-disabled token oracle installation (sampling_backend != 'token_oracle') returns no TokenOracleManager."""
        _publish(self, sampling_backend="auto")
        hook = install_token_oracle_from_env(vocab_size=1000)
        self.assertIsNone(hook)

    def test_install_token_oracle_from_env_enabled_registers_oracle_backend(
        self,
    ) -> None:
        """Verify token oracle installation via sampling_backend='token_oracle' registers the oracle backend."""
        _publish(self, sampling_backend="token_oracle")
        hook = install_token_oracle_from_env(vocab_size=512)
        self.assertIsNotNone(hook)
        self.assertIn("token_oracle", _CUSTOM_SAMPLER_FACTORIES)

    def test_install_token_oracle_from_env_enabled_returns_hook_with_hash_oracle(
        self,
    ) -> None:
        """Verify token oracle installation via sampling_backend='token_oracle' returns a TokenOracleManager wrapping a HashOracle."""
        _publish(self, sampling_backend="token_oracle")
        hook = install_token_oracle_from_env(vocab_size=256)
        self.assertIsNotNone(hook)
        self.assertIsInstance(hook.oracle, HashOracle)
        self.assertEqual(hook.oracle.vocab_size, 256)


if __name__ == "__main__":
    unittest.main()
