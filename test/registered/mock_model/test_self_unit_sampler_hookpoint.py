"""install_oracle_sampler registration into sglang's sampler-backend registry.

Instantiating the registered _OracleSampler factory requires a live distributed (TP) group
plus a populated global ServerArgs, so the forward-path behavior of _OracleSampler is covered
by the e2e harness rather than this unit file. Here we only assert the registration-side
contract: the backend name shows up in the registry / choice set, and second install replaces
the factory with one bound to the new oracle.
"""

from __future__ import annotations

import os
import unittest

os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = "1"

from sglang.srt.kv_canary.token_oracle.oracle import HashOracle
from sglang.srt.kv_canary.token_oracle.sampler import install_oracle_sampler
from sglang.srt.layers.sampler import _CUSTOM_SAMPLER_FACTORIES
from sglang.srt.server_args import SAMPLING_BACKEND_CHOICES
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=60, suite="extra-a-test-1-gpu-small-amd")


class TestInstallOracleSampler(CustomTestCase):
    def test_install_oracle_sampler_twice_returns_distinct_hooks_with_replaced_oracle(
        self,
    ) -> None:
        """Verify reinstalling the oracle sampler replaces the registered factory."""
        oracle_a = HashOracle(vocab_size=100)
        oracle_b = HashOracle(vocab_size=100)

        hook_a = install_oracle_sampler(oracle=oracle_a)
        self.assertIn("token_oracle", _CUSTOM_SAMPLER_FACTORIES)
        self.assertIn("token_oracle", SAMPLING_BACKEND_CHOICES)
        factory_a = _CUSTOM_SAMPLER_FACTORIES["token_oracle"]
        self.assertIs(hook_a.oracle, oracle_a)

        hook_b = install_oracle_sampler(oracle=oracle_b)
        factory_b = _CUSTOM_SAMPLER_FACTORIES["token_oracle"]
        self.assertIs(hook_b.oracle, oracle_b)
        self.assertIsNot(hook_a, hook_b)
        self.assertIsNot(factory_a, factory_b)


if __name__ == "__main__":
    unittest.main()
