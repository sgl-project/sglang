"""install_oracle_sampler registration into sglang's sampler-backend registry.

Instantiating the registered _OracleSampler factory requires a live distributed (TP) group
plus a populated global ServerArgs, so the forward-path behavior of _OracleSampler is covered
by the e2e harness rather than this unit file. Here we only assert the registration-side
contract: the backend name shows up in the registry / choice set, and second install replaces
the factory with one bound to the new oracle.
"""

from __future__ import annotations

import unittest

from sglang.srt.kv_canary.token_oracle.oracle import HashOracle
from sglang.srt.kv_canary.token_oracle.sampler import install_oracle_sampler
from sglang.srt.layers.sampler import _CUSTOM_SAMPLER_FACTORIES
from sglang.srt.server_args import SAMPLING_BACKEND_CHOICES
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")


class TestInstallOracleSampler(CustomTestCase):
    def test_install_oracle_sampler_twice_returns_distinct_hooks_with_replaced_oracle(
        self,
    ) -> None:
        oracle_a = HashOracle(seed=10, vocab_size=100)
        oracle_b = HashOracle(seed=99, vocab_size=100)

        hook_a = install_oracle_sampler(oracle=oracle_a)
        self.assertIn("oracle", _CUSTOM_SAMPLER_FACTORIES)
        self.assertIn("oracle", SAMPLING_BACKEND_CHOICES)
        factory_a = _CUSTOM_SAMPLER_FACTORIES["oracle"]
        self.assertIs(hook_a.oracle, oracle_a)

        hook_b = install_oracle_sampler(oracle=oracle_b)
        factory_b = _CUSTOM_SAMPLER_FACTORIES["oracle"]
        self.assertIs(hook_b.oracle, oracle_b)
        self.assertIsNot(hook_a, hook_b)
        self.assertIsNot(factory_a, factory_b)


if __name__ == "__main__":
    unittest.main()
