from __future__ import annotations

import unittest

from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-large")


class TestSelfUnitConfig(CustomTestCase):
    def test_canary_config_requires_explicit_from_env_fields(self) -> None:
        """Verify production config fields stay explicit at construction."""
        with self.assertRaises(TypeError):
            CanaryConfig(mode=CanaryMode.RAISE)

    def test_perturb_config_requires_explicit_from_env_fields(self) -> None:
        """Verify perturb config fields stay explicit at construction."""
        with self.assertRaises(TypeError):
            PerturbConfig(req_to_token_prob=0.0)


if __name__ == "__main__":
    unittest.main()
