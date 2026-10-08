"""An opt-in derived head cache must never silently survive online updates."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.kernels.ops.gemm.sm120_lossless_bf16_head import lossless_bf16_head_enabled
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class TestLosslessHeadUpdateGuard(CustomTestCase):
    def test_all_public_updates_reject_before_mutation(self):
        fake = SimpleNamespace(_assert_weight_cache_inactive=Mock(), get_model=Mock())
        calls = (
            (WeightUpdater.update_weights_from_disk, ("unused", "auto")),
            (WeightUpdater.update_weights_from_distributed, ([], [], [], "unused")),
            (WeightUpdater.update_weights_from_tensor, ([],)),
            (WeightUpdater.update_weights_from_ipc, (None,)),
        )
        with patch.dict(os.environ, {"SGLANG_SM120_LOSSLESS_BF16_HEAD": "1"}):
            lossless_bf16_head_enabled.cache_clear()
            try:
                for method, args in calls:
                    with self.subTest(method=method.__name__):
                        success, error = method(fake, *args)
                        self.assertFalse(success)
                        self.assertIn("SGLANG_SM120_LOSSLESS_BF16_HEAD", error)
                fake.get_model.assert_not_called()
            finally:
                lossless_bf16_head_enabled.cache_clear()


if __name__ == "__main__":
    unittest.main()
