import logging
import unittest
from unittest.mock import Mock

from sglang.srt.kv_canary.perturb import real_kv_used
from sglang.srt.kv_canary.perturb.config import PerturbConfig, TargetGroupKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCanaryPerturb(CustomTestCase):
    def test_real_kv_used_logs_when_target_group_has_no_real_kv_sources(self) -> None:
        """Verify real KV used perturbation logs when the target group has no sources."""
        config = PerturbConfig(
            req_to_token_prob=0.0,
            real_kv_used_prob=1.0,
            real_kv_unused_cache_prob=0.0,
            real_kv_post_forward_prob=0.0,
            target_group_kind=TargetGroupKind.FULL,
            warmup_steps=0,
        )
        warmup_gate = Mock()
        warmup_gate.is_in_warmup.return_value = False

        # Empty buffer_groups means pick_target_group returns None, so run() takes
        # the early-return branch before any slot is picked.
        with self.assertLogs(real_kv_used.logger.name, level=logging.INFO) as logs:
            real_kv_used.run(
                maybe_inaccurate_forward_batch=Mock(),
                config=config,
                req_to_token_pool=Mock(),
                buffer_groups=(),
                swa_window_size=0,
                warmup_gate=warmup_gate,
            )

        self.assertIn(
            "kv_canary perturb real_kv_used: skipped because no target group with "
            "real_kv_sources_k matched target_group_kind=full",
            "\n".join(logs.output),
        )


if __name__ == "__main__":
    unittest.main()
