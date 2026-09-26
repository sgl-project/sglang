"""CPU-only tests for the hybrid Mamba checkpoint-capacity warning."""

import unittest
from unittest import mock

from sglang.srt.mem_cache.kv_cache_configurator import (
    _warn_if_mamba_checkpoint_capacity_is_insufficient,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMambaCheckpointCapacityWarning(unittest.TestCase):
    @staticmethod
    def _warn(*, token_capacity, chunked_prefill_size, max_mamba_cache_size):
        with mock.patch(
            "sglang.srt.mem_cache.kv_cache_configurator.logger.warning"
        ) as warning:
            _warn_if_mamba_checkpoint_capacity_is_insufficient(
                token_capacity=token_capacity,
                chunked_prefill_size=chunked_prefill_size,
                max_mamba_cache_size=max_mamba_cache_size,
            )
        return warning

    def test_warns_when_checkpoint_demand_exceeds_pool(self):
        warning = self._warn(
            token_capacity=433_919,
            chunked_prefill_size=2_048,
            max_mamba_cache_size=165,
        )

        warning.assert_called_once()
        message = warning.call_args.args[0]
        self.assertIn("checkpoint capacity may be insufficient", message)
        self.assertEqual(warning.call_args.args[1:], (212, 433_919, 2_048, 165))

    def test_rounds_checkpoint_demand_up(self):
        warning = self._warn(
            token_capacity=8_193,
            chunked_prefill_size=8_192,
            max_mamba_cache_size=1,
        )

        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[1], 2)

    def test_does_not_warn_when_capacity_is_sufficient(self):
        warning = self._warn(
            token_capacity=16_384,
            chunked_prefill_size=8_192,
            max_mamba_cache_size=2,
        )

        warning.assert_not_called()

    def test_does_not_warn_when_chunked_prefill_is_disabled(self):
        warning = self._warn(
            token_capacity=433_919,
            chunked_prefill_size=-1,
            max_mamba_cache_size=1,
        )

        warning.assert_not_called()

    def test_does_not_warn_without_a_mamba_pool(self):
        warning = self._warn(
            token_capacity=433_919,
            chunked_prefill_size=2_048,
            max_mamba_cache_size=None,
        )

        warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
