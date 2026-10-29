"""CPU-only tests for the hybrid Mamba checkpoint-capacity warning."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.kv_cache_configurator import (
    KVCacheConfigurator,
    _warn_if_mamba_checkpoint_capacity_is_insufficient,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMambaCheckpointCapacityWarning(CustomTestCase):
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

    def test_does_not_warn_for_unconfigured_limits(self):
        for tokens, chunk_size, cache_size in (
            (0, 2048, 1),
            (8192, None, 1),
            (8192, 2048, 0),
        ):
            with self.subTest(
                tokens=tokens, chunk_size=chunk_size, cache_size=cache_size
            ):
                warning = self._warn(
                    token_capacity=tokens,
                    chunked_prefill_size=chunk_size,
                    max_mamba_cache_size=cache_size,
                )
                warning.assert_not_called()

    def test_startup_warning_matches_checkpoint_pool_mode(self):
        configurator = SimpleNamespace(
            model_config=SimpleNamespace(context_len=8192),
            attn_dp_size=1,
            mambaish_config=object(),
            _calculate_mamba_ratio=lambda: 1,
        )
        schedule = SimpleNamespace(
            max_running_requests=1,
            chunked_prefill_size=2048,
            max_mamba_cache_size=2,
        )
        module = "sglang.srt.mem_cache.kv_cache_configurator"
        for disabled, extra_buffer, int8_checkpoints, expected in (
            (False, True, False, True),
            (True, True, False, False),
            (False, False, False, False),
            (False, True, True, False),
        ):
            with self.subTest(
                disabled=disabled,
                extra_buffer=extra_buffer,
                int8_checkpoints=int8_checkpoints,
            ):
                memory = SimpleNamespace(disable_radix_cache=disabled)
                execution = SimpleNamespace(
                    mamba=SimpleNamespace(
                        enable_mamba_extra_buffer=extra_buffer,
                        enable_int8_mamba_checkpoint=int8_checkpoints,
                    )
                )
                with (
                    mock.patch(f"{module}.get_memory", return_value=memory),
                    mock.patch(f"{module}.get_exec", return_value=execution),
                    mock.patch(f"{module}.get_schedule", return_value=schedule),
                    mock.patch(f"{module}.logger.warning") as warning,
                ):
                    max_reqs = KVCacheConfigurator.resolve_max_num_reqs(
                        configurator, 8192
                    )
                self.assertEqual(max_reqs, 1)
                self.assertEqual(warning.call_count, int(expected))


if __name__ == "__main__":
    unittest.main()
