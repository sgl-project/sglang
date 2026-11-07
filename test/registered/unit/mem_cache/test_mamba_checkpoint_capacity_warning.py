"""Warn about device checkpoint pressure without warning for other pool modes."""

import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_cache_configurator import (
    KVCacheConfigurator,
    _warn_if_mamba_checkpoint_capacity_is_insufficient,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_LOGGER = "sglang.srt.mem_cache.kv_cache_configurator"


class TestMambaCheckpointCapacityWarning(CustomTestCase):
    def test_warns_when_checkpoint_demand_exceeds_pool(self):
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            _warn_if_mamba_checkpoint_capacity_is_insufficient(
                token_capacity=433_919,
                chunked_prefill_size=2_048,
                max_mamba_cache_size=165,
            )
        self.assertEqual(len(logs.output), 1)
        self.assertIn("212 checkpoint(s) for 433919 tokens", logs.output[0])
        self.assertIn("max_mamba_cache_size=165", logs.output[0])
        self.assertIn("--max-mamba-cache-size", logs.output[0])

    def test_rounds_checkpoint_demand_up(self):
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            _warn_if_mamba_checkpoint_capacity_is_insufficient(
                token_capacity=8_193,
                chunked_prefill_size=8_192,
                max_mamba_cache_size=1,
            )
        self.assertIn("2 checkpoint(s)", logs.output[0])

    def test_does_not_warn_when_capacity_is_sufficient(self):
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            _warn_if_mamba_checkpoint_capacity_is_insufficient(
                token_capacity=16_384,
                chunked_prefill_size=8_192,
                max_mamba_cache_size=2,
            )

    def test_does_not_warn_when_chunked_prefill_is_disabled(self):
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            _warn_if_mamba_checkpoint_capacity_is_insufficient(
                token_capacity=433_919,
                chunked_prefill_size=-1,
                max_mamba_cache_size=1,
            )

    def test_does_not_warn_without_a_mamba_pool(self):
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            _warn_if_mamba_checkpoint_capacity_is_insufficient(
                token_capacity=433_919,
                chunked_prefill_size=2_048,
                max_mamba_cache_size=None,
            )

    def test_does_not_warn_for_unconfigured_limits(self):
        for tokens, chunk_size, cache_size in (
            (0, 2048, 1),
            (8192, None, 1),
            (8192, 2048, 0),
        ):
            with (
                self.subTest(
                    tokens=tokens, chunk_size=chunk_size, cache_size=cache_size
                ),
                self.assertNoLogs(_LOGGER, level="WARNING"),
            ):
                _warn_if_mamba_checkpoint_capacity_is_insufficient(
                    token_capacity=tokens,
                    chunked_prefill_size=chunk_size,
                    max_mamba_cache_size=cache_size,
                )

    def test_startup_warning_matches_checkpoint_pool_mode(self):
        """Host-only checkpoints must not be measured against the device pool."""
        configurator = SimpleNamespace(
            model_config=SimpleNamespace(context_len=8192),
            attn_dp_size=1,
            mambaish_config=object(),
            _unified_sized=None,
            _calculate_mamba_ratio=lambda: 1,
        )
        for disabled, strategy, int8_checkpoints, host_only, expected in (
            (False, "extra_buffer", False, False, True),
            (True, "extra_buffer", False, False, False),
            (False, "no_buffer", False, False, False),
            (False, "extra_buffer", True, False, False),
            (False, "extra_buffer", False, True, False),
        ):
            with (
                self.subTest(
                    disabled=disabled,
                    strategy=strategy,
                    int8_checkpoints=int8_checkpoints,
                    host_only=host_only,
                ),
                envs.SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT.override(host_only),
                get_context().override_server_args(
                    max_running_requests=1,
                    chunked_prefill_size=2048,
                    max_mamba_cache_size=2,
                    disable_radix_cache=disabled,
                    mamba_radix_cache_strategy=strategy,
                    enable_int8_mamba_checkpoint=int8_checkpoints,
                    enable_hierarchical_cache=host_only,
                    enable_unified_memory=False,
                ),
            ):
                if expected:
                    with self.assertLogs(_LOGGER, level="WARNING") as logs:
                        max_reqs = KVCacheConfigurator.resolve_max_num_reqs(
                            configurator, 8192
                        )
                    self.assertIn(
                        "checkpoint capacity may be insufficient", logs.output[0]
                    )
                else:
                    with self.assertNoLogs(_LOGGER, level="WARNING"):
                        max_reqs = KVCacheConfigurator.resolve_max_num_reqs(
                            configurator, 8192
                        )
                self.assertEqual(max_reqs, 1)


if __name__ == "__main__":
    unittest.main()
