"""CPU-only tests for model-runner KV cache capacity validation."""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_runner(
    token_capacity,
    context_len=4096,
    *,
    is_draft_worker=False,
    prefill_only_disable_kv_cache=False,
    is_hybrid_swa=False,
    full_max_total_num_tokens=None,
    sliding_window_size=None,
    dcp_size=1,
):
    return SimpleNamespace(
        is_draft_worker=is_draft_worker,
        server_args=SimpleNamespace(
            prefill_only_disable_kv_cache=prefill_only_disable_kv_cache
        ),
        model_config=SimpleNamespace(context_len=context_len),
        max_token_pool_size=token_capacity,
        is_hybrid_swa=is_hybrid_swa,
        full_max_total_num_tokens=full_max_total_num_tokens,
        sliding_window_size=sliding_window_size,
        dcp_size=dcp_size,
    )


class TestKVCacheCapacity(CustomTestCase):
    def test_rejects_capacity_below_context_length(self):
        runner = _make_runner(token_capacity=2048)

        with self.assertRaisesRegex(
            ValueError,
            "context length \\(4096\\).*only 2048 tokens are available",
        ):
            ModelRunnerKVCacheMixin._validate_kv_cache_capacity(runner)

    def test_accepts_capacity_at_or_above_context_length(self):
        for token_capacity in (4096, 8192):
            with self.subTest(token_capacity=token_capacity):
                runner = _make_runner(token_capacity=token_capacity)
                ModelRunnerKVCacheMixin._validate_kv_cache_capacity(runner)

    def test_pure_swa_requires_only_sliding_window_capacity(self):
        runner = _make_runner(
            token_capacity=4096,
            context_len=131072,
            is_hybrid_swa=True,
            full_max_total_num_tokens=0,
            sliding_window_size=4096,
        )
        ModelRunnerKVCacheMixin._validate_kv_cache_capacity(runner)

    def test_pure_swa_without_window_uses_context_length(self):
        runner = _make_runner(
            token_capacity=4096,
            context_len=131072,
            is_hybrid_swa=True,
            full_max_total_num_tokens=0,
        )

        with self.assertRaisesRegex(ValueError, "at least 131072 tokens"):
            ModelRunnerKVCacheMixin._validate_kv_cache_capacity(runner)

    def test_dcp_uses_logical_capacity_across_ranks(self):
        runner = _make_runner(token_capacity=1024, context_len=4096, dcp_size=4)
        ModelRunnerKVCacheMixin._validate_kv_cache_capacity(runner)

    def test_skips_draft_and_no_kv_workers(self):
        for kwargs in (
            {"is_draft_worker": True},
            {"prefill_only_disable_kv_cache": True},
        ):
            with self.subTest(**kwargs):
                runner = _make_runner(token_capacity=2048, **kwargs)
                ModelRunnerKVCacheMixin._validate_kv_cache_capacity(runner)


if __name__ == "__main__":
    unittest.main()
