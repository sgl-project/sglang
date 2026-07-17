from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from contextlib import ExitStack
from typing import Optional
from unittest.mock import MagicMock, patch

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.mem_cache.kv_cache_builder import build_kv_cache
from sglang.test.test_utils import CustomTestCase

_MODULE = "sglang.srt.mem_cache.kv_cache_builder"


class _StopAfterGuards(Exception):
    pass


def _build(
    *,
    strip_thinking_cache: bool,
    is_hybrid_swa: bool,
    full_tokens_per_layer: Optional[int],
) -> None:
    server_args = MagicMock()
    server_args.strip_thinking_cache = strip_thinking_cache
    server_args.disaggregation_decode_enable_radix_cache = False
    server_args.disable_radix_cache = False

    tp_worker = MagicMock()
    tp_worker.is_hybrid_swa = is_hybrid_swa
    tp_worker.sliding_window_size = 128
    tp_worker.get_tokens_per_layer_info.return_value = (full_tokens_per_layer, 64)
    tp_worker.get_memory_pool.side_effect = _StopAfterGuards()

    model_config = MagicMock()
    model_config.is_multimodal = False

    with ExitStack() as stack:
        stack.enter_context(
            patch(f"{_MODULE}.get_resolved_model_impl", return_value=ModelImpl.SGLANG)
        )
        stack.enter_context(
            patch(f"{_MODULE}.linear_attn_model_spec", return_value=None)
        )
        for name in (
            "hybrid_gdn_config",
            "mamba2_config",
            "kimi_linear_config",
            "hybrid_lightning_config",
        ):
            stack.enter_context(patch(f"{_MODULE}.{name}", return_value=None))
        stack.enter_context(patch(f"{_MODULE}.is_deepseek_dsa", return_value=False))

        build_kv_cache(
            server_args=server_args,
            model_config=model_config,
            tp_worker=tp_worker,
            page_size=1,
            spec_algorithm=MagicMock(),
            attn_tp_cpu_group=MagicMock(),
            tp_cpu_group=MagicMock(),
            attn_cp_cpu_group=MagicMock(),
            enable_metrics=False,
            enable_kv_cache_events=False,
            ps=MagicMock(),
            tp_group=MagicMock(),
            pp_group=MagicMock(),
            enable_hierarchical_cache=False,
        )


class TestStripThinkingCacheAllSwaGuard(CustomTestCase):
    def test_strip_thinking_cache_with_an_all_swa_model_raises(self):
        """All-SWA eviction puts the freed hole above the stripped committed length."""
        with self.assertRaises(ValueError) as ctx:
            _build(
                strip_thinking_cache=True, is_hybrid_swa=True, full_tokens_per_layer=0
            )
        self.assertIn("strip-thinking-cache", str(ctx.exception))

    def test_strip_thinking_cache_with_a_hybrid_swa_model_is_allowed(self):
        """Hybrid SWA keeps a full-attention pool that is never evicted early."""
        with self.assertRaises(_StopAfterGuards):
            _build(
                strip_thinking_cache=True, is_hybrid_swa=True, full_tokens_per_layer=32
            )

    def test_strip_thinking_cache_without_swa_is_allowed(self):
        """Non-SWA models never punch a hole below the committed length."""
        with self.assertRaises(_StopAfterGuards):
            _build(
                strip_thinking_cache=True,
                is_hybrid_swa=False,
                full_tokens_per_layer=None,
            )

    def test_an_all_swa_model_without_strip_thinking_cache_is_allowed(self):
        """All-SWA alone is fine: the hole stays below the committed length."""
        with self.assertRaises(_StopAfterGuards):
            _build(
                strip_thinking_cache=False, is_hybrid_swa=True, full_tokens_per_layer=0
            )


if __name__ == "__main__":
    unittest.main()
