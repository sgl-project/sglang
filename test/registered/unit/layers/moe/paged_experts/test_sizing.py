"""Unit tests for srt/layers/moe/paged_experts/sizing.py"""

import unittest

from sglang.srt.layers.moe.paged_experts.sizing import (
    compute_num_resident_experts,
    compute_window_experts,
    kv_reserve_bytes_mha,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Qwen3-30B-A3B (GQA): 48 layers, 4 KV heads, head_dim 128, fp16 KV -> 98304 B/token.
_QWEN3_KV = dict(
    num_layers=48,
    num_kv_heads=4,
    head_dim=128,
    kv_dtype_bytes=2,
    max_running_requests=1,
    context_length=2048,
)


class TestPagedExpertsSizing(CustomTestCase):
    def test_kv_cell_matches_qwen3(self):
        cell = kv_reserve_bytes_mha(**_QWEN3_KV)
        self.assertEqual(int(cell / 2048), 98304)  # per-token KV bytes

    def test_reproduces_measured_resident_K(self):
        # Reproduces a measured Qwen3-30B-int4 boot: free 6.66 GB, mem_fraction 0.85, K=25/128.
        K = compute_num_resident_experts(
            free_vram_bytes=6.66e9,
            mem_fraction=0.85,
            nonexpert_bytes=2.5e9,
            kv_reserve_bytes=kv_reserve_bytes_mha(**_QWEN3_KV),
            moe_layers=48,
            per_expert_layer_bytes=2.45e6,
            top_k=8,
            num_experts=128,
        )
        self.assertEqual(K, 25)

    def test_clamps_to_topk_and_E(self):
        common = dict(
            mem_fraction=0.85,
            nonexpert_bytes=2.5e9,
            kv_reserve_bytes=0,
            moe_layers=48,
            per_expert_layer_bytes=2.45e6,
            top_k=8,
            num_experts=128,
        )
        self.assertEqual(compute_num_resident_experts(free_vram_bytes=1e9, **common), 8)
        self.assertEqual(
            compute_num_resident_experts(free_vram_bytes=200e9, **common), 128
        )

    def test_kv_reserve_clamped_to_physical(self):
        # Regression: a worst-case KV reserve (high --max-running-requests x full ctx) can exceed the
        # whole VRAM budget. It must NOT drive K negative/garbage — the K-slot pool and KV share one
        # budget, so the reserve is clamped to what's left after a top_k floor.
        common = dict(
            free_vram_bytes=16e9,
            mem_fraction=0.85,
            nonexpert_bytes=2.5e9,
            moe_layers=48,
            per_expert_layer_bytes=2.45e6,
            top_k=8,
            num_experts=128,
        )
        # 128 reqs x 2560 ctx ~ 32 GB on a 16 GB card -> old code: budget negative -> K floored to top_k.
        huge = kv_reserve_bytes_mha(
            num_layers=48,
            num_kv_heads=4,
            head_dim=128,
            kv_dtype_bytes=2,
            max_running_requests=128,
            context_length=2560,
        )
        K_worstcase = compute_num_resident_experts(kv_reserve_bytes=huge, **common)
        self.assertEqual(K_worstcase, 8)  # clamped to top_k, never negative
        # The resolver now reserves a SINGLE-STREAM context instead — which frees K far above top_k on
        # the same card (this is the bug fix: K no longer collapses under --max-running-requests).
        single_stream = kv_reserve_bytes_mha(
            num_layers=48,
            num_kv_heads=4,
            head_dim=128,
            kv_dtype_bytes=2,
            max_running_requests=1,
            context_length=2560,
        )
        K_ss = compute_num_resident_experts(kv_reserve_bytes=single_stream, **common)
        self.assertGreater(K_ss, 8)  # ~92/128 — not starved by a phantom KV reserve


class TestComputeWindowExperts(CustomTestCase):
    # 48 moe layers, ~0.47 GB per expert across all layers (bf16-30B-ish: 9.72 MB/layer).
    _common = dict(moe_layers=48, per_expert_layer_bytes=9.72e6, num_experts=128)

    def test_whole_store_fits_returns_zero(self):
        # budget covers all 128 experts -> 0 (full pin, no window)
        self.assertEqual(
            compute_window_experts(pin_budget_bytes=200e9, **self._common), 0
        )

    def test_windowed_when_budget_is_partial(self):
        # ~0.467 GB/expert; a 24 GB budget fits ~51 experts
        per_expert_pool = 48 * 9.72e6
        self.assertEqual(
            compute_window_experts(pin_budget_bytes=24e9, **self._common),
            int(24e9 / per_expert_pool),
        )

    def test_clamps_to_at_least_one(self):
        # tiny (but nonzero) budget -> at least one hot expert, never 0-by-underflow
        self.assertEqual(
            compute_window_experts(pin_budget_bytes=1.0, **self._common), 1
        )

    def test_zero_budget_is_one(self):
        # 0 budget still yields a minimal window (1), not a full pin (0)
        self.assertEqual(
            compute_window_experts(pin_budget_bytes=0.0, **self._common), 1
        )

    def test_at_capacity_collapses_to_full_pin(self):
        # a budget for exactly E experts is a full pin -> 0
        per_expert_pool = 48 * 9.72e6
        self.assertEqual(
            compute_window_experts(
                pin_budget_bytes=128 * per_expert_pool, **self._common
            ),
            0,
        )

    def test_degenerate_geometry_is_zero(self):
        self.assertEqual(
            compute_window_experts(
                pin_budget_bytes=10e9,
                moe_layers=0,
                per_expert_layer_bytes=0,
                num_experts=128,
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
