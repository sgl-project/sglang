"""Unit tests for deterministic FlashInfer CUDA-graph decode launch bounds."""

import unittest

from tvm_ffi import Array

from sglang.srt.layers.attention.flashinfer_backend import (
    _narrow_deterministic_cuda_graph_decode_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _plan_info(
    *,
    padded_batch_size=132,
    total_num_rows=2,
    cta_tile_q=16,
    enable_cuda_graph=1,
    split_kv=0,
    offset_seed=100,
):
    """Build FlashInfer 0.6.14's 15-field PrefillPlanInfo vector."""
    return [
        padded_batch_size,
        total_num_rows,
        offset_seed + 2,
        cta_tile_q,
        offset_seed + 4,
        offset_seed + 5,
        offset_seed + 6,
        offset_seed + 7,
        offset_seed + 8,
        offset_seed + 9,
        offset_seed + 10,
        offset_seed + 11,
        offset_seed + 12,
        enable_cuda_graph,
        split_kv,
    ]


class TestFlashInferDecodePlanLaunchBound(CustomTestCase):
    def test_narrows_mha_and_gqa_to_one_tile_per_request(self):
        for name, num_qo_heads, num_kv_heads in (
            ("mha", 16, 16),
            ("gqa", 16, 2),
        ):
            with self.subTest(name=name):
                plan_info = _plan_info(total_num_rows=4)
                narrowed = _narrow_deterministic_cuda_graph_decode_plan(
                    plan_info,
                    batch_size=4,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                )
                self.assertEqual(narrowed[0], 4)

    def test_group_larger_than_cta_tile_keeps_all_query_tiles(self):
        plan_info = _plan_info(total_num_rows=3)
        narrowed = _narrow_deterministic_cuda_graph_decode_plan(
            plan_info,
            batch_size=3,
            num_qo_heads=32,
            num_kv_heads=1,
        )

        # GQA group size 32 needs two CTA-q tiles per request when CTA_TILE_Q=16.
        self.assertEqual(narrowed[0], 6)

    def test_preserves_native_ffi_array_without_mutating_input(self):
        plan_info = Array(_plan_info())
        original = tuple(plan_info)

        narrowed = _narrow_deterministic_cuda_graph_decode_plan(
            plan_info,
            batch_size=2,
            num_qo_heads=16,
            num_kv_heads=2,
        )

        self.assertIs(type(narrowed), type(plan_info))
        self.assertIsNot(narrowed, plan_info)
        self.assertEqual(tuple(plan_info), original)

    def test_changes_only_padded_batch_size(self):
        plan_info = _plan_info(offset_seed=700)
        narrowed = _narrow_deterministic_cuda_graph_decode_plan(
            plan_info,
            batch_size=2,
            num_qo_heads=16,
            num_kv_heads=2,
        )

        self.assertEqual(narrowed[0], 2)
        self.assertEqual(narrowed[1:], plan_info[1:])

    def test_equal_bound_returns_the_same_object(self):
        plan_info = _plan_info(padded_batch_size=2)
        narrowed = _narrow_deterministic_cuda_graph_decode_plan(
            plan_info,
            batch_size=2,
            num_qo_heads=16,
            num_kv_heads=2,
        )

        self.assertIs(narrowed, plan_info)

    def test_representative_normal_and_fast_plan_vectors(self):
        # The normal capture planner and fast replay planner can produce
        # different workspace offsets, but share the same scheduling ABI.
        for planner, offset_seed in (("normal", 100), ("fast", 1000)):
            with self.subTest(planner=planner):
                plan_info = _plan_info(offset_seed=offset_seed)
                narrowed = _narrow_deterministic_cuda_graph_decode_plan(
                    plan_info,
                    batch_size=2,
                    num_qo_heads=16,
                    num_kv_heads=2,
                )
                self.assertEqual(narrowed[0], 2)
                self.assertEqual(narrowed[1:], plan_info[1:])

    def test_rejects_malformed_plan_abi(self):
        for length in (14, 16):
            with self.subTest(length=length):
                plan_info = _plan_info()
                plan_info = plan_info[:length] if length < 15 else plan_info + [0]
                with self.assertRaisesRegex(RuntimeError, "expected 15 fields"):
                    _narrow_deterministic_cuda_graph_decode_plan(
                        plan_info,
                        batch_size=2,
                        num_qo_heads=16,
                        num_kv_heads=2,
                    )

    def test_rejects_non_target_plan_flags_and_row_count(self):
        cases = (
            (
                "not_cuda_graph",
                _plan_info(enable_cuda_graph=0),
                "expected a CUDA graph plan",
            ),
            (
                "split_kv",
                _plan_info(split_kv=1),
                "expected split-KV to be disabled",
            ),
            (
                "multiple_query_rows",
                _plan_info(total_num_rows=3),
                "expected one query row per request",
            ),
        )
        for name, plan_info, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, message):
                    _narrow_deterministic_cuda_graph_decode_plan(
                        plan_info,
                        batch_size=2,
                        num_qo_heads=16,
                        num_kv_heads=2,
                    )

    def test_rejects_invalid_dimensions(self):
        cases = (
            (
                "batch_size",
                _plan_info(),
                dict(batch_size=0, num_qo_heads=16, num_kv_heads=2),
                "expected a positive value",
            ),
            (
                "qo_heads",
                _plan_info(),
                dict(batch_size=2, num_qo_heads=0, num_kv_heads=2),
                "num_qo_heads and num_kv_heads must be positive",
            ),
            (
                "kv_heads",
                _plan_info(),
                dict(batch_size=2, num_qo_heads=16, num_kv_heads=0),
                "num_qo_heads and num_kv_heads must be positive",
            ),
            (
                "head_divisibility",
                _plan_info(),
                dict(batch_size=2, num_qo_heads=10, num_kv_heads=3),
                r"num_qo_heads \(10\) must be divisible by num_kv_heads \(3\)",
            ),
            (
                "zero_cta_tile",
                _plan_info(cta_tile_q=0),
                dict(batch_size=2, num_qo_heads=16, num_kv_heads=2),
                "CTA q tile size must be positive",
            ),
            (
                "negative_cta_tile",
                _plan_info(cta_tile_q=-16),
                dict(batch_size=2, num_qo_heads=16, num_kv_heads=2),
                "CTA q tile size must be positive",
            ),
        )
        for name, plan_info, kwargs, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, message):
                    _narrow_deterministic_cuda_graph_decode_plan(plan_info, **kwargs)

    def test_rejects_attempted_enlargement(self):
        plan_info = _plan_info(padded_batch_size=1)
        with self.assertRaisesRegex(RuntimeError, "narrowing would enlarge"):
            _narrow_deterministic_cuda_graph_decode_plan(
                plan_info,
                batch_size=2,
                num_qo_heads=16,
                num_kv_heads=2,
            )


if __name__ == "__main__":
    unittest.main()
