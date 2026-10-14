"""Chunk-plan properties for the chunked GEMM+AR overlap.

Every TP rank recomputes the plan independently from (M, env knobs); the
collectives only line up if the plan is a total, deterministic cover of the
row range. A malformed plan (gap, overlap, undersized chunk) would corrupt
the all-reduce silently, so pin the invariants.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import chunked_gemm_ar
from sglang.srt.layers.chunked_gemm_ar import compute_chunk_plan
from sglang.srt.lora.layers import RowParallelLinearWithLoRA
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ALIGN = 256
MIN_ROWS = 512


class TestChunkPlan(CustomTestCase):
    def _check_cover(self, plan, m):
        self.assertEqual(plan[0][0], 0)
        self.assertEqual(plan[-1][1], m)
        for (_, hi), (lo, _) in zip(plan[:-1], plan[1:]):
            self.assertEqual(hi, lo)
        for lo, hi in plan:
            self.assertGreaterEqual(hi - lo, MIN_ROWS)
        # Internal boundaries stay GEMM-tile aligned.
        for lo, hi in plan[:-1]:
            self.assertEqual(hi % ALIGN, 0)

    def test_geometric_plan_25k(self):
        plan = compute_chunk_plan(25000, 4, 0.36)
        self.assertIsNotNone(plan)
        self._check_cover(plan, 25000)
        self.assertEqual(len(plan), 4)
        sizes = [hi - lo for lo, hi in plan]
        # First chunk is the largest (it is the only GEMM with no AR to hide)
        # and later chunks decay.
        self.assertEqual(max(sizes), sizes[0])
        self.assertTrue(all(a >= b for a, b in zip(sizes, sizes[1:])), sizes)

    def test_equal_split(self):
        m = 16384
        plan = compute_chunk_plan(m, 4, 1.0)
        self.assertIsNotNone(plan)
        self._check_cover(plan, m)
        sizes = [hi - lo for lo, hi in plan]
        self.assertEqual(len(plan), 4)
        for s in sizes:
            self.assertAlmostEqual(s, m / 4, delta=ALIGN)

    def test_small_m_returns_none(self):
        self.assertIsNone(compute_chunk_plan(2 * MIN_ROWS - 1, 4, 0.36))
        self.assertIsNone(compute_chunk_plan(0, 4, 0.36))
        self.assertIsNone(compute_chunk_plan(300, 4, 1.0))

    def test_single_chunk_returns_none(self):
        self.assertIsNone(compute_chunk_plan(25000, 1, 0.36))
        self.assertIsNone(compute_chunk_plan(25000, 0, 0.36))

    def test_chunks_shrink_with_m(self):
        # Small-but-eligible M keeps every surviving chunk >= MIN_ROWS even
        # when the geometric tail would round below it.
        for m in (2048, 4096, 8192, 8193, 12345):
            plan = compute_chunk_plan(m, 4, 0.36)
            if plan is None:
                continue
            self._check_cover(plan, m)
            self.assertGreaterEqual(len(plan), 2)

    def test_ratio_clamped(self):
        # Degenerate ratios must not produce empty or runaway chunks.
        for ratio in (0.0, -1.0, 5.0):
            plan = compute_chunk_plan(25000, 4, ratio)
            self.assertIsNotNone(plan)
            self._check_cover(plan, 25000)


class _FakeCudaTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data, False)

    @property
    def is_cuda(self):
        return True


class TestPipelineSymmetricGuard(CustomTestCase):
    # Regression: the guard imported dp_attention.is_allocation_symmetric,
    # which is always True under plain TP and made the pipeline unreachable.

    def _layer(self, apply_into):
        return SimpleNamespace(
            quant_method=SimpleNamespace(apply_into=apply_into),
            output_size=8,
            tp_group=object(),
        )

    def _run(self, symm_enabled):
        x = _FakeCudaTensor(torch.zeros(2048, 16))
        apply_into = MagicMock()
        group = SimpleNamespace(world_size=1, device_group=None, unique_name="tp")
        work = SimpleNamespace(wait=lambda: None)
        with (
            patch.object(
                chunked_gemm_ar,
                "_knobs",
                return_value=SimpleNamespace(
                    min_tokens=1, num_chunks=2, ratio=0.5, reserve_sms=0
                ),
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch(
                "sglang.srt.layers.moe.utils.should_skip_mlp_all_reduce",
                return_value=False,
            ),
            patch(
                "sglang.srt.runtime_context.get_forward",
                return_value=SimpleNamespace(sp_active=False),
            ),
            patch(
                "sglang.srt.distributed.device_communicators.pynccl_allocator.is_symmetric_memory_enabled",
                return_value=symm_enabled,
            ),
            patch(
                "sglang.srt.distributed.utils.get_group_rank_size",
                return_value=(0, 1),
            ),
            patch("torch.distributed.all_reduce", return_value=work) as all_reduce,
        ):
            result = chunked_gemm_ar._try_forward_pipeline(
                self._layer(apply_into), x, None, None, group
            )
        return result, apply_into, all_reduce

    def test_plain_tp_without_symmetric_memory_engages(self):
        result, apply_into, all_reduce = self._run(symm_enabled=False)
        self.assertIsNotNone(result)
        self.assertEqual(tuple(result.shape), (2048, 8))
        self.assertEqual(apply_into.call_count, 2)
        self.assertEqual(all_reduce.call_count, 2)

    def test_symmetric_memory_bails(self):
        result, apply_into, all_reduce = self._run(symm_enabled=True)
        self.assertIsNone(result)
        apply_into.assert_not_called()
        all_reduce.assert_not_called()


class TestInlineEntryEligibility(CustomTestCase):
    def test_lora_wrapped_projection_keeps_its_regular_forward(self):
        wrapped = RowParallelLinearWithLoRA.__new__(RowParallelLinearWithLoRA)
        torch.nn.Module.__init__(wrapped)
        with patch.object(chunked_gemm_ar, "_ENABLED", True):
            self.assertIsNone(
                chunked_gemm_ar.try_forward_chunked_gemm_ar_inline(
                    layer=wrapped,
                    input_parallel=torch.zeros(16, 8),
                    forward_batch=None,
                    reduce_on_attn_tp=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
