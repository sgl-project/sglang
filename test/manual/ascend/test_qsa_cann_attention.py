"""Small-memory QSA CANN tests; no checkpoint or server required.

Run on 910C with CANN 9.0 / torch-npu 2.10:
    python -m unittest discover -s test/manual/ascend -p test_qsa_cann_attention.py -v

Set ASCEND_RT_VISIBLE_DEVICES before launching to select the test device.
"""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.qsa import native_npu
from sglang.srt.layers.attention.qsa.kernel import (
    qsa_sparse_attention,
    qsa_sparse_attention_reference,
    torch_expand_qsa_block_indices,
)

try:
    import torch_npu
except ImportError:
    torch_npu = None


class TestQsaNativeCPU(unittest.TestCase):
    def test_cpu_keeps_reference(self):
        q = torch.randn(2, 16, 256)
        k = torch.randn(8, 2, 256)
        v = torch.randn_like(k)
        slots = torch.tensor([[0, 7, -1], [3, 1, 5]], dtype=torch.int32)
        with patch.dict(os.environ, {"SGLANG_NPU_QSA_NATIVE_PREFILL": "1"}):
            with patch.object(native_npu, "try_qsa_native_prefill") as native:
                actual = qsa_sparse_attention(q, k, v, slots, allow_npu_prefill=True)
                native.assert_not_called()
        torch.testing.assert_close(
            actual, qsa_sparse_attention_reference(q, k, v, slots)
        )
        self.assertIsNone(native_npu.try_qsa_native_prefill(q, k, v, slots))


@unittest.skipUnless(torch_npu is not None, "torch-npu required")
class TestQsaNativeNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU required")
        if not torch.npu.get_device_name(0).startswith("Ascend910_93"):
            raise unittest.SkipTest("Validated native path requires Ascend 910C")
        if not hasattr(torch_npu, "npu_sparse_flash_attention"):
            raise unittest.SkipTest("CANN SparseFlashAttention required")

    def setUp(self):
        torch.manual_seed(1234)

    def inputs(self, rows=12, capacity=4096, heads=16, kv_heads=2):
        # Generate on CPU so the FP32 reference consumes exactly the BF16 data
        # sent to the NPU, independent of the device random-number generator.
        return (
            torch.randn(rows, heads, 256, dtype=torch.bfloat16),
            torch.randn(capacity, kv_heads, 256, dtype=torch.bfloat16),
            torch.randn(capacity, kv_heads, 256, dtype=torch.bfloat16),
        )

    def check_case(self, q, k, v, slots, scale=None):
        expected = qsa_sparse_attention_reference(
            q.float(), k.float(), v.float(), slots, scale
        )
        actual = native_npu.try_qsa_native_prefill(
            q.npu(), k.npu(), v.npu(), slots.npu(), scale
        )
        self.assertIsNotNone(actual, "Test must execute the CANN path")
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.shape, q.shape)
        actual = actual.cpu().float()
        self.assertTrue(torch.isfinite(actual).all())
        # BF16 native attention is not bitwise equal to an FP32 softmax.
        torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)
        relative_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
        self.assertLess(float(relative_l2), 0.008)
        empty = (slots < 0).all(dim=1)
        self.assertTrue((actual[empty] == 0).all())
        print(
            f"rows={q.shape[0]} width={slots.shape[1]} scale={scale} rel_l2={relative_l2:.6f}"
        )

    def test_random_slots_padding_and_scale(self):
        q, k, v = self.inputs()
        counts = [0, 1, 2, 3, 4, 5, 63, 64, 65, 128, 2048, 2051]
        for width in (1, 63, 64, 65, 2051):
            slots = torch.full((len(counts), width), -1, dtype=torch.int32)
            for row, count in enumerate(counts):
                count = min(count, width)
                # Physical addresses may be unordered; -1 padding is trailing.
                slots[row, :count] = torch.randperm(k.shape[0])[:count].int()
            for scale in (None, 0.03125):
                with self.subTest(width=width, scale=scale):
                    self.check_case(q, k, v, slots, scale)

    def test_causal_tail_and_request_mapping(self):
        visible = torch.tensor([1, 2, 3, 4, 65, 66, 67, 68])
        q, k, v = self.inputs(rows=len(visible), capacity=512)
        blocks = torch.full((len(visible), 512), -1, dtype=torch.int32)
        for row, length in enumerate(visible):
            count = int(length) // 4
            blocks[row, :count] = torch.randperm(count).int()
        logical = torch_expand_qsa_block_indices(
            blocks, visible - 1, visible, compress_ratio=4, token_topk=2048
        )
        # Two requests share a prefix but use different physical suffix pages.
        table = torch.randperm(512).reshape(2, 256).int()
        table[1, :16] = table[0, :16]
        request = torch.arange(len(visible)) % 2
        slots = table[request[:, None], logical.clamp_min(0).long()]
        slots = torch.where(logical >= 0, slots, -1)
        self.check_case(q, k, v, slots)
        # Reused physical slots must read the current cache contents.
        self.check_case(q, -k, v * 2, slots)

    def test_flash_next_local_head_shapes(self):
        # These are per-rank tensor shapes, not a distributed TP/model test.
        for heads, kv_heads in ((24, 2), (12, 1), (6, 1), (3, 1)):
            for rows, width in ((1, 3), (257, 2051)):
                with self.subTest(heads=heads, kv_heads=kv_heads, rows=rows):
                    q, k, v = self.inputs(rows=rows, heads=heads, kv_heads=kv_heads)
                    slots = torch.stack(
                        [torch.randperm(k.shape[0])[:width] for _ in range(rows)]
                    ).int()
                    if rows > 1:
                        slots[0] = -1
                        slots[1, 2:] = -1
                    self.check_case(q, k, v, slots)

    def test_noncontiguous_cache_and_queries(self):
        q, k, v = self.inputs(rows=4, capacity=64)
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
        k = k.transpose(0, 1).contiguous().transpose(0, 1)
        v = v.transpose(0, 1).contiguous().transpose(0, 1)
        slots = torch.tensor(
            [[7, 0, -1], [63, 2, 5], [-1, -1, -1], [9, 1, 8]], dtype=torch.int32
        )
        self.check_case(q, k, v, slots)

    def test_empty_inputs(self):
        q, k, v = (x.npu() for x in self.inputs(rows=2, capacity=8))
        for rows, width in ((0, 3), (2, 0), (2, 3)):
            slots = torch.full((rows, width), -1, dtype=torch.int32, device=q.device)
            out = native_npu.try_qsa_native_prefill(q[:rows], k, v, slots)
            self.assertIsNotNone(out)
            self.assertEqual(out.shape, q[:rows].shape)
            self.assertTrue((out == 0).all().item())

    def test_long_prefill_sampled_reference(self):
        q, k, v = self.inputs(rows=7810, capacity=65536)
        slots = (
            (torch.arange(7810)[:, None] * 7 + torch.arange(2051)[None, :] * 31) % 65536
        ).int()
        qn, kn, vn, sn = (x.npu() for x in (q, k, v, slots))
        torch.npu.reset_peak_memory_stats()
        output = native_npu.try_qsa_native_prefill(qn, kn, vn, sn)
        self.assertIsNotNone(output)
        self.assertTrue(torch.isfinite(output).all().item())
        sample = torch.tensor([0, 1, 63, 255, 1024, 4095, 7808, 7809])
        expected = qsa_sparse_attention_reference(
            q[sample].float(), k.float(), v.float(), slots[sample]
        )
        actual = output.cpu()[sample].float()
        torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)
        relative_l2 = (actual - expected).norm() / expected.norm()
        self.assertLess(float(relative_l2), 0.008)
        peak_mib = torch.npu.max_memory_allocated() / 2**20
        print(
            f"rows=7810 sampled_rel_l2={relative_l2:.6f} peak_allocated_mib={peak_mib:.1f}"
        )

    def test_unsupported_inputs_and_capture_fall_back(self):
        q, k, v = (x.npu() for x in self.inputs(rows=2, capacity=64))
        slots = torch.tensor([[63, 1], [2, -1]], dtype=torch.int32).npu()
        for args in (
            (q.float(), k, v, slots),
            (q[:, :8], k, v, slots),
            (q, k, v, slots.long()),
        ):
            self.assertIsNone(native_npu.try_qsa_native_prefill(*args))
        with patch.object(native_npu, "_MAX_CACHE_EXTENT", 32):
            self.assertIsNone(native_npu.try_qsa_native_prefill(q, k, v, slots))
        with patch.object(torch.npu, "is_current_stream_capturing", return_value=True):
            self.assertIsNone(native_npu.try_qsa_native_prefill(q, k, v, slots))
        # Native CANN stops at interior padding. Preserve the reference's
        # more general semantics by falling back before invoking the operator.
        holes = torch.tensor([[1, -1, 3], [-1, 2, -1]], dtype=torch.int32).npu()
        self.assertIsNone(native_npu.try_qsa_native_prefill(q, k, v, holes))
        with patch.dict(os.environ, {"SGLANG_NPU_QSA_NATIVE_PREFILL": "1"}):
            actual = qsa_sparse_attention(q, k, v, holes, allow_npu_prefill=True)
        expected = qsa_sparse_attention_reference(q, k, v, holes)
        torch.testing.assert_close(actual, expected)

    def test_prefill_dispatch_is_opt_in(self):
        q, k, v = (x.npu() for x in self.inputs(rows=2, capacity=8))
        slots = torch.tensor([[0, 7], [1, 3]], dtype=torch.int32).npu()
        for enabled, prefill in (("0", True), ("1", False)):
            with patch.dict(os.environ, {"SGLANG_NPU_QSA_NATIVE_PREFILL": enabled}):
                with patch.object(native_npu, "try_qsa_native_prefill") as native:
                    qsa_sparse_attention(q, k, v, slots, allow_npu_prefill=prefill)
                    native.assert_not_called()
        with patch.dict(os.environ, {"SGLANG_NPU_QSA_NATIVE_PREFILL": "1"}):
            with patch.object(
                native_npu,
                "try_qsa_native_prefill",
                wraps=native_npu.try_qsa_native_prefill,
            ) as native:
                output = qsa_sparse_attention(q, k, v, slots, allow_npu_prefill=True)
                native.assert_called_once()
            self.assertEqual(output.shape, q.shape)


if __name__ == "__main__":
    unittest.main()
