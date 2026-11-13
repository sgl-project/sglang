"""Dispatch, head padding and configuration guards for sparse FlashMLA prefill."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashMLASparseHeadPadding(CustomTestCase):
    def run_case(
        self,
        heads,
        dim=576,
        enabled=True,
        topk=128,
        allow_small_heads=True,
        kv_dtype=torch.bfloat16,
    ):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        q = torch.randn(3, heads, dim, dtype=torch.bfloat16)
        sink = torch.randn(heads)
        indices = torch.zeros(3, topk, dtype=torch.int32)
        lengths = torch.full((3,), topk, dtype=torch.int32)
        calls = []

        def small(padded, kv, ids, scale, active, sinks, lens):
            calls.append("small")
            self.assertEqual(active, 8 if heads <= 8 else 32)
            self.assertEqual(padded.shape, (3, 64, dim))
            torch.testing.assert_close(padded[:, :heads], q)
            self.assertEqual(torch.count_nonzero(padded[:, heads:]).item(), 0)
            torch.testing.assert_close(sinks[:heads], sink)
            self.assertEqual(torch.count_nonzero(sinks[heads:]).item(), 0)
            self.assertIs(lens, lengths)
            torch.testing.assert_close(ids, indices.unsqueeze(1))
            return torch.ones(3, active, 512), None, None

        def native(**kwargs):
            calls.append("native")
            padded = kwargs["q"].shape[1]
            self.assertEqual(padded, 128)
            torch.testing.assert_close(kwargs["q"][:, :heads], q)
            self.assertIs(kwargs["topk_length"], lengths)
            return torch.ones(3, padded, 512), None, None

        backend = SimpleNamespace(
            device_sm_major=10,
            sparse_prefill_small_heads=enabled,
            _sink_pad_cache={},
            _call_flash_mla_sparse_fwd_small_heads=small,
            _call_flash_mla_sparse_fwd=native,
        )
        out = DeepseekSparseAttnBackend._forward_flashmla_sparse(
            backend,
            q,
            torch.empty(1, dtype=kv_dtype),
            512,
            indices,
            0.5,
            lengths,
            sink,
            allow_small_heads=allow_small_heads,
        )
        self.assertEqual(out.shape, (3, heads, 512))
        return calls

    def test_all_small_head_counts_dispatch_automatically(self):
        for heads in range(1, 33):
            for dim in (512, 576):
                with self.subTest(heads=heads, dim=dim):
                    self.assertEqual(self.run_case(heads, dim), ["small"])

    def test_native_fallback_preserves_padding(self):
        for heads in (8, 16, 32, 64, 128):
            with self.subTest(heads=heads):
                self.assertEqual(self.run_case(heads, enabled=False), ["native"])
        for heads in (64, 128):
            self.assertEqual(self.run_case(heads), ["native"])
        self.assertEqual(self.run_case(8, enabled=False), ["native"])
        for topk in (64, 130):
            self.assertEqual(self.run_case(8, topk=topk), ["native"])

    def test_decode_and_kv_dtype_stay_native(self):
        self.assertEqual(self.run_case(8, allow_small_heads=False), ["native"])
        self.assertEqual(self.run_case(8, kv_dtype=torch.float16), ["native"])

    def test_startup_configuration(self):
        from sglang.srt.layers.attention.dsa_backend import _small_heads_prefill_enabled

        args = [10, torch.bfloat16, 8, 512, 64, "flashmla_sparse"]
        self.assertTrue(_small_heads_prefill_enabled(*args))
        args[-1] = "flashmla_auto"
        self.assertTrue(_small_heads_prefill_enabled(*args))
        for index, value in [
            (0, 9),
            (1, torch.float16),
            (2, 64),
            (3, 256),
            (4, 128),
            (5, "trtllm"),
        ]:
            invalid = list(args)
            invalid[index] = value
            self.assertFalse(_small_heads_prefill_enabled(*invalid))


if __name__ == "__main__":
    unittest.main()
