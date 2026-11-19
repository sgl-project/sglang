"""K3 MLA target verify captures one graph per AITER verify kernel and picks it on the host."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.graph_variants import (
    K3MlaVerifyGraphVariants,
    create_k3_mla_verify_graph_variants,
    k3_mla_verify_hk_threshold,
    k3_mla_verify_max_kv_len,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestK3MlaVerifyGraphVariants(CustomTestCase):
    def test_max_kv_len_adds_draft_tokens(self):
        batch = SimpleNamespace(seq_lens_cpu=torch.tensor([5, 900, 17]))
        self.assertEqual(k3_mla_verify_max_kv_len(batch, 8), 908)

    def test_missing_host_lengths_pick_hk_without_a_device_read(self):
        variants = K3MlaVerifyGraphVariants(num_draft_tokens=8)
        # A device max would sync; seq_lens is never read.
        batch = SimpleNamespace(batch_size=2, seq_lens_cpu=None, seq_lens=None)
        self.assertIsNone(k3_mla_verify_max_kv_len(batch, 8))
        self.assertEqual(variants.select(batch), "hk")

    def test_select_uses_batch_size_and_max_kv_len(self):
        variants = K3MlaVerifyGraphVariants(num_draft_tokens=8)
        self.assertEqual(variants.capture_labels, ("gluon", "hk"))
        short = SimpleNamespace(batch_size=2, seq_lens_cpu=torch.tensor([100, 2000]))
        long = SimpleNamespace(batch_size=2, seq_lens_cpu=torch.tensor([100, 20000]))
        self.assertEqual(variants.select(short), "gluon")
        self.assertEqual(variants.select(long), "hk")

    def test_threshold_boundaries(self):
        variants = K3MlaVerifyGraphVariants(num_draft_tokens=8)
        for bs in (1, 2, 3, 4, 8, 9, 12, 16, 24, 64, 96):
            thresh = k3_mla_verify_hk_threshold(bs)
            self.assertGreater(thresh, 0)
            at = -(-thresh // bs)  # smallest max_kv_len with bs * len >= thresh
            for max_kv_len, want in ((at - 1, "gluon"), (at, "hk")):
                lens = torch.full((bs,), max_kv_len - 8)
                batch = SimpleNamespace(batch_size=bs, seq_lens_cpu=lens)
                self.assertEqual(variants.select(batch), want, (bs, max_kv_len))

    def test_factory_only_for_k3_target_verify(self):
        k3 = SimpleNamespace(use_k3_mla_verify=True, num_draft_tokens=8)
        hybrid = SimpleNamespace(full_attn_backend=k3)
        for backend in (k3, hybrid):
            variants = create_k3_mla_verify_graph_variants(
                backend, ForwardMode.TARGET_VERIFY
            )
            self.assertEqual(variants, K3MlaVerifyGraphVariants(8))
        self.assertIsNone(create_k3_mla_verify_graph_variants(k3, ForwardMode.DECODE))
        forced = SimpleNamespace(use_k3_mla_verify=False, num_draft_tokens=8)
        self.assertIsNone(
            create_k3_mla_verify_graph_variants(forced, ForwardMode.TARGET_VERIFY)
        )
        self.assertIsNone(
            create_k3_mla_verify_graph_variants(
                SimpleNamespace(), ForwardMode.TARGET_VERIFY
            )
        )


if __name__ == "__main__":
    unittest.main()
