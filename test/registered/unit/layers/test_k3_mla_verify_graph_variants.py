"""K3 MLA target verify captures one graph per AITER verify kernel and picks it on the host."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.graph_variants import (
    K3MlaVerifyGraphVariants,
    create_k3_mla_verify_graph_variants,
    k3_mla_verify_max_kv_len,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_aiter_mla_verify(calls):
    mod = ModuleType("aiter.mla_verify")

    def select_mla_verify_kernel(batch_size, max_kv_len):
        calls.append((batch_size, max_kv_len))
        return "hk" if batch_size * max_kv_len >= 30000 else "gluon"

    mod.select_mla_verify_kernel = select_mla_verify_kernel
    return mod


class TestK3MlaVerifyGraphVariants(CustomTestCase):
    def test_max_kv_len_adds_draft_tokens(self):
        batch = SimpleNamespace(seq_lens_cpu=torch.tensor([5, 900, 17]))
        self.assertEqual(k3_mla_verify_max_kv_len(batch, 8), 908)
        batch = SimpleNamespace(seq_lens_cpu=None, seq_lens=torch.tensor([3, 40]))
        self.assertEqual(k3_mla_verify_max_kv_len(batch, 8), 48)

    def test_select_uses_batch_size_and_max_kv_len(self):
        calls = []
        variants = K3MlaVerifyGraphVariants(num_draft_tokens=8)
        self.assertEqual(variants.capture_labels, ("gluon", "hk"))
        with patch.dict(
            sys.modules, {"aiter.mla_verify": _fake_aiter_mla_verify(calls)}
        ):
            short = SimpleNamespace(
                batch_size=2, seq_lens_cpu=torch.tensor([100, 2000])
            )
            long = SimpleNamespace(
                batch_size=2, seq_lens_cpu=torch.tensor([100, 20000])
            )
            self.assertEqual(variants.select(short), "gluon")
            self.assertEqual(variants.select(long), "hk")
        self.assertEqual(calls, [(2, 2008), (2, 20008)])

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
