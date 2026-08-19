"""CPU tests for AITER unified target-verify GQA packing.

This is the AITER counterpart of Triton grouped-head shared-KV verify
(PR #34517). AITER serving must not call ``verify_shared_kv_fwd``; it keeps
the true KV-head count so ``unified_attention`` packs Q heads against one KV
load.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.aiter_verify_gqa import pack_unified_verify_kv
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

PAGE = 16
HEAD_DIM = 256
H_Q = 16
H_KV = 1


def _kv(num_blocks=4, num_kv_heads=H_KV):
    k = torch.zeros(num_blocks, PAGE, num_kv_heads, HEAD_DIM)
    v = torch.zeros(num_blocks, PAGE, num_kv_heads, HEAD_DIM)
    k[..., 0, 0] = 1.0
    return k, v


class TestAiterVerifyGqaPack(CustomTestCase):
    def test_default_keeps_true_kv_head_count(self):
        k, v = _kv()
        packed_k, packed_v = pack_unified_verify_kv(
            k, v, tp_k_head_num=H_KV, tp_q_head_num=H_Q, gqa_pack=True
        )
        self.assertEqual(tuple(packed_k.shape), (4, PAGE, H_KV, HEAD_DIM))
        self.assertEqual(tuple(packed_v.shape), (4, PAGE, H_KV, HEAD_DIM))
        self.assertIs(packed_k, k)
        self.assertIs(packed_v, v)

    def test_opt_out_expands_to_fake_mha(self):
        k, v = _kv()
        packed_k, packed_v = pack_unified_verify_kv(
            k, v, tp_k_head_num=H_KV, tp_q_head_num=H_Q, gqa_pack=False
        )
        self.assertEqual(tuple(packed_k.shape), (4, PAGE, H_Q, HEAD_DIM))
        self.assertEqual(tuple(packed_v.shape), (4, PAGE, H_Q, HEAD_DIM))
        self.assertEqual(packed_k.stride(2), 0)
        self.assertEqual(packed_k[0, 0, 0, 0].item(), packed_k[0, 0, H_Q - 1, 0].item())

    def test_does_not_expand_multiple_local_kv_heads(self):
        k, v = _kv(num_kv_heads=2)
        packed_k, packed_v = pack_unified_verify_kv(
            k, v, tp_k_head_num=2, tp_q_head_num=H_Q, gqa_pack=False
        )
        self.assertEqual(tuple(packed_k.shape), (4, PAGE, 2, HEAD_DIM))
        self.assertIs(packed_k, k)
        self.assertIs(packed_v, v)

    def test_env_default_is_on(self):
        self.assertTrue(envs.SGLANG_ENABLE_AITER_VERIFY_GQA_PACK.get())

    def test_env_override_restores_expand(self):
        k, v = _kv()
        with envs.SGLANG_ENABLE_AITER_VERIFY_GQA_PACK.override(False):
            packed_k, _ = pack_unified_verify_kv(
                k,
                v,
                tp_k_head_num=H_KV,
                tp_q_head_num=H_Q,
                gqa_pack=envs.SGLANG_ENABLE_AITER_VERIFY_GQA_PACK.get(),
            )
        self.assertEqual(packed_k.shape[2], H_Q)

    def test_backend_wires_helper_not_triton_kernel(self):
        from pathlib import Path

        import sglang.srt.layers.attention.aiter_verify_gqa as gqa

        source = (Path(gqa.__file__).with_name("aiter_backend.py")).read_text(
            encoding="utf-8"
        )
        self.assertIn("pack_unified_verify_kv", source)
        self.assertNotIn("verify_shared_kv_fwd", source)


if __name__ == "__main__":
    unittest.main()
